#include "stencil.hpp"
#include "stencil.def.h"

CProxy_CodeGenCache codegen_proxy;

CodeGenCache::CodeGenCache()
{
    lock = CmiCreateLock();
}

CodeGenCache::~CodeGenCache()
{
    CmiDestroyLock(lock);
}

compute_fun_t CodeGenCache::lookup(size_t hash)
{
    return cache[hash];
}

void CodeGenCache::receive(int size, char *cmd, CProxy_Stencil stencil_proxy_)
{
    start_time = CmiWallTimer();
    int num_kernels = extract<int>(cmd);
    CkPrintf("Received %i kernels\n", num_kernels);
    for (int i = 0; i < num_kernels; i++)
    {
        Kernel *knl = build_kernel(cmd);
        kernels[knl->kernel_id] = knl;
        generate_kernel(knl, thisIndex);
        cache[knl->hash] = load_kernel(knl->hash, thisIndex);
    }

    stencil_proxy = stencil_proxy_;
    saved_dag_size = extract<int>(cmd);
    saved_dag = new char[saved_dag_size];
    memcpy(saved_dag, cmd, saved_dag_size);

    int done = 1;
    CkCallback cb(CkReductionTarget(CodeGenCache, send_dag), thisProxy[0]);
    contribute(sizeof(int), (void *)&done, CkReduction::sum_int, cb);
}

void CodeGenCache::send_dag(int done)
{
    CkPrintf("Kernels compiled in %f seconds\n", CmiWallTimer() - start_time);
    stencil_proxy.receive_dag(saved_dag_size, saved_dag);
    delete[] saved_dag;
}

void CodeGenCache::operation_done(double start)
{
    double runtime = CmiWallTimer() - start;
    CkPrintf("Execution took %f seconds\n", runtime);
    CcsSendDelayedReply(operation_reply, sizeof(double), &runtime);
    // CkExit();
}

void CodeGenCache::gather(int name, int index_x, int index_y, int local_dim, int num_chares, int data_size, float *data)
{
    int total_dim = num_chares * local_dim;

    // gather data from all sources
    int start_x = index_x * local_dim;
    int start_y = index_y * local_dim;

    int stop_x = start_x + local_dim;
    int stop_y = start_y + local_dim;

    // DEBUG_PRINT("DEBUG> (%i, %i) > (%i, %i) > (%i, %i)\n", index_x, index_y, start_x, stop_x, start_y, stop_y);

    auto it = gathered_arrays.find(name);
    float *all_data;
    if (it == gathered_arrays.end())
    {
        all_data = (float *)malloc(sizeof(float) * total_dim * total_dim);

        // copy data to all_data
        for (int j = 0; j < local_dim; j++)
            for (int i = 0; i < local_dim; i++)
            {
                int local_index = j * local_dim + i;
                int global_index = (j + start_y) * total_dim + (i + start_x);
                all_data[global_index] = data[local_index];
            }

        gathered_arrays[name] = std::make_pair(1, all_data);
    }
    else
    {
        all_data = it->second.second;

        // copy data to all_data
        for (int j = 0; j < local_dim; j++)
            for (int i = 0; i < local_dim; i++)
            {
                int local_index = j * local_dim + i;
                int global_index = (j + start_y) * total_dim + (i + start_x);
                all_data[global_index] = data[local_index];
            }

        it->second.first++;
    }

    if (gathered_arrays[name].first == num_chares * num_chares)
    {
        CcsSendDelayedReply(fetch_reply, sizeof(float) * total_dim * total_dim, all_data);
        free(all_data);
        gathered_arrays.erase(name);
    }
}

Stencil::Stencil(int num_chares_x, int num_chares_y)
    : num_nbrs(0)
{
    char dummy;
    DAG_DONE = &dummy;

    index[0] = thisIndex.x;
    index[1] = thisIndex.y;

    num_chares[0] = num_chares_x;
    num_chares[1] = num_chares_y;

    for (int i = 0; i < 2; i++)
    {
        if (index[i] > 0)
        {
            num_nbrs++;
            boundary[2 * i] = false;
        }
        else
            boundary[2 * i] = true;
        if (index[i] < num_chares[i] - 1)
        {
            num_nbrs++;
            boundary[2 * i + 1] = false;
        }
        else
            boundary[2 * i + 1] = true;
    }

    CUdevice cuDevice;
    CUcontext cuContext;
    hapiCheck(cudaFree(0));

    hapiCheck(cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, 0));
    hapiCheck(cudaStreamCreateWithPriority(&comm_stream, cudaStreamDefault, -1));

    hapiCheck(cudaEventCreateWithFlags(&compute_event, cudaEventDisableTiming));
    hapiCheck(cudaEventCreateWithFlags(&comm_event, cudaEventDisableTiming));

    /*for (int i = 0; i < num_nbrs; i++)
    {
        hapiCheck(cudaMalloc((void**)&send_ghosts[i], sizeof(double) * ghost_size));
        hapiCheck(cudaMalloc((void**)&recv_ghosts[i], sizeof(double) * ghost_size));
    }

    thisProxy(thisIndex.x, thisIndex.y, thisIndex.z).start();*/
}

Stencil::Stencil(CkMigrateMessage *m) {}

Stencil::~Stencil()
{
    // delete_cache();
    // for (int i = 0; i < num_nbrs; i++)
    //{
    //     hapiCheck(cudaFree(send_ghosts[i]));
    //     hapiCheck(cudaFree(recv_ghosts[i]));
    // }
    // free(send_ghosts);
    // free(recv_ghosts);
    for (auto &entry : arrays)
    {
        Array *array = entry.second;
        delete array;
    }
}

void Stencil::gather(int name)
{
    Array *array = arrays[name];
    float *data = array->to_host();
    float *local_data = array->get_local(data);
    // for (int i = 0; i < array->total_size; i++) {
    //     printf("%f ", data[i]);
    // }
    // printf("\n");
    codegen_proxy[0].gather(name, index[0], index[1], array->local_shape[0], num_chares[0], array->total_local_size, local_data);
    delete[] data;
    delete[] local_data;
}

void Stencil::receive_dag(int size, char *graph)
{
    start_time = CmiWallTimer();
    std::vector<DAGNode *> goals = build_dag(graph, node_cache, codegen_proxy.ckLocalBranch()->kernels, ghost_info);
    // DEBUG_PRINT("PE %i> Num goals = %i\n", CkMyPe(), goals.size());
    if (thisIndex.x == 0 && thisIndex.y == 0)
        CkPrintf("Building DAG took %f seconds\n", CmiWallTimer() - start_time);
    start_time = CmiWallTimer();
    for (auto &goal : goals)
    {
        bool is_done = traverse_dag(goal);
        if (!is_done)
            goals_waiting.insert(goal->node_id);
    }
    // wait for all futures

    for (const auto &goal : goals_waiting)
    {
        if (thisIndex.x == 0 && thisIndex.y == 0)
            DEBUG_PRINT("Goal ID: %i\n", goal);
    }

    if (goals_waiting.empty())
    {
        DEBUG_PRINT("PE %i> No goals waiting\n", CkMyPe());
        cudaStreamSynchronize(compute_stream);
        cudaStreamSynchronize(comm_stream);
        CkCallback cb(CkReductionTarget(CodeGenCache, operation_done), codegen_proxy[0]);
        contribute(sizeof(double), (void *)&start_time, CkReduction::min_double, cb);
    }
}

void Stencil::mark_done(DAGNode *node)
{
    // if (thisIndex.x == 0 && thisIndex.y == 0)
    DEBUG_PRINT("(%i, %i)> Marking node %i as done\n", thisIndex.x, thisIndex.y, node->node_id);
    node->done = true;
    // if (thisIndex.x == 0 && thisIndex.y == 0)
    //{
    //     for (auto& waiting_node : node->waiting)
    //     {
    //         DEBUG_PRINT("Node %i> Waiting node %i, %p\n", node->node_id, waiting_node->node_id, waiting_node);
    //     }
    // }
    while (!node->waiting.empty())
    {
        // DEBUG_PRINT("PE %i> Traversing waiting node %i\n", CkMyPe(), node->waiting.back()->node_id);
        DAGNode *waiting = *node->waiting.begin();
        node->waiting.erase(node->waiting.begin());
        waiting->status = NodeStatus::UnVisited;
        traverse_dag(waiting);
    }

    if (goals_waiting.find(node->node_id) != goals_waiting.end())
    {
        goals_waiting.erase(node->node_id);
        // send a message to the codegen cache
        // DEBUG_PRINT("goals waiting size = %i\n", goals_waiting.size());
        if (goals_waiting.empty())
        {
            DEBUG_PRINT("PE %i> No goals waiting\n", CkMyPe());
            cudaStreamSynchronize(compute_stream);
            cudaStreamSynchronize(comm_stream);
            int done = 1;
            CkCallback cb(CkReductionTarget(CodeGenCache, operation_done), codegen_proxy[0]);
            contribute(sizeof(double), (void *)&start_time, CkReduction::min_double, cb);
        }
    }
}

bool Stencil::traverse_dag(DAGNode *node)
{
    if (node->status == NodeStatus::Visited)
        return node->done;

    if (auto *array_node = dynamic_cast<ArrayDAGNode *>(node))
    {
        // create the array
        create_array(array_node->name, array_node->shape);
        node->status = NodeStatus::Visited;
        mark_done(node);
        return node->done;
    }

    auto *kernel_node = dynamic_cast<KernelDAGNode *>(node);
    node->status = NodeStatus::Visited;

    // DEBUG_PRINT("Traversing kernel node %i\n", kernel_node->kernel_id);

    bool dep_done = true;
    // DEBUG_PRINT("PE %i> Kernel node %i; dependencies = %i\n", CkMyPe(), kernel_node->node_id, kernel_node->dependencies.size());
    for (auto &dep : kernel_node->dependencies)
    {
        // traverse the dependencies
        if (!traverse_dag(dep))
        {
            // if the dependency is not done, return false
            dep->waiting.insert(node);
            // if (thisIndex.x == 0 && thisIndex.y == 0 && dep->node_id == 7)
            // DEBUG_PRINT("PE %i> Adding node %i to waiting list of %i\n", CkMyPe(), node->node_id, dep->node_id);
            dep_done = false;
        }
    }

    if (!dep_done)
    {
        // dependency is not done
        return false;
    }

    // execute this node and return a future
    // if (thisIndex.x == 0 && thisIndex.y == 0)
    bool ghost_exchange_needed = false;

    for (int i = 0; i < kernel_node->inputs.size(); i++)
    {
        int ghost_depth = arrays[kernel_node->inputs[i]]->ghost_depth;
        if (ghost_depth > 0)
        {
            ghost_exchange_needed = true;
            break;
        }
    }

    if (ghost_exchange_needed)
    {
        DEBUG_PRINT("(%i, %i)> Send ghosts called for %i\n", thisIndex.x, thisIndex.y, kernel_node->node_id);
        send_ghost_data(kernel_node);
    }
    else{
        DEBUG_PRINT("(%i, %i)> Execute Kernel called for %i\n", thisIndex.x, thisIndex.y, kernel_node->node_id);
        execute_kernel(kernel_node);
    }

    return kernel_node->done;
}

void Stencil::receive_ghost_data(int node_id, int name, int dir, int &size, float *&buf, CkDeviceBufferPost *device_post)
{
    Array *array = arrays[name];
    // if (thisIndex.x == 0 && thisIndex.y == 0)
    DEBUG_PRINT("(%i, %i)> Post %i receiving ghost data for array %i in dir %i, ptr = %p\n",
                thisIndex.x, thisIndex.y, node_id, name, dir, array->recv_ghost_buffers[dir]);
    buf = array->recv_ghost_buffers[dir];
    // cudaMalloc((void**)&buf, sizeof(float) * array->ghost_size);
    device_post[0].cuda_stream = comm_stream;
}

void Stencil::receive_ghost_data(int node_id, int name, int dir, int size, float *buf)
{
    Array *array = arrays[name];
    // call unpacking kernel

    // if (thisIndex.x == 0 && thisIndex.y == 0)
    DEBUG_PRINT("(%i, %i)> Receiving ghost data for %i array %i in dir %i, ptr = %p, %p\n",
                thisIndex.x, thisIndex.y, node_id, name, dir, array->recv_ghost_buffers[dir], buf);

    switch (dir)
    {
    case NORTH:
    {
        int startx = array->ghost_depth;
        int stopx = startx + array->local_shape[1];
        int starty = array->local_shape[0] + array->ghost_depth;
        int stopy = starty + array->ghost_depth;
        invoke_ns_unpacking_kernel(array->data, array->recv_ghost_buffers[dir], array->ghost_depth,
                                   startx, stopx, starty, stopy, array->strides[0], array->local_shape[0],
                                   comm_stream);
        break;
    }

    case SOUTH:
    {
        int startx = array->ghost_depth;
        int stopx = startx + array->local_shape[1];
        int starty = 0;
        int stopy = array->ghost_depth;
        invoke_ns_unpacking_kernel(array->data, array->recv_ghost_buffers[dir], array->ghost_depth,
                                   startx, stopx, starty, stopy, array->strides[0], array->local_shape[0],
                                   comm_stream);
        break;
    }

    case EAST:
    {
        int startx = array->local_shape[1] + array->ghost_depth;
        int stopx = startx + array->ghost_depth;
        int starty = array->ghost_depth;
        int stopy = starty + array->local_shape[0];
        invoke_ew_unpacking_kernel(array->data, array->recv_ghost_buffers[dir], array->ghost_depth,
                                   startx, stopx, starty, stopy, array->strides[0], array->local_shape[0],
                                   comm_stream);
        break;
    }

    case WEST:
    {
        int startx = 0;
        int stopx = array->ghost_depth;
        int starty = array->ghost_depth;
        int stopy = starty + array->local_shape[0];
        invoke_ew_unpacking_kernel(array->data, array->recv_ghost_buffers[dir], array->ghost_depth,
                                   startx, stopx, starty, stopy, array->strides[0], array->local_shape[0],
                                   comm_stream);
        break;
    }

    default:
        break;
    }

    // cudaStreamSynchronize(comm_stream);

    auto it = ghost_counts.find(node_id);
    if (it == ghost_counts.end())
    {
        ghost_counts[node_id] = 1;
    }
    else
    {
        it->second++;
        check_ghost_completion(node_id);
    }
}

void Stencil::check_ghost_completion(int node_id)
{
    if (ghosts_expected.find(node_id) != ghosts_expected.end())
    {
        auto it = ghost_counts.find(node_id);
        // if (it != ghost_counts.end() && thisIndex.x == 0 && thisIndex.y == 0)
        //     DEBUG_PRINT("(%i, %i)> Checking ghost completion for node %i, expected = %i, received = %i\n",
        //         thisIndex.x, thisIndex.y, node_id, ghosts_expected[node_id], it->second);
        if (ghosts_expected[node_id] == 0 || (it != ghost_counts.end() && it->second == ghosts_expected[node_id]))
        {
            // all ghosts received
            ghosts_expected.erase(node_id);
            ghost_counts.erase(node_id);
            handle_ghost_completion(node_id);
        }
    }
}

void Stencil::ghost_done(KernelCallbackMsg *msg)
{
    DEBUG_PRINT("(%i, %i)> Ghost done for node %i\n", thisIndex.x, thisIndex.y, msg->node_id);
    DAGNode *node = node_cache[msg->node_id];
    execute_kernel(static_cast<KernelDAGNode *>(node));
}

void Stencil::handle_ghost_completion(int node_id)
{
    // DEBUG_PRINT("(%i, %i)> Handling ghost completion for node %i\n", thisIndex.x, thisIndex.y, node_id);
    for (int i = 0; i < ghost_arrays[node_id].size(); i++)
    {
        int input = ghost_arrays[node_id][i];
        Array *array = arrays[input];
        array->ghost_generation = array->generation;
        array->exchange_in_progress = false;
    }
    ghost_arrays.erase(node_id);
    // CkCallback* cb = new CkCallback(CkIndex_Stencil::ghost_done(NULL), thisProxy[thisIndex]);
    // KernelCallbackMsg* msg = new KernelCallbackMsg(node_id);
    // hapiAddCallback(comm_stream, cb, msg);
    DAGNode *node = node_cache[node_id];
    execute_kernel(static_cast<KernelDAGNode *>(node));
}

void Stencil::send_ghost_data(KernelDAGNode *node)
{
    hapiCheck(cudaEventRecord(compute_event, compute_stream));
    hapiCheck(cudaStreamWaitEvent(comm_stream, compute_event, 0));

    ghost_arrays[node->node_id] = std::vector<int>();
    // data transfer required for this node
    for (int i = 0; i < node->inputs.size(); i++)
    {
        int input = node->inputs[i];
        Array *array = arrays[input];
        // if (thisIndex.x == 0 && thisIndex.y == 0)
        // DEBUG_PRINT("(%i, %i)> array %i, generation = %i, ghost_generation = %i, in progress = %d\n",
        //     thisIndex.x, thisIndex.y, input, array->generation, array->ghost_generation, array->exchange_in_progress);
        if (array->generation > array->ghost_generation && !array->exchange_in_progress && array->ghost_depth > 0)
        {
            // DEBUG_PRINT("PE %i> Sending ghost data for array %i\n", CkMyPe(), input);
            //  ghost data is stale
            //  send the ghost data to the neighbors
            array->exchange_in_progress = true;
            ghost_arrays[node->node_id].push_back(input);

            if (!boundary[NORTH])
            {
                int startx = array->ghost_depth;
                int stopx = startx + array->local_shape[1];
                int starty = array->local_shape[0];
                int stopy = starty + array->ghost_depth;
                invoke_ns_packing_kernel(array->data, array->send_ghost_buffers[NORTH], array->ghost_depth,
                                         startx, stopx, starty, stopy, array->strides[0], array->local_shape[0],
                                         comm_stream);
                // send ghost to north chare
                // if (thisIndex.x == 0 && thisIndex.y == 0)
                // DEBUG_PRINT("PE %i> Sending ghost data %i to dir %i\n", CkMyPe(), input, NORTH);
                thisProxy(thisIndex.x, thisIndex.y + 1).receive_ghost_data(node->node_id, input, SOUTH, array->ghost_size, CkDeviceBuffer(array->send_ghost_buffers[NORTH], comm_stream));
            }

            if (!boundary[SOUTH])
            {
                int startx = array->ghost_depth;
                int stopx = startx + array->local_shape[1];
                int starty = array->ghost_depth;
                int stopy = starty + array->ghost_depth;
                invoke_ns_packing_kernel(array->data, array->send_ghost_buffers[SOUTH], array->ghost_depth,
                                         startx, stopx, starty, stopy, array->strides[0], array->local_shape[0],
                                         comm_stream);
                // send ghost to south chare
                // if (thisIndex.x == 0 && thisIndex.y == 0)
                // DEBUG_PRINT("PE %i> Sending ghost data %i to dir %i\n", CkMyPe(), input, SOUTH);
                thisProxy(thisIndex.x, thisIndex.y - 1).receive_ghost_data(node->node_id, input, NORTH, array->ghost_size, CkDeviceBuffer(array->send_ghost_buffers[SOUTH], comm_stream));
            }

            if (!boundary[EAST])
            {
                int startx = array->local_shape[1];
                int stopx = startx + array->ghost_depth;
                int starty = array->ghost_depth;
                int stopy = starty + array->local_shape[0];
                invoke_ew_packing_kernel(array->data, array->send_ghost_buffers[EAST], array->ghost_depth,
                                         startx, stopx, starty, stopy, array->strides[0], array->local_shape[0],
                                         comm_stream);
                // send ghost to east chare
                // if (thisIndex.x == 0 && thisIndex.y == 0)
                // DEBUG_PRINT("PE %i> Sending ghost data %i to dir %i\n", CkMyPe(), input, EAST);
                thisProxy(thisIndex.x + 1, thisIndex.y).receive_ghost_data(node->node_id, input, WEST, array->ghost_size, CkDeviceBuffer(array->send_ghost_buffers[EAST], comm_stream));
            }

            if (!boundary[WEST])
            {
                int startx = array->ghost_depth;
                int stopx = startx + array->ghost_depth;
                int starty = array->ghost_depth;
                int stopy = starty + array->local_shape[0];
                invoke_ew_packing_kernel(array->data, array->send_ghost_buffers[WEST], array->ghost_depth,
                                         startx, stopx, starty, stopy, array->strides[0], array->local_shape[0],
                                         comm_stream);
                // send ghost to west chare
                // if (thisIndex.x == 0 && thisIndex.y == 0)
                // DEBUG_PRINT("PE %i> Sending ghost data %i to dir %i\n", CkMyPe(), input, WEST);
                thisProxy(thisIndex.x - 1, thisIndex.y).receive_ghost_data(node->node_id, input, EAST, array->ghost_size, CkDeviceBuffer(array->send_ghost_buffers[WEST], comm_stream));
            }
        }
    }

    // cudaStreamSynchronize(comm_stream);

    ghosts_expected[node->node_id] = num_nbrs * ghost_arrays[node->node_id].size();
    check_ghost_completion(node->node_id);
}

void Stencil::kernel_done(KernelCallbackMsg *msg)
{
    // if (thisIndex.x == 0 && thisIndex.y == 0)
    //     DEBUG_PRINT("PE %i> Kernel done %i\n", CkMyPe(), msg->node_id);
    mark_done(node_cache[msg->node_id]);
}

void Stencil::execute_kernel(KernelDAGNode *node)
{
    // if (thisIndex.x == 0 && thisIndex.y == 0)
    DEBUG_PRINT("(%i, %i)> Executing kernel %i\n", thisIndex.x, thisIndex.y, node->node_id);
    hapiCheck(cudaEventRecord(comm_event, comm_stream));
    hapiCheck(cudaStreamWaitEvent(compute_stream, comm_event, 0));

    Kernel *kernel = codegen_proxy.ckLocalBranch()->kernels[node->kernel_id];
    std::vector<void *> args;
    for (int i = 0; i < kernel->num_args; i++)
    {
        int input = node->inputs[i];
        Array *array = arrays[input];
        args.push_back(&(array->data));
        args.push_back(&(array->strides[0]));
        DEBUG_PRINT("Chare (%i, %i)> Arg %i: array %i, strides = (%i, %i)\n",
                    index[0], index[1], i, input, array->strides[0], array->strides[1]);
    }

    std::vector<Slice *> bounds;
    bool is_required = false;
    for (int i = 0; i < kernel->num_outputs; i++)
    {
        int output_index = kernel->outputs[i];
        int output = node->inputs[output_index];
        Array *array = arrays[output];
        Slice *bound = new Slice();
        // FIXME assumption - each array is only written to once
        // in a kernel
        *bound = kernel->get_launch_bounds(output_index, array, index);
        if (bound->index[0].start != bound->index[0].stop && bound->index[1].start != bound->index[1].stop)
            is_required = true;
        bounds.push_back(bound);
        args.push_back(&(bound->index[0].start));
        args.push_back(&(bound->index[0].stop));
        args.push_back(&(bound->index[1].start));
        args.push_back(&(bound->index[1].stop));
        // if(thisIndex.x == 0 && thisIndex.y == 0)
        DEBUG_PRINT("Chare (%i, %i)> Bounds: (%i : %i), (%i : %i)\n",
                    index[0], index[1],
                    bound->index[0].start, bound->index[0].stop,
                    bound->index[1].start, bound->index[1].stop);
        array->generation++;
        // if (thisIndex.x == 0 && thisIndex.y == 0)
        //     DEBUG_PRINT("PE %i> Kernel %i, array %i, generation = %i\n", CkMyPe(), node->node_id, output, array->generation);
    }

    if (is_required)
    {
        compute_fun_t fn = codegen_proxy.ckLocalBranch()->lookup(kernel->hash);
        int threads_per_block[2];
        int grid_dims[2];
        kernel->get_launch_params(bounds, threads_per_block, grid_dims);

        int shmem_size = 0;
        for (int idx : kernel->context->shmem_info)
            shmem_size += sizeof(float) * (threads_per_block[0] + 2 * kernel->ghost_info[idx]) * (threads_per_block[1] + 2 * kernel->ghost_info[idx]);

        DEBUG_PRINT("PE %i> Launch kernel (%i, %i); grid (%i, %i); num_args = %i, shmem = %i\n",
                    CkMyPe(), threads_per_block[0], threads_per_block[1], grid_dims[0], grid_dims[1], args.size(), shmem_size);

        launch_kernel(args, fn, compute_stream, shmem_size, threads_per_block, grid_dims);
    }

    mark_done(node);

    for (auto &bound : bounds)
        delete bound;
}

/*void wait(ck::future<bool> done)
{
    ck::wait_all(futures.begin(), futures.end());
    for (auto& f: futures)
        f.release();
    futures.clear();
    done.set(true);
}*/

void Stencil::create_array(int name, std::vector<int> shape)
{
    DEBUG_PRINT("PE %i> Create field %i with depth\n", CkMyPe(), name);
    std::vector<int> local_shape;
    int ghost_depth = ghost_info[name];
    for (int i = 0; i < 2; i++)
    {
        int local_dim = shape[i] / num_chares[i];
        local_shape.push_back(local_dim);
    }
    arrays[name] = new Array(name, local_shape, shape, ghost_depth, boundary);
    invoke_init_array(arrays[name]->data, arrays[name]->total_size, compute_stream);
}
