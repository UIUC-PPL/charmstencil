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

void CodeGenCache::start_time(double time)
{
    start = time;
}

void CodeGenCache::end_time(double time)
{
    CkPrintf("Total time = %f\n", time - start);
}

void CodeGenCache::receive(int size, char* cmd, CProxy_Stencil stencil_proxy_)
{
    int num_kernels = extract<int>(cmd);
    CkPrintf("Received %i kernels\n", num_kernels);
    for (int i = 0; i < num_kernels; i++)
    {
        Kernel* knl = build_kernel(cmd);
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
    contribute(sizeof(int), (void*) &done, CkReduction::sum_int, cb);
}

void CodeGenCache::send_dag(int done)
{
    stencil_proxy.receive_dag(saved_dag_size, saved_dag);
    delete[] saved_dag;
}

void CodeGenCache::operation_done(int done)
{
    CkPrintf("Operation done\n");
    int ret = 1;
    CcsSendDelayedReply(operation_reply, sizeof(int), &ret);
}

void CodeGenCache::gather(int name, int index_x, int index_y, int local_dim, int num_chares, int data_size, float* data)
{
    int total_dim = num_chares * local_dim;

    // gather data from all sources
    int start_x = index_x * local_dim;
    int start_y = index_y * local_dim;

    int stop_x = start_x + local_dim;
    int stop_y = start_y + local_dim;

    //DEBUG_PRINT("DEBUG> (%i, %i) > (%i, %i) > (%i, %i)\n", index_x, index_y, start_x, stop_x, start_y, stop_y);

    auto it = gathered_arrays.find(name);
    float* all_data;
    if (it == gathered_arrays.end())
    {
        all_data = (float*) malloc(sizeof(float) * total_dim * total_dim);

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
        // for (int j = start_y; j < stop_y; j++) {
        //     for (int i = start_x; i < stop_x; i++) {
        //         int index = j * total_dim + i;
        //         printf("Data[%d][%d] = %f\n", j, i, all_data[index]);
        //     }
        // }

        // done gathering
        CcsSendDelayedReply(fetch_reply, sizeof(float) * total_dim * total_dim, all_data);
        free(all_data);
        gathered_arrays.erase(name);
    }
}


Stencil::Stencil(int num_chares_x, int num_chares_y)
{
    char dummy;
    DAG_DONE = &dummy;

    index[0] = thisIndex.x;
    index[1] = thisIndex.y;

    num_chares[0] = num_chares_x;
    num_chares[1] = num_chares_y;

    CUdevice cuDevice;
    CUcontext cuContext;
    hapiCheck(cudaFree(0));

    hapiCheck(cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, 0));
    hapiCheck(cudaStreamCreateWithPriority(&comm_stream, cudaStreamDefault, -1));
    

    /*for (int i = 0; i < num_nbrs; i++)
    {
        hapiCheck(cudaMalloc((void**)&send_ghosts[i], sizeof(double) * ghost_size));
        hapiCheck(cudaMalloc((void**)&recv_ghosts[i], sizeof(double) * ghost_size));
    }

    thisProxy(thisIndex.x, thisIndex.y, thisIndex.z).start();*/
}

Stencil::Stencil(CkMigrateMessage* m) {}

Stencil::~Stencil()
{
    //delete_cache();
    //for (int i = 0; i < num_nbrs; i++)
    //{
    //    hapiCheck(cudaFree(send_ghosts[i]));
    //    hapiCheck(cudaFree(recv_ghosts[i]));
    //}
    //free(send_ghosts);
    //free(recv_ghosts);
    for (auto& entry : arrays)
    {
        Array* array = entry.second;
        delete array;
    }
}

void Stencil::gather(int name)
{
    Array* array = arrays[name];
    float* data = array->to_host();
    float* local_data = array->get_local(data);
    // for (int i = 0; i < array->total_size; i++) {
    //     printf("%f ", data[i]);
    // }
    // printf("\n");
    codegen_proxy[0].gather(name, index[0], index[1], array->local_shape[0], num_chares[0], array->total_local_size, local_data);
    delete[] data;
    delete[] local_data;
}

void Stencil::receive_dag(int size, char* graph)
{
    std::unordered_map<int, DAGNode*> node_cache;
    std::vector<DAGNode*> goals = build_dag(graph, node_cache, ghost_info);
    std::vector<CkFutureID> futures;
    for (auto& goal : goals)
    {
        CkLocalFuture fut = traverse_dag(goal);
        futures.push_back(fut.id);
        DEBUG_PRINT("DEBUG> %i > Waiting for future %i\n", CkMyPe(), fut.id);
    }
    // wait for all futures
    CkWaitAllIDs(futures);

    DEBUG_PRINT("DEBUG> %i DONE WAITING!!!\n", CkMyPe());

    //int done = 1;
    //CkCallback cb(CkReductionTarget(CodeGenCache, operation_done), codegen_proxy[0]);
    //contribute(sizeof(int), (void*) &done, CkReduction::sum_int, cb);
}

CkLocalFuture Stencil::traverse_dag(DAGNode* node)
{
    if (node->status == NodeStatus::Visited)
        return node->future;

    if (auto* array_node = dynamic_cast<ArrayDAGNode*>(node))
    {
        // create the array
        CkSendToLocalFuture(node->future, DAG_DONE);
        DEBUG_PRINT("%i> Send to local future %i\n", CkMyPe(), node->future.id);
        create_array(array_node->name, array_node->shape);
        node->status = NodeStatus::Visited;
        return node->future;
    }

    auto* kernel_node = dynamic_cast<KernelDAGNode*>(node);
    std::vector<CkFutureID> input_futures;

    DEBUG_PRINT("Traversing kernel node %i\n", kernel_node->kernel_id);

    for (auto& dep : kernel_node->dependencies)
    {
        // traverse the dependencies
        CkLocalFuture inp_fut = traverse_dag(dep);
        // add the future to the inputs of the kernel
        input_futures.push_back(inp_fut.id);
    }

    // wait for all input_futures
    //CkWaitAllIDs(input_futures);

    // execute this node and return a future
    thisProxy[thisIndex].execute(kernel_node->kernel_id, kernel_node->inputs, node->future);
    kernel_node->status = NodeStatus::Visited;
    return node->future;
}

void Stencil::execute(int kernel_id, std::vector<int> inputs, CkLocalFuture future)
{
    DEBUG_PRINT("Execute kernel %i\n", kernel_id);
    // execute the kernel
    // TODO
    // first data transfers
    send_ghost_data();
    //CkSendToLocalFuture(future, true);
    execute_kernel(kernel_id, inputs, future);
}

void Stencil::send_ghost_data()
{}

void Stencil::kernel_done(KernelCallbackMsg* msg)
{
    CkSendToLocalFuture(msg->future, DAG_DONE);
}

void Stencil::execute_kernel(int kernel_id, std::vector<int> inputs, CkLocalFuture future)
{
    Kernel* kernel = codegen_proxy.ckLocalBranch()->kernels[kernel_id];
    std::vector<void*> args;
    std::vector<Array*> array_args;
    for (int i = 0; i < kernel->num_args; i++)
    {
        int input = inputs[i];
        Array* array = arrays[input];
        args.push_back(&(array->data));
        args.push_back(&(array->strides[0]));
        array_args.push_back(array);
    }

    std::vector<Slice*> bounds;
    bool is_required = false;
    for (int i = 0; i < kernel->num_outputs; i++)
    {
        int output = kernel->outputs[i];
        Array* array = arrays[output];
        Slice* bound = new Slice();
        // FIXME assumption - each array is only written to once
        // in a kernel
        *bound = kernel->get_launch_bounds(output, array, index);
        if (bound->index[0].start != bound->index[0].stop && bound->index[1].start != bound->index[1].stop)
            is_required = true;
        bounds.push_back(bound);
        args.push_back(&(bound->index[0].start));
        args.push_back(&(bound->index[0].stop));
        args.push_back(&(bound->index[1].start));
        args.push_back(&(bound->index[1].stop));
        DEBUG_PRINT("Chare (%i, %i)> Bounds: (%i : %i), (%i : %i)\n",
            index[0], index[1],
            bound->index[0].start, bound->index[0].stop,
            bound->index[1].start, bound->index[1].stop);
    }

    if (is_required)
    {
        compute_fun_t fn = codegen_proxy.ckLocalBranch()->lookup(kernel->hash);
        int threads_per_block[2];
        int grid_dims[2];
        kernel->get_launch_params(bounds, threads_per_block, grid_dims);
        launch_kernel(args, fn, compute_stream, threads_per_block, grid_dims);
        //KernelCallbackMsg* msg = new KernelCallbackMsg(future);
        //CkCallback* cb = new CkCallback(CkIndex_Stencil::kernel_done(NULL), thisProxy[thisIndex]);
        //hapiAddCallback(compute_stream, cb, msg);
    }
    else
    {
        CkSendToLocalFuture(future, DAG_DONE);
        DEBUG_PRINT("Chare (%i, %i)> No need to execute kernel %i\n", index[0], index[1], kernel_id);
    }

    for (auto& bound : bounds)
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
#ifndef NDEBUG
    CkPrintf("Create field %i with depth\n", name);
#endif
    std::vector<int> local_shape;
    int ghost_depth = ghost_info[name];
    for (int i = 0; i < 2; i++)
    {
        int local_dim = shape[i] / num_chares[i];
        local_shape.push_back(local_dim);
    }
    arrays[name] = new Array(name, local_shape, shape, ghost_depth);
    invoke_init_array(arrays[name]->data, arrays[name]->total_size, compute_stream);
}