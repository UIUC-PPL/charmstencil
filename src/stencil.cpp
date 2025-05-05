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

void CodeGenCache::receive(int size, char* msg, CProxy_Stencil stencil_proxy)
{
    char* cmd = msg + CmiMsgHeaderSizeBytes;
    int num_kernels = extract<int>(cmd);
    for (int i = 0; i < num_kernels; i++)
    {
        Kernel* knl = build_kernel(cmd);
        kernels[knl->kernel_id] = knl;
        generate_kernel(knl, thisIndex);
        cache[knl->hash] = load_kernel(knl->hash, thisIndex);
    }

    contribute();

    int dag_size = extract<int>(cmd);
    if (thisIndex == 0)
        stencil_proxy.receive_dag(dag_size, cmd);
}

Stencil::Stencil(int num_chares_x, int num_chares_y)
{
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

void Stencil::receive_dag(int size, char* cmd)
{
    char* graph = cmd + CmiMsgHeaderSizeBytes;
    std::unordered_map<int, DAGNode*> node_cache;
    std::vector<DAGNode*> goals = build_dag(graph, node_cache);
    std::vector<CkFutureID> futures;
    for (auto& goal : goals)
    {
        CkLocalFuture fut = traverse_dag(goal);
        futures.push_back(fut.id);
    }
    // wait for all futures
    CkWaitAllIDs(futures);
}

CkLocalFuture Stencil::traverse_dag(DAGNode* node)
{
    if (node->status == NodeStatus::Visited)
        return node->future;

    if (auto* array_node = dynamic_cast<ArrayDAGNode*>(node))
    {
        // create the array
        CkSendToLocalFuture(node->future, NULL);
        create_array(array_node->name, array_node->shape);
        node->status = NodeStatus::Visited;
        return node->future;
    }

    auto* kernel_node = dynamic_cast<KernelDAGNode*>(node);
    std::vector<CkFutureID> input_futures;

    for (auto& dep : kernel_node->dependencies)
    {
        // traverse the dependencies
        CkLocalFuture inp_fut = traverse_dag(dep);
        // add the future to the inputs of the kernel
        input_futures.push_back(inp_fut.id);
    }

    // wait for all input_futures
    CkWaitAllIDs(input_futures);

    // execute this node and return a future
    thisProxy[thisIndex].execute(kernel_node->kernel_id, kernel_node->inputs, node->future);
    kernel_node->status = NodeStatus::Visited;
    return node->future;
}

void Stencil::execute(int kernel_id, std::vector<int> inputs, CkLocalFuture future)
{
    // execute the kernel
    // TODO
    // first data transfers
    send_ghost_data();
    //CkSendToLocalFuture(future, true);
    execute_kernel(kernel_id, inputs);
    CkSendToLocalFuture(future, NULL);
}

void Stencil::send_ghost_data()
{}

void Stencil::execute_kernel(int kernel_id, std::vector<int> inputs)
{
    Kernel* kernel = codegen_proxy.ckLocalBranch()->kernels[kernel_id];
    std::vector<void*> args;
    std::vector<Array*> array_args;
    for (int i = 0; i < kernel->num_inputs; i++)
    {
        int input = inputs[i];
        Array* array = arrays[input];
        args.push_back(array->data);
        Slice bounds = kernel->get_launch_bounds(input, array);
        if (i < kernel->num_outputs)
        {
            args.push_back(&(bounds.index[0].start));
            args.push_back(&(bounds.index[0].stop));
        }
        args.push_back(&(array->strides[0]));
        array_args.push_back(array);
    }

    compute_fun_t fn = codegen_proxy.ckLocalBranch()->lookup(kernel->hash);
    int threads_per_block[2];
    int grid_dims[2];
    kernel->get_launch_params(array_args, threads_per_block, grid_dims);
    launch_kernel(args.data(), fn, compute_stream, threads_per_block, grid_dims);
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
    for (int i = 0; i < 2; i++)
    {
        int local_dim = shape[i] / num_chares[i];
        local_shape.push_back(local_dim);
    }
    arrays[name] = new Array(name, local_shape);
    //invoke_init_field(field_array, total_local_size, compute_stream);
}