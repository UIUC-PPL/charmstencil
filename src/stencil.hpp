#ifndef STENCIL_HPP
#define STENCIL_HPP
#include <vector>
#include <cstring>
#include <unordered_set>
#include "codegen.hpp"
#include "dag.hpp"
#include "array.hpp"
#include "stencil.decl.h"


#define CHECK(expression)                                    \
  {                                                          \
    CUresult status = (expression);                          \
    if (status != CUDA_SUCCESS) {                            \
      const char* err_str;                                   \
      cuGetErrorString(status, &err_str);                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << err_str << std::endl;                     \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }


#define WEST 0
#define EAST 1
#define SOUTH 2
#define NORTH 3 


extern void invoke_ns_packing_kernel(float* array, float* ghost_data, int ghost_depth, 
    int startx, int stopx, int starty, int stopy,
    int stride, int local_size, cudaStream_t& stream);

extern void invoke_ew_packing_kernel(float* array, float* ghost_data, int ghost_depth, 
    int startx, int stopx, int starty, int stopy,
    int stride, int local_size, cudaStream_t& stream);

extern void invoke_ns_unpacking_kernel(float* array, float* ghost_data, int ghost_depth, 
    int startx, int stopx, int starty, int stopy,
    int stride, int local_size, cudaStream_t& stream);

extern void invoke_ew_unpacking_kernel(float* array, float* ghost_data, int ghost_depth, 
    int startx, int stopx, int starty, int stopy,
    int stride, int local_size, cudaStream_t& stream);

extern void invoke_init_array(float* array, int total_size, cudaStream_t& stream);

extern CUfunction load_kernel(size_t &hash, int suffix);

extern void launch_kernel(std::vector<void*> args, CUfunction& compute_kernel, cudaStream_t& stream,
    int shmem_size, int* threads_per_block, int* grid);


extern CProxy_CodeGenCache codegen_proxy;

extern CcsDelayedReply operation_reply;
extern CcsDelayedReply fetch_reply;


class KernelCallbackMsg : public CMessage_KernelCallbackMsg
{
public:
    int node_id;

    KernelCallbackMsg(int node_id_)
        : CMessage_KernelCallbackMsg()
        , node_id(node_id_)
    {
    }
};


class CodeGenCache : public CBase_CodeGenCache
{
private:
    double start_time;
    CmiNodeLock lock;
    std::unordered_map<size_t, compute_fun_t> cache;

    std::unordered_map<int, std::pair<int, float*>> gathered_arrays;
    //std::unordered_map<int, int> gather_recv_count;

    char* saved_dag;
    int saved_dag_size;

    CProxy_Stencil stencil_proxy;

public:
    std::unordered_map<int, Kernel*> kernels;

    CodeGenCache();

    ~CodeGenCache();

    compute_fun_t lookup(size_t hash);

    void receive(int size, char* msg, CProxy_Stencil stencil_proxy);

    void send_dag(int done);

    void operation_done(double start);

    void gather(int name, int index_x, int index_y, int local_dim, int num_chares, int data_size, float* data);
};


class Stencil : public CBase_Stencil
{
private:
    double start_time;
    int num_nbrs;
    bool boundary[4];
    char* DAG_DONE;

    std::unordered_map<int, int> ghost_info;
    std::unordered_map<int, DAGNode*> node_cache;
    std::unordered_set<int> goals_waiting;

    std::unordered_map<int, int> ghost_counts;
    std::unordered_map<int, int> ghosts_expected;
    std::unordered_map<int, std::vector<int>> ghost_arrays;
public:    
    // expects that the number of dimensions and length in each 
    // dimension will be specified at the time of creation
    std::unordered_map<int, Array*> arrays;

    int num_chares[2];
    int index[2];

    cudaStream_t compute_stream;
    cudaStream_t comm_stream;

    cudaEvent_t compute_event;
    cudaEvent_t comm_event;

    Stencil(int num_chares_x, int num_chares_y);

    Stencil(CkMigrateMessage* m);

    ~Stencil();

    void mark_done(DAGNode* node);

    void kernel_done(KernelCallbackMsg* msg);

    void gather(int name);

    void receive_ghost_data(int node_id, int name, int dir, int& size, float* &buf, CkDeviceBufferPost* device_post);

    void receive_ghost_data(int node_id, int name, int dir, int size, float* buf);

    void check_ghost_completion(int node_id);

    void ghost_done(KernelCallbackMsg* msg);

    void handle_ghost_completion(int node_id);

    void receive_dag(int size, char* cmd);

    bool traverse_dag(DAGNode* node);

    void send_ghost_data(KernelDAGNode* node);

    void execute_kernel(KernelDAGNode* node);

    void create_array(int name, std::vector<int> shape);
};

#endif // STENCIL_HPP