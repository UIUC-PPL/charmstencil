#ifndef STENCIL_HPP
#define STENCIL_HPP
#include <vector>
#include <cstring>
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


#define LEFT 0
#define RIGHT 1
#define FRONT 2
#define BACK 3
#define DOWN 4
#define UP 5


extern void invoke_fb_unpacking_kernel(double* f, double* ghost_data, int ghost_depth, int startx, 
    int starty, int startz, int stepx, int stepy, uint32_t* local_size, cudaStream_t stream);
extern void invoke_ud_unpacking_kernel(double* f, double* ghost_data, int ghost_depth, int startx, 
    int starty, int startz, int stepx, int stepy, uint32_t* local_size, cudaStream_t stream);
extern void invoke_rl_unpacking_kernel(double* f, double* ghost_data, int ghost_depth, int startx, 
    int starty, int startz, int stepx, int stepy, uint32_t* local_size, cudaStream_t stream);
extern void invoke_fb_packing_kernel(double* f, double* ghost_data, int ghost_depth, int startx, 
    int starty, int startz, int stepx, int stepy, uint32_t* local_size, cudaStream_t stream);
extern void invoke_ud_packing_kernel(double* f, double* ghost_data, int ghost_depth, int startx, 
    int starty, int startz, int stepx, int stepy, uint32_t* local_size, cudaStream_t stream);
extern void invoke_rl_packing_kernel(double* f, double* ghost_data, int ghost_depth, int startx, 
    int starty, int startz, int stepx, int stepy, uint32_t* local_size, cudaStream_t stream);
extern void invoke_init_fields(double** fields, uint8_t num_fields, uint32_t total_size, cudaStream_t stream);
extern CUfunction load_kernel(size_t &hash, int suffix);
extern void launch_kernel(void** args, CUfunction& compute_kernel, cudaStream_t& stream,
    int* threads_per_block, int* grid);


extern CProxy_CodeGenCache codegen_proxy;


class CodeGenCache : public CBase_CodeGenCache
{
private:
    double start;
    CmiNodeLock lock;
    std::unordered_map<size_t, compute_fun_t> cache;

public:
    std::unordered_map<int, Kernel*> kernels;

    CodeGenCache();

    ~CodeGenCache();

    compute_fun_t lookup(size_t hash);

    void start_time(double time);

    void end_time(double time);

    void receive(int size, char* msg, CProxy_Stencil stencil_proxy);
};


class Stencil : public CBase_Stencil
{
public:    
    // expects that the number of dimensions and length in each 
    // dimension will be specified at the time of creation
    std::unordered_map<int, Array*> arrays;

    int num_chares[2];
    int index[2];

    cudaStream_t compute_stream;
    cudaStream_t comm_stream;

    Stencil(int num_chares_x, int num_chares_y);

    Stencil(CkMigrateMessage* m);

    ~Stencil();

    void receive_dag(int size, char* cmd);

    CkLocalFuture traverse_dag(DAGNode* node);

    void execute(int kernel_id, std::vector<int> inputs, CkLocalFuture future);

    void send_ghost_data();

    void execute_kernel(int kernel_id, std::vector<int> inputs);

    void create_array(int name, std::vector<int> shape);
};

#endif // STENCIL_HPP