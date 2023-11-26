#include <cuda.h>
#include <vector>
#include <cinttypes>
#include <cmath>
#include <fstream>
#include <cstring>
#include <fmt/format.h>
#include "hapi.h"

#define COMPUTE_FUNC "_Z12compute_funcv"

#define BSZ1D 128
#define BSZ2D 16
#define IDX(i, j, k, stepx, stepy) ((i) + (stepx) * (j) + (stepx) * (stepy) * (k))

__global__
void init_fields(double* f, int total_size)
{
    int d0 = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(d0 < total_size)
        f[d0] = 0;
}

// this is to send to the right/left
__global__
void rl_packing_kernel(double* f, double* ghost_data, int ghost_depth, 
    int startx, int starty, int startz, int stepx, int stepy, int local_size)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.y * blockIdx.y + threadIdx.y;

    if (j < local_size && k < local_size)
        for (int i = 0; i < ghost_depth; i++)
            ghost_data[IDX(i, j, k, ghost_depth, local_size)] = f[IDX(startx + i, starty + j, startz + k, stepx, stepy)];
}

// FIXME memory coalescing here can improve performance
// change the order of accesses to ghost_data
__global__
void ud_packing_kernel(double* f, double* ghost_data, int ghost_depth, 
    int startx, int starty, int startz, int stepx, int stepy, int local_size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < local_size && j < local_size)
        for (int k = 0; k < ghost_depth; k++)
            ghost_data[IDX(i, j, k, ghost_depth, local_size)] = f[IDX(startx + i, starty + j, startz + k, stepx, stepy)];
}

__global__
void fb_packing_kernel(double* f, double* ghost_data, int ghost_depth, 
    int startx, int starty, int startz, int stepx, int stepy, int local_size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < local_size && k < local_size)
        for (int j = 0; j < ghost_depth; j++)
            ghost_data[IDX(i, j, k, ghost_depth, local_size)] = f[IDX(startx + i, starty + j, startz + k, stepx, stepy)];
}

// this is to send to the right/left
__global__
void rl_unpacking_kernel(double* f, double* ghost_data, int ghost_depth, 
    int startx, int starty, int startz, int stepx, int stepy, int local_size)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.y * blockIdx.y + threadIdx.y;

    if (j < local_size && k < local_size)
        for (int i = 0; i < ghost_depth; i++)
            f[IDX(startx + i, starty + j, startz + k, stepx, stepy)] = ghost_data[IDX(i, j, k, ghost_depth, local_size)];
}

// FIXME memory coalescing here can improve performance
// change the order of accesses to ghost_data
__global__
void ud_unpacking_kernel(double* f, double* ghost_data, int ghost_depth, 
    int startx, int starty, int startz, int stepx, int stepy, int local_size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < local_size && j < local_size)
        for (int k = 0; k < ghost_depth; k++)
            f[IDX(startx + i, starty + j, startz + k, stepx, stepy)] = ghost_data[IDX(i, j, k, ghost_depth, local_size)];
}

__global__
void fb_unpacking_kernel(double* f, double* ghost_data, int ghost_depth, 
    int startx, int starty, int startz, int stepx, int stepy, int local_size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < local_size && k < local_size)
        for (int j = 0; j < ghost_depth; j++)
            f[IDX(startx + i, starty + j, startz + k, stepx, stepy)] = ghost_data[IDX(i, j, k, ghost_depth, local_size)];
}


void invoke_rl_packing_kernel(double* f, double* ghost_data, int ghost_depth, int startx, 
    int starty, int startz, int stepx, int stepy, uint32_t* local_size, cudaStream_t stream)
{
    dim3 block(BSZ2D, BSZ2D);
    dim3 grid(ceil((float) local_size[1] / BSZ2D), ceil((float) local_size[2] / BSZ2D));
    rl_packing_kernel<<<grid, block, 0, stream>>>(f, ghost_data, ghost_depth, startx, starty, startz, 
        stepx, stepy, local_size[0]);
    hapiCheck(cudaPeekAtLastError());
}

void invoke_ud_packing_kernel(double* f, double* ghost_data, int ghost_depth, int startx, 
    int starty, int startz, int stepx, int stepy, uint32_t* local_size, cudaStream_t stream)
{
    dim3 block(BSZ2D, BSZ2D);
    dim3 grid(ceil((float) local_size[0] / BSZ2D), ceil((float) local_size[1] / BSZ2D));
    ud_packing_kernel<<<grid, block, 0, stream>>>(f, ghost_data, ghost_depth, startx, starty, startz, 
        stepx, stepy, local_size[0]);
    hapiCheck(cudaPeekAtLastError());
}

void invoke_fb_packing_kernel(double* f, double* ghost_data, int ghost_depth, int startx, 
    int starty, int startz, int stepx, int stepy, uint32_t* local_size, cudaStream_t stream)
{
    dim3 block(BSZ2D, BSZ2D);
    dim3 grid(ceil((float) local_size[0] / BSZ2D), ceil((float) local_size[2] / BSZ2D));
    fb_packing_kernel<<<grid, block, 0, stream>>>(f, ghost_data, ghost_depth, startx, starty, startz, 
        stepx, stepy, local_size[0]);
    hapiCheck(cudaPeekAtLastError());
}

void invoke_rl_unpacking_kernel(double* f, double* ghost_data, int ghost_depth, int startx, 
    int starty, int startz, int stepx, int stepy, uint32_t* local_size, cudaStream_t stream)
{
    dim3 block(BSZ2D, BSZ2D);
    dim3 grid(ceil((float) local_size[1] / BSZ2D), ceil((float) local_size[2] / BSZ2D));
    rl_unpacking_kernel<<<grid, block, 0, stream>>>(f, ghost_data, ghost_depth, startx, starty, startz, 
        stepx, stepy, local_size[0]);
    hapiCheck(cudaPeekAtLastError());
}

void invoke_ud_unpacking_kernel(double* f, double* ghost_data, int ghost_depth, int startx, 
    int starty, int startz, int stepx, int stepy, uint32_t* local_size, cudaStream_t stream)
{
    dim3 block(BSZ2D, BSZ2D);
    dim3 grid(ceil((float) local_size[0] / BSZ2D), ceil((float) local_size[1] / BSZ2D));
    ud_unpacking_kernel<<<grid, block, 0, stream>>>(f, ghost_data, ghost_depth, startx, starty, startz, 
        stepx, stepy, local_size[0]);
    hapiCheck(cudaPeekAtLastError());
}

void invoke_fb_unpacking_kernel(double* f, double* ghost_data, int ghost_depth, int startx, 
    int starty, int startz, int stepx, int stepy, uint32_t* local_size, cudaStream_t stream)
{
    dim3 block(BSZ2D, BSZ2D);
    dim3 grid(ceil((float) local_size[0] / BSZ2D), ceil((float) local_size[2] / BSZ2D));
    fb_unpacking_kernel<<<grid, block, 0, stream>>>(f, ghost_data, ghost_depth, startx, starty, startz, 
        stepx, stepy, local_size[0]);
    hapiCheck(cudaPeekAtLastError());
}

void invoke_init_fields(double** fields, uint8_t num_fields, uint32_t total_size, cudaStream_t stream)
{
    // TODO better to launch one kernel for all fields?
    printf("total size = %u\n", total_size);
    int num_blocks = ceil((float) total_size / BSZ1D);
    for(int i = 0; i < num_fields; i++)
    {
        init_fields<<<num_blocks, BSZ1D, 0, stream>>>(fields[i], total_size);
        hapiCheck(cudaPeekAtLastError());
    }
}

void* get_module(std::string &fatbin_file)
{
    std::ifstream file(fatbin_file, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    void* buffer = malloc(size);
    file.read((char*) buffer, size);
    return buffer;
}

CUfunction load_kernel(size_t &hash)
{
    CUmodule cumodule;
    CUfunction kernel;

    std::string fatbin_file = fmt::format("./stencil_{}.fatbin", hash);

    cuModuleLoadFatBinary(&cumodule, get_module(fatbin_file));
    cuModuleGetFunction(&kernel, cumodule, COMPUTE_FUNC);
    return kernel;
}

void launch_kernel(void** args, uint32_t* local_size, int* block_sizes, 
    CUfunction& compute_kernel, cudaStream_t& stream)
{
    // figure out how to load compute kernel
    cuLaunchKernel(compute_kernel, 
        ceil(((float) local_size[0]) / block_sizes[0]),
        ceil(((float) local_size[1]) / block_sizes[1]),
        ceil(((float) local_size[2]) / block_sizes[2]),
        block_sizes[0], block_sizes[1], block_sizes[2],
        0, stream, args, NULL);
    hapiCheck(cudaPeekAtLastError());
}