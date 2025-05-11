#include <cuda.h>
#include <vector>
#include <cinttypes>
#include <cmath>
#include <fstream>
#include <cstring>
#include <iostream>
#define FMT_HEADER_ONLY
#include <fmt/format.h>
#include "hapi.h"
#include "utils.hpp"

#define COMPUTE_FUNC "compute_func"

#define CHECK(expression)                                    \
  {                                                          \
    CUresult status = (expression);                          \
    if (status != CUDA_SUCCESS) {                            \
      const char* err_str;                                   \
      cuGetErrorString(status, &err_str);                    \
      std::cerr << "CUDA Error on line " << __LINE__ << ": " \
                << err_str << std::endl;                     \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

#define CHECK_RT(expression)                                    \
  {                                                          \
    cudaError_t status = (expression);                          \
    if (status != CUDA_SUCCESS) {                            \
      const char* err_str;                                   \
      cudaGetErrorString(status, &err_str);                    \
      std::cerr << "CUDA Error on line " << __LINE__ << ": " \
                << err_str << std::endl;                     \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

#define BSZ1D 128
#define BSZ2D 16
#define IDX2D(y, x, stride) ((y) * (stride) + (x))

__global__
void init_array(float* f, int total_size)
{
    int d0 = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(d0 < total_size)
        f[d0] = 0;
}

// __global__
// void fb_unpacking_kernel(double* f, double* ghost_data, int ghost_depth, 
//     int startx, int starty, int startz, int stepx, int stepy, int local_size)
// {
//     int i = blockDim.x * blockIdx.x + threadIdx.x;
//     int k = blockDim.y * blockIdx.y + threadIdx.y;

//     if (i < local_size && k < local_size)
//         for (int j = 0; j < ghost_depth; j++)
//             f[IDX(startx + i, starty + j, startz + k, stepx, stepy)] = ghost_data[IDX(i, j, k, ghost_depth, local_size)];
// }

__global__
void ns_packing_kernel(const float* array, float* ghost_data, int ghost_depth, 
    int startx, int starty, int stride, int local_size)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x >= local_size)
        return;

    for (int y = 0; y < ghost_depth; y++)
        ghost_data[IDX2D(y, x, local_size)] = array[IDX2D(y + starty, x + startx, stride)];
}

__global__
void ew_packing_kernel(const float* array, float* ghost_data, int ghost_depth, 
    int startx, int starty, int stride, int local_size)
{
    int y = blockDim.x * blockIdx.x + threadIdx.x;

    if (y >= local_size)
        return;

    for (int x = 0; x < ghost_depth; x++)
        ghost_data[IDX2D(x, y, local_size)] = array[IDX2D(y + starty, x + startx, stride)];
}

__global__
void ns_unpacking_kernel(float* array, const float* ghost_data, int ghost_depth, 
    int startx, int starty, int stride, int local_size)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x >= local_size)
        return;

    for (int y = 0; y < ghost_depth; y++)
        array[IDX2D(y + starty, x + startx, stride)] = ghost_data[IDX2D(y, x, local_size)];
}

__global__
void ew_unpacking_kernel(float* array, const float* ghost_data, int ghost_depth, 
    int startx, int starty, int stride, int local_size)
{
    int y = blockDim.x * blockIdx.x + threadIdx.x;

    if (y >= local_size)
        return;

    for (int x = 0; x < ghost_depth; x++)
        array[IDX2D(y + starty, x + startx, stride)] = ghost_data[IDX2D(x, y, local_size)];
}

void invoke_ns_packing_kernel(float* array, float* ghost_data, int ghost_depth, 
    int startx, int stopx, int starty, int stopy,
    int stride, int local_size, cudaStream_t& stream)
{
    dim3 block(BSZ1D, 1, 1);
    dim3 grid(ceil((float) (stopx - startx) / BSZ1D), 1, 1);

    ns_packing_kernel<<<grid, block, 0, stream>>>(array, ghost_data, ghost_depth, startx, starty, 
        stride, local_size);
    hapiCheck(cudaPeekAtLastError());
    //cudaStreamSynchronize(stream);
}

void invoke_ew_packing_kernel(float* array, float* ghost_data, int ghost_depth, 
    int startx, int stopx, int starty, int stopy,
    int stride, int local_size, cudaStream_t& stream)
{
    dim3 block(BSZ1D, 1, 1);
    dim3 grid(ceil((float) (stopy - starty) / BSZ1D), 1, 1);

    ew_packing_kernel<<<grid, block, 0, stream>>>(array, ghost_data, ghost_depth, startx, starty, 
        stride, local_size);
    hapiCheck(cudaPeekAtLastError());
    //cudaStreamSynchronize(stream);
}

void invoke_ns_unpacking_kernel(float* array, float* ghost_data, int ghost_depth, 
    int startx, int stopx, int starty, int stopy,
    int stride, int local_size, cudaStream_t& stream)
{
    dim3 block(BSZ1D, 1, 1);
    dim3 grid(ceil((float) (stopx - startx) / BSZ1D), 1, 1);

    //DEBUG_PRINT("Unpacking NS kernel: %d %d %d %d %d %d\n", startx, stopx, starty, stopy, stride, local_size);

    ns_unpacking_kernel<<<grid, block, 0, stream>>>(array, ghost_data, ghost_depth, startx, starty, 
        stride, local_size);
    hapiCheck(cudaPeekAtLastError());
    //cudaStreamSynchronize(stream);
}

void invoke_ew_unpacking_kernel(float* array, float* ghost_data, int ghost_depth, 
    int startx, int stopx, int starty, int stopy,
    int stride, int local_size, cudaStream_t& stream)
{
    dim3 block(BSZ1D, 1, 1);
    dim3 grid(ceil((float) (stopy - starty) / BSZ1D), 1, 1);

    //DEBUG_PRINT("Unpacking EW kernel: %d %d %d %d %d %d\n", startx, stopx, starty, stopy, stride, local_size);

    ew_unpacking_kernel<<<grid, block, 0, stream>>>(array, ghost_data, ghost_depth, startx, starty, 
        stride, local_size);
    hapiCheck(cudaPeekAtLastError());
    //cudaStreamSynchronize(stream);
}

void invoke_init_array(float* array, int total_size, cudaStream_t& stream)
{
    // TODO better to launch one kernel for all fields?
    int num_blocks = ceil((float) total_size / BSZ1D);
    init_array<<<num_blocks, BSZ1D, 0, stream>>>(array, total_size);
    hapiCheck(cudaPeekAtLastError());
}

CUfunction load_kernel(size_t &hash, int suffix)
{
    CUmodule cumodule;
    CUfunction kernel;

    std::string ptx_file = fmt::format("generated/kernel_{}_{}.ptx", hash, suffix);

    DEBUG_PRINT("Loading kernel from file: %s\n", ptx_file.c_str());

    CHECK(cuModuleLoad(&cumodule, ptx_file.c_str()));
    CHECK(cuModuleGetFunction(&kernel, cumodule, COMPUTE_FUNC));
    return kernel;
}

void launch_kernel(std::vector<void*> args, CUfunction& compute_kernel, cudaStream_t& stream,
    int* threads_per_block, int* grid)
{
    // figure out how to load compute kernel
    //printf("Launch kernel: %d %d %d %d\n", threads_per_block[0], threads_per_block[1], grid[0], grid[1]);
    CHECK(cuLaunchKernel(compute_kernel, 
        grid[0], grid[1], 1,
        threads_per_block[0], threads_per_block[1], 1,
        0, stream, args.data(), NULL));
    //cuStreamSynchronize(stream);
    //hapiCheck(cudaPeekAtLastError());
}