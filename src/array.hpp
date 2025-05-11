#ifndef ARRAY_HPP
#define ARRAY_HPP
#include <vector>
#include <cuda_runtime.h>
#include "hapi.h"

class Array
{
public:
    int name;
    int total_size;
    int total_local_size;
    int ghost_depth;
    int generation;
    int ghost_generation;
    float* data;
    std::vector<int> shape;
    std::vector<int> local_shape;
    std::vector<int> global_shape;
    std::vector<int> strides;

    bool exchange_in_progress;
    int ghost_size;
    float* send_ghost_buffers[4];
    float* recv_ghost_buffers[4];

    Array(int name_, std::vector<int> shape_, std::vector<int> global_shape_, int ghost_depth_, bool* boundary)
        : name(name_)
        , local_shape(shape_)
        , global_shape(global_shape_)
        , ghost_depth(ghost_depth_)
        , generation(0)
        , ghost_generation(-1)
        , exchange_in_progress(false)
    {
        for(int i = 0; i < shape_.size(); i++)
            shape.push_back(shape_[i] + 2 * ghost_depth_);
        strides.resize(shape.size());
        strides[shape.size() - 1] = 1;
        for (int i = shape.size() - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];
        total_size = shape[0] * shape[1];
        total_local_size = local_shape[0] * local_shape[1];
        hapiCheck(cudaMalloc((void**) &data, sizeof(float) * total_size));
        ghost_size = ghost_depth * local_shape[0]; // FIXME assuming square array
        allocate_ghost_buffers(boundary);
    }

    ~Array()
    {
        hapiCheck(cudaFree(data));
    }

    void allocate_ghost_buffers(bool* boundary)
    {
        for (int i = 0; i < 4; i++)
        {
            if (!boundary[i])
            {
                DEBUG_PRINT("PE %i> Allocating ghost buffers for array %i in dir %i\n", CkMyPe(), name, i);
                hapiCheck(cudaMalloc((void**) &(send_ghost_buffers[i]), sizeof(float) * ghost_size));
                hapiCheck(cudaMalloc((void**) &(recv_ghost_buffers[i]), sizeof(float) * ghost_size));
                DEBUG_PRINT("PE %i> Send ghost buffer %i: %p\n", CkMyPe(), i, send_ghost_buffers[i]);
                DEBUG_PRINT("PE %i> Recv ghost buffer %i: %p\n", CkMyPe(), i, recv_ghost_buffers[i]);
            }
        }
    }

    float* to_host()
    {
        float* host_data = new float[total_size];
        hapiCheck(cudaMemcpy(host_data, data, sizeof(float) * total_size, cudaMemcpyDeviceToHost));
        return host_data;
    }

    float* get_local(float* data)
    {
        float* local_data = new float[total_local_size];
        for (int y = 0; y < local_shape[0]; y++)
            for (int x = 0; x < local_shape[1]; x++)
            {
                int idx = (y + ghost_depth) * strides[0] + (x + ghost_depth);
                local_data[y * local_shape[1] + x] = data[idx];
            }
        
        return local_data;
    }
};
#endif // ARRAY_HPP