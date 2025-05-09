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
    float* data;
    std::vector<int> shape;
    std::vector<int> local_shape;
    std::vector<int> global_shape;
    std::vector<int> strides;

    Array(int name_, std::vector<int> shape_, std::vector<int> global_shape_, int ghost_depth_)
        : name(name_)
        , local_shape(shape_)
        , global_shape(global_shape_)
        , ghost_depth(ghost_depth_)
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
    }

    ~Array()
    {
        hapiCheck(cudaFree(data));
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