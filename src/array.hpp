#ifndef ARRAY_HPP
#define ARRAY_HPP
#include <vector>
#include <cuda_runtime.h>
#include "hapi.h"

class Array
{
public:
    int name;
    float* data;
    std::vector<int> shape;
    std::vector<int> strides;

    Array(int name_, std::vector<int> shape_)
        : name(name_)
        , shape(shape_)
    {
        strides.resize(shape.size());
        strides[shape.size() - 1] = 1;
        for (int i = shape.size() - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];
        hapiCheck(cudaMalloc((void**) &data, sizeof(float) * shape[0] * shape[1]));
    }

    ~Array()
    {
        hapiCheck(cudaFree(data));
    }
};
#endif // ARRAY_HPP