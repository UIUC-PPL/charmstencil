#include "hapi.h"
#include <vector>
#include <cstring>

class Field
{
public:
    double* data;

    Field() {}

    Field(uint32_t data_size, bool is_gpu)
    {
        if (is_gpu)
        {
            hapiCheck(cudaMalloc((void**) &data, sizeof(double) * data_size));
        }
        else
        {
            data = (double*) malloc(sizeof(double) * data_size);
        }
    }

    ~Field()
    {
        if (is_gpu)
            hapiCheck(cudaFree(data));
        else
            free(data);
    }
};

