#include "hapi.h"
#include <vector>
#include <cstring>

class Field
{
public:
    double* data;

    Field() {}

    Field(uint32_t data_size)
    {
        hapiCheck(cudaMalloc((void**) &data, sizeof(double) * data_size));

    }

    ~Field()
    {
        hapiCheck(cudaFree(data));
    }
};

