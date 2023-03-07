#include <vector>
#include <cstring>

class Field
{
public:
    std::vector<double> data;

    Field() {}

    Field(uint32_t data_size)
    {
        data.reserve(data_size);
        std::fill(data.begin(), data.end(), 0);
    }
};

