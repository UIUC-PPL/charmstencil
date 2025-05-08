#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <string>
#include <cuda.h>

#ifndef NDEBUG
#define DEBUG_PRINT(...) CkPrintf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#endif

template<class T>
inline T extract(char* &msg, bool increment=true)
{
    T arg = *(reinterpret_cast<T*>(msg));
    if (increment)
        msg += sizeof(T);
    return arg;
}

enum class OperandType : uint8_t
{
    array = 0,
    tuple_slice = 1,
    slice = 2,
    tuple_int = 3,
    int_t = 4,
    float_t = 5
};

enum class Operation : uint8_t
{
    noop = 0,
    create = 1,
    add = 2,
    sub = 3,
    mul = 4,
    norm = 5,
    getitem = 6,
    setitem = 7,
    exchange_ghosts = 8
};

std::string get_op_string(Operation& oper);

Operation lookup_operation(uint8_t opcode);

OperandType lookup_type(uint8_t typecode);


struct Slice1D
{
    int start;
    int stop;
    int step;

    Slice1D()
    {
         start = 0;
         stop = 0;
         step = 1;
    }

    Slice1D(const Slice1D& other)
    {
        start = other.start;
        stop = other.stop;
        step = other.step;
    }

    Slice1D calculate_relative(Slice1D& base)
    {
        Slice1D offset;
        offset.start = start - base.start;
        offset.stop = stop;
        offset.step = step / base.step;
        return offset;
    }
};


struct Slice
{
    Slice1D index[2];

    Slice() = default;

    Slice(const Slice& other)
    {
        index[0] = other.index[0];
        index[1] = other.index[1];
    }

    Slice calculate_relative(Slice& base)
    {
        Slice offset;
        offset.index[0] = index[0].calculate_relative(base.index[0]);
        offset.index[1] = index[1].calculate_relative(base.index[1]);
        return offset;
    }
};

Slice get_slice(char* &cmd);

#endif // UTILS_HPP