#include "utils.hpp"

std::string get_op_string(Operation& oper)
{
    switch (oper)
    {
        case Operation::add:
            return "+";
        case Operation::sub:
            return "-";
        case Operation::mul:
            return "*";
        default:
            return "";
    }
}

Operation lookup_operation(uint8_t opcode)
{
    return static_cast<Operation>(opcode);
}


OperandType lookup_type(uint8_t typecode)
{
    return static_cast<OperandType>(typecode);
}

Slice get_slice(char* &cmd)
{
    Slice key;
    int start, stop, step;

    int ndims = 2;

    for(int i = 0; i < ndims; i++)
    {
        start = extract<int>(cmd);
        stop = extract<int>(cmd);
        step = extract<int>(cmd);

        key.index[i].start = start;
        key.index[i].stop = stop;
        key.index[i].step = step;
    }

    return key;
}