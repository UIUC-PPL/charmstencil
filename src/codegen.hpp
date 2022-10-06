#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <cinttypes>
#include <unordered_map>
#define FMT_HEADER_ONLY
#include <fmt/format.h>

//using scalar_t = std::variant<int, double>


template<class T>
inline T extract(char* &msg, bool increment=true)
{
    T arg = *(reinterpret_cast<T*>(msg));
    if (increment)
        msg += sizeof(T);
    return arg;
}


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


enum class OperandType : uint8_t
{
    field = 0,
    slice = 1,
    tuple = 2,
    int_t = 3,
    double_t = 4
};


Operation lookup_operation(uint8_t opcode)
{
    return static_cast<Operation>(opcode);
}


OperandType lookup_type(uint8_t typecode)
{
    return static_cast<OperandType>(typecode);
}


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
};


struct Slice
{
    Slice1D index[3];
};


Slice get_slice(char* &cmd, int ndim, 
        std::vector<uint32_t> num_chares, std::vector<uint32_t> local_size)
{
    Slice key;
    int start, stop, step;

    OperandType operand_type = lookup_type(extract<uint8_t>(cmd));

    switch (operand_type)
    {
        case OperandType::slice:
        {
            printf("Checking slice\n");
            for(int i = 0; i < ndim; i++)
            {
                start = extract<int>(cmd);
                stop = extract<int>(cmd);
                step = extract<int>(cmd);

                key.index[i].start = start;
                key.index[i].stop = stop < 0 ? stop + local_size[i] : stop;
            }
        }
    }
    
    return key;
}

void generate_code(FILE* genfile, char* &cmd, int ndim, std::vector<uint32_t> ghost_depth,
        std::vector<uint32_t> local_size, std::vector<uint32_t> num_chares);


void generate_headers(FILE* genfile);


void generate(char* cmd, int ndim, std::vector<uint32_t> ghost_depth,
        std::vector<uint32_t> local_size, std::vector<uint32_t> num_chares)
{
    // first lookup local cache
    // if not found then lookup shared library of the name = hash of ast
    // if not found then lookup generated code of name = hash of ast
    // else generate code and compile and load function dynamically
    FILE* genfile = fopen("testgen.cpp", "w");
    fprintf(genfile, "#include \"stencil.hpp\"\n\n");
    fprintf(genfile, "void compute_func(Stencil* st) {\n");
    generate_code(genfile, cmd, ndim, ghost_depth, local_size, num_chares);
    fprintf(genfile, "}\n");
}


void generate_headers(FILE* genfile)
{
}


std::string generate_loop_rhs(FILE* genfile, char* &cmd, int ndim, uint32_t depth,
        std::vector<uint32_t> local_size, std::vector<uint32_t> num_chares);


void generate_code(FILE* genfile, char* &cmd, int ndim, 
        std::vector<uint32_t> ghost_depth,
        std::vector<uint32_t> local_size, 
        std::vector<uint32_t> num_chares)
{
    char* start = cmd;
    OperandType operand_type = lookup_type(extract<uint8_t>(cmd));
    uint8_t opcode = extract<uint8_t>(cmd);
    Operation oper = lookup_operation(opcode);
    switch (oper)
    {
        case Operation::setitem:
        {
            //uint8_t num_operands = extract<uint8_t>(cmd);
            uint8_t ftype = extract<uint8_t>(cmd);
            uint8_t fname_optype = extract<uint8_t>(cmd);
            //uint8_t fname_oper = extract<uint8_t>(cmd);
            uint8_t fname = extract<uint8_t>(cmd);
            printf("Before = %p\n", cmd);
            Slice key = get_slice(cmd, ndim, num_chares, local_size);
            printf("After = %p\n", cmd);

            uint32_t depth = ghost_depth[fname];
            uint32_t step_x, step_y, step_z;

            if(ndim > 0)
                step_x = local_size[0] + 2 * depth;
            if(ndim > 1)
                step_y = local_size[1] + 2 * depth;
            if(ndim > 2)
                step_z = local_size[2] + 2 * depth;

            fprintf(genfile, "Field& f%" PRIu8 " = st->fields[%" PRIu8 "];\n", fname, fname);
            //fprintf(genfile, "int start_idx[%i];\n", ndim);
            fprintf(genfile, "int stop_idx[%i];\n", ndim);
            //for(int i = 0; i < ndim; i++)
            //    fprintf(genfile, "int start_idx[%i] = st->index[%i] == 0 ? %i : %i;\n", 
            //            i, i, key.index[0].start + depth, depth);
            for(int i = 0; i < ndim; i++)
                fprintf(genfile, "stop_idx[%i] = st->index[%i] == "
                        "st->num_chares[%i] - 1 || st->index[%i] == 0 ? %i : %i;\n", 
                    i, i, i, i, key.index[0].stop - depth, local_size[i] - depth);
           
            // write the loops
            for (int i = 0; i < ndim; i++)
                fprintf(genfile, "for (int d%i = 0; d%i * %i"
                        " < stop_idx[%i]; d%i++) {\n", 
                        i, i, key.index[i].step, i, i);

            std::string index_str;

            if (ndim == 1)
                index_str = fmt::format("d0 * {} + {}", key.index[0].step, depth);
            if (ndim == 2)
                index_str = fmt::format("d0 * {} + {} + {} * (d1 * {} + {})", 
                        key.index[0].step, depth, step_x, key.index[1].step, depth);
            if (ndim == 3)
                index_str = fmt::format(
                        "d0 * {} + {} + {} * (d1 * {} + {}) + {} * (d2 * {} + {})", 
                        key.index[0].step, depth, step_x, key.index[1].step, depth, step_x * step_y,
                        key.index[2].step, depth);

            printf("offset = %p, %p, %i\n", cmd, start, cmd - start);

            fprintf(genfile, "f%i.data[%s] = %s;\n", fname, index_str.c_str(), 
                    generate_loop_rhs(genfile, cmd, ndim, depth, local_size, num_chares).c_str());

            for (int i = 0; i < ndim; i++)
                fprintf(genfile, "}\n");
        }
        case Operation::norm:
        {}
        case Operation::exchange_ghosts:
        {}
        default:
        {}
    }
}

std::string generate_loop_rhs(FILE* genfile, char* &cmd, int ndim, uint32_t depth,
        std::vector<uint32_t> local_size, std::vector<uint32_t> num_chares)
{
    printf("RHS cmd = %p\n", cmd);
    OperandType operand_type = lookup_type(extract<uint8_t>(cmd));
    uint8_t opcode = extract<uint8_t>(cmd);
    Operation oper = lookup_operation(opcode);
    printf("optype = %" PRIu8 "\n", operand_type);
    printf("opcode rhs = %" PRIu8 "\n", opcode);
    switch (oper)
    {
        case Operation::noop:
        {
            std::string res = "";
            switch(operand_type)
            {
                case OperandType::double_t:
                {
                    double scalar = extract<double>(cmd);
                    res = std::to_string(scalar);
                    break;
                }
                case OperandType::field:
                {
                    uint8_t fname = extract<uint8_t>(cmd);
                    res = fmt::format("st->fields[{}]", fname);
                    break;
                }
                default:
                {
                    std::cerr << "ERROR: Invalid operand type " << 
                        static_cast<uint8_t>(operand_type) << std::endl;
                }
            }
            return res;
        }
        case Operation::add: 
        case Operation::sub: 
        case Operation::mul: 
        {
            std::string op1, op2, opstr;

            op1 = generate_loop_rhs(genfile, cmd, ndim, depth, local_size, 
                            num_chares);
            op2 = generate_loop_rhs(genfile, cmd, ndim, depth, local_size, 
                            num_chares);

            if (oper == Operation::add)
                opstr = "+";
            else if (oper == Operation::sub)
                opstr = "-";
            else if (oper == Operation::mul)
                opstr = "*";

            return "(" + op1 + opstr + op2 + ")";
        }
        case Operation::getitem: 
        {
            std::string f_str = generate_loop_rhs(
                    genfile, cmd, ndim, depth, local_size, num_chares);

            Slice key = get_slice(cmd, ndim, num_chares, local_size);
            
            uint32_t step_x, step_y, step_z;
            if(ndim > 0)
                step_x = local_size[0] + 2 * depth;
            if(ndim > 1)
                step_y = local_size[1] + 2 * depth;
            if(ndim > 2)
                step_z = local_size[2] + 2 * depth;

            std::string index_str;

            if (ndim == 1)
                index_str = fmt::format("{} + d0 * {}", 
                        key.index[0].start, key.index[0].step);
            if (ndim == 2)
                index_str = fmt::format("{} * ({} + d0 * {}) + ({} + d1 * {})", 
                        step_x, key.index[1].start, key.index[1].step,
                        key.index[0].start, key.index[0].step);
            if (ndim == 3)
                index_str = fmt::format(
                        "{} * ({} + d0 * {}) + {} * ({} + d1 * {}) + ({} + d2 * {})", 
                        step_x * step_y, key.index[2].start, key.index[2].step,
                        step_x, key.index[1].start, key.index[1].step,
                        key.index[0].start, key.index[0].step);

            std::string result_str = fmt::format("({}).data[{}]", f_str, index_str);

            return result_str;
        }
        default:
        {
            return "";
        }
    }
}
