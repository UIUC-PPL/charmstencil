#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <cinttypes>
#include <unordered_map>
#include <string_view>
#include <dlfcn.h>
#include "charm++.h"
#define FMT_HEADER_ONLY
#include <fmt/format.h>


typedef void* (*compute_fun_t)(std::vector<Field>&, std::vector<uint32_t>&, std::vector<int>&);


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


Slice get_slice(char* &cmd, int ndims, 
        std::vector<uint32_t> num_chares, std::vector<uint32_t> local_size)
{
    Slice key;
    int start, stop, step;

    OperandType operand_type = lookup_type(extract<uint8_t>(cmd));

    switch (operand_type)
    {
        case OperandType::slice:
        {
            for(int i = 0; i < ndims; i++)
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

void generate_code(FILE* genfile, char* &cmd, int ndims, std::vector<uint32_t> ghost_depth,
        std::vector<uint32_t> local_size, std::vector<uint32_t> num_chares);


void generate_headers(FILE* genfile);


std::string generate(char* cmd, uint32_t cmd_size, int ndims, 
        std::vector<uint32_t> ghost_depth,
        std::vector<uint32_t> local_size, std::vector<uint32_t> num_chares);


compute_fun_t get_compute_fun(char* cmd, uint32_t cmd_size, int ndims, 
        std::vector<uint32_t> ghost_depth,
        std::vector<uint32_t> local_size, std::vector<uint32_t> num_chares)
{
    std::string filename = generate(cmd, cmd_size, ndims, ghost_depth, 
            local_size, num_chares);

    // compile filename
    std::string compile_cmd = fmt::format(
            "g++ {}.cpp -o {}.so -I$STENCIL_PATH -shared -fPIC -std=c++17 -O3 -g", 
            filename, filename);

    system(compile_cmd.c_str());

    // now load the compute func from the compiled shared object
    
    void* handle = dlopen((filename + ".so").c_str(), RTLD_LAZY);

    if (!handle) 
    {
        std::cerr << "Cannot open library: " << dlerror() << '\n';
        CkAbort("Couldn't open the compiled shared object file!");
    }

    compute_fun_t fun = (compute_fun_t) dlsym(handle, "compute_func");
    return fun;
}

std::string generate(char* cmd, uint32_t cmd_size, int ndims, std::vector<uint32_t> ghost_depth,
        std::vector<uint32_t> local_size, std::vector<uint32_t> num_chares)
{
    // first lookup local cache
    // if not found then lookup shared library of the name = hash of ast
    // if not found then lookup generated code of name = hash of ast
    // else generate code and compile and load function dynamically
    std::string_view graph_str(cmd, cmd_size);
    size_t graph_hash = std::hash<std::string_view>{}(graph_str);
    // TODO: Add logic to check local cache here

    std::string filename = fmt::format("/tmp/.charmstencil/stencil_{}", graph_hash);
    FILE* genfile = fopen((filename + ".cpp").c_str(), "w");
    fprintf(genfile, "#include <iostream>\n");
    fprintf(genfile, "#include <vector>\n");
    fprintf(genfile, "#include \"field.hpp\"\n\n");
    fprintf(genfile, "extern \"C\" {\n");
    fprintf(genfile, "void compute_func(std::vector<Field> &fields, "
            "std::vector<uint32_t> &num_chares, std::vector<int> &index) {\n");

    // Add some prints for debugging
    fprintf(genfile, "std::cout << \"Generated function called\\n\";\n");

    generate_code(genfile, cmd, ndims, ghost_depth, local_size, num_chares);
    fprintf(genfile, "}\n");
    fprintf(genfile, "}\n");
    fclose(genfile);
    return filename;
}


void generate_headers(FILE* genfile)
{
}


std::string generate_loop_rhs(FILE* genfile, char* &cmd, int ndims, uint32_t depth,
        std::vector<uint32_t> local_size, std::vector<uint32_t> num_chares);


void generate_code(FILE* genfile, char* &cmd, int ndims, 
        std::vector<uint32_t> ghost_depth,
        std::vector<uint32_t> local_size, 
        std::vector<uint32_t> num_chares)
{
    OperandType operand_type = lookup_type(extract<uint8_t>(cmd));
    uint8_t opcode = extract<uint8_t>(cmd);
    Operation oper = lookup_operation(opcode);
    switch (oper)
    {
        case Operation::create:
        {
           return; 
        }
        case Operation::setitem:
        {
            uint8_t ftype = extract<uint8_t>(cmd);
            uint8_t fname_optype = extract<uint8_t>(cmd);
            uint8_t fname = extract<uint8_t>(cmd);
            Slice key = get_slice(cmd, ndims, num_chares, local_size);

            uint32_t depth = ghost_depth[fname];

            fprintf(genfile, "Field& f%" PRIu8 " = fields[%" PRIu8 "];\n", fname, fname);
            fprintf(genfile, "int stop_idx[%i];\n", ndims);
            fprintf(genfile, "int step[%i];\n", ndims);

            // calculate stop index
            for(int i = 0; i < ndims; i++)
                fprintf(genfile, "stop_idx[%i] = index[%i] == "
                        "num_chares[%i] - 1 || index[%i] == 0 ? %i : %i;\n", 
                    i, i, i, i, key.index[0].stop - depth, local_size[i] - depth);

            // calculate local sizes with depth
            for(int i = 0; i < ndims; i++)
                fprintf(genfile, "step[%i] = index[%i] == "
                        "num_chares[%i] - 1 || index[%i] == 0 ? %i : %i;\n", 
                    i, i, i, i, local_size[i] + depth, local_size[i] + 2 * depth);
 
            // write the loops
            for (int i = 0; i < ndims; i++)
                fprintf(genfile, "for (int d%i = 0; d%i * %i"
                        " < stop_idx[%i]; d%i++) {\n", 
                        i, i, key.index[i].step, i, i);

            std::string index_str;

            if (ndims == 1)
                index_str = fmt::format("d0 * {} + {}", key.index[0].step, depth);
            if (ndims == 2)
                index_str = fmt::format("d0 * {} + {} + step[0] * (d1 * {} + {})", 
                        key.index[0].step, depth, key.index[1].step, depth);
            if (ndims == 3)
                index_str = fmt::format(
                        "d0 * {} + {} + step[0] * (d1 * {} + {}) + step[0] * step[1] * (d2 * {} + {})", 
                        key.index[0].step, depth, key.index[1].step, depth,
                        key.index[2].step, depth);

            fprintf(genfile, "f%i.data[%s] = %s;\n", fname, index_str.c_str(), 
                    generate_loop_rhs(genfile, cmd, ndims, depth, local_size, num_chares).c_str());

            for (int i = 0; i < ndims; i++)
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

std::string generate_loop_rhs(FILE* genfile, char* &cmd, int ndims, uint32_t depth,
        std::vector<uint32_t> local_size, std::vector<uint32_t> num_chares)
{
    OperandType operand_type = lookup_type(extract<uint8_t>(cmd));
    uint8_t opcode = extract<uint8_t>(cmd);
    Operation oper = lookup_operation(opcode);
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
                    res = fmt::format("fields[{}]", fname);
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

            op1 = generate_loop_rhs(genfile, cmd, ndims, depth, local_size, 
                            num_chares);
            op2 = generate_loop_rhs(genfile, cmd, ndims, depth, local_size, 
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
                    genfile, cmd, ndims, depth, local_size, num_chares);

            Slice key = get_slice(cmd, ndims, num_chares, local_size);
            
            std::string index_str;

            if (ndims == 1)
                index_str = fmt::format("{} + d0 * {}", 
                        key.index[0].start, key.index[0].step);
            if (ndims == 2)
                index_str = fmt::format("step[0] * ({} + d0 * {}) + ({} + d1 * {})", 
                        key.index[1].start, key.index[1].step,
                        key.index[0].start, key.index[0].step);
            if (ndims == 3)
                index_str = fmt::format(
                        "step[0] * step[1] * ({} + d0 * {}) + step[0] * ({} + d1 * {}) + ({} + d2 * {})", 
                        key.index[2].start, key.index[2].step,
                        key.index[1].start, key.index[1].step,
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
