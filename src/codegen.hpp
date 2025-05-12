#ifndef CODEGEN_HPP
#define CODEGEN_HPP
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <cinttypes>
#include <unordered_map>
#include <string_view>
#include <dlfcn.h>
#include <cuda.h>
#include "charm++.h"
#define FMT_HEADER_ONLY
#include <fmt/format.h>
#include "hapi.h"
#include "ast.hpp"

typedef CUfunction compute_fun_t;

static std::string get_kernel_header();

Context* write_kernel(FILE* genfile, Kernel* knl);

void generate_kernel(Kernel* knl, int suffix);

/*void generate_code(FILE* genfile, char* &cmd, int ndims, std::vector<uint32_t> ghost_depth,
        std::vector<uint32_t> local_size, std::vector<uint32_t> num_chares,
        std::vector<uint8_t> &ghost_fields);

void generate_code_cuda(FILE* genfile, char* &cmd, int ndims, 
        std::vector<uint32_t> ghost_depth,
        std::vector<uint32_t> local_size, 
        std::vector<uint32_t> num_chares,
        std::vector<uint8_t> &ghost_fields);

void generate_headers(FILE* genfile);


size_t generate(char* cmd, uint32_t cmd_size, int ndims, 
        std::vector<uint32_t> ghost_depth,
        std::vector<uint32_t> local_size, std::vector<uint32_t> num_chares,
        std::vector<uint8_t> &ghost_fields);

size_t generate_cuda(char* cmd, uint32_t cmd_size, int ndims, int num_fields,
        std::vector<uint32_t> ghost_depth,
        std::vector<uint32_t> local_size, std::vector<uint32_t> num_chares,
        std::vector<uint8_t> &ghost_fields);


compute_fun_t load_compute_fun(size_t hash)
{
    std::string filename = fmt::format("./stencil_{}", hash);
    void* handle = dlopen((filename + ".so").c_str(), RTLD_LAZY);

    if (!handle) 
    {
        std::cerr << "Cannot open library: " << dlerror() << '\n';
        CkAbort("Couldn't open the compiled shared object file!");
    }

    compute_fun_t fun = (compute_fun_t) dlsym(handle, "compute_func");
    return fun;
}


size_t compile_compute_fun(char* cmd, uint32_t cmd_size, int ndims, 
        std::vector<uint32_t> ghost_depth,
        std::vector<uint32_t> local_size, std::vector<uint32_t> num_chares,
        std::vector<uint8_t> &ghost_fields)
{
    size_t hash = generate(cmd, cmd_size, ndims, ghost_depth, 
            local_size, num_chares, ghost_fields);

    std::string filename = fmt::format("stencil_{}", hash);

    // compile filename
    std::string compile_cmd = fmt::format(
            "g++ {}.cpp -o {}.so -I$STENCIL_PATH -shared -fPIC -std=c++17 -O3 -g", 
            filename, filename);

    system(compile_cmd.c_str());

    // now load the compute func from the compiled shared object
    return hash; 
}

size_t compile_compute_kernel_cuda(char* cmd, uint32_t cmd_size, int ndims, 
        int num_fields, std::vector<uint32_t> ghost_depth,
        std::vector<uint32_t> local_size, std::vector<uint32_t> num_chares,
        std::vector<uint8_t> &ghost_fields)
{
    size_t hash = generate_cuda(cmd, cmd_size, ndims, num_fields, ghost_depth, 
            local_size, num_chares, ghost_fields);

    std::string filename = fmt::format("stencil_{}", hash);

    // compile filename
    std::string compile_cmd = fmt::format(
            "nvcc -std=c++11 -arch sm_60 {}.cu -o {}.fatbin -I$STENCIL_PATH -I$CHARM_PATH -O3 -g --fatbin -lcuda", 
            filename, filename);

    system(compile_cmd.c_str());

    // now load the compute func from the compiled shared object
    return hash; 
}

size_t generate(char* cmd, uint32_t cmd_size, int ndims, std::vector<uint32_t> ghost_depth,
        std::vector<uint32_t> local_size, std::vector<uint32_t> num_chares,
        std::vector<uint8_t> &ghost_fields)
{
    // first lookup local cache
    // if not found then lookup shared library of the name = hash of ast
    // if not found then lookup generated code of name = hash of ast
    // else generate code and compile and load function dynamically
    std::string_view graph_str(cmd, cmd_size);
    size_t graph_hash = std::hash<std::string_view>{}(graph_str);
    // TODO: Add logic to check local cache here

    std::string filename = fmt::format("stencil_{}", graph_hash);
    FILE* genfile = fopen((filename + ".cpp").c_str(), "w");
    fprintf(genfile, "#include <iostream>\n");
    fprintf(genfile, "#include <vector>\n");
    fprintf(genfile, "#include \"field.hpp\"\n\n");
    fprintf(genfile, "extern \"C\" {\n");
    fprintf(genfile, "void compute_func(std::vector<Field> &fields, "
            "std::vector<uint32_t> &num_chares, std::vector<int> &index, "
            "std::vector<uint32_t> &local_size) {\n");

    // Add some prints for debugging
    //fprintf(genfile, "std::cout << \"Generated function called\\n\";\n");

    generate_code(genfile, cmd, ndims, ghost_depth, local_size, num_chares, ghost_fields);
    fprintf(genfile, "}\n}\n");
    fclose(genfile);
    return graph_hash;
}

size_t generate_cuda(char* cmd, uint32_t cmd_size, int ndims, int num_fields,
        std::vector<uint32_t> ghost_depth,
        std::vector<uint32_t> local_size, std::vector<uint32_t> num_chares,
        std::vector<uint8_t> &ghost_fields)
{
    // first lookup local cache
    // if not found then lookup shared library of the name = hash of ast
    // if not found then lookup generated code of name = hash of ast
    // else generate code and compile and load function dynamically
    std::string_view graph_str(cmd, cmd_size);
    size_t graph_hash = std::hash<std::string_view>{}(graph_str);
    // TODO: Add logic to check local cache here

    // make string of all fields as arguments
    std::string fields_args = "";
    std::string num_chares_args = "";
    std::string index_args = "";
    std::string local_size_args = "";

    for (int i = 0; i < num_fields; i++)
        fields_args += fmt::format("double* f{}, ", i);

    for (int i = 0; i < ndims; i++)
    {
        num_chares_args += fmt::format("uint32_t nchares{}, ", i);
        index_args += fmt::format("int idx{}, ", i);
        local_size_args += fmt::format("uint32_t lsz{}", i);
        if (i < ndims - 1)
            local_size_args += ", ";
    }

    std::string args = fields_args + num_chares_args + index_args + local_size_args;

    std::string filename = fmt::format("stencil_{}", graph_hash);
    FILE* genfile = fopen((filename + ".cu").c_str(), "w");
    fprintf(genfile, "#include <iostream>\n");
    fprintf(genfile, "#include \"hapi.h\"\n\n");
    fprintf(genfile, "__global__ void compute_func(%s) {\n\t", args.c_str());

    // Add some prints for debugging
    //fprintf(genfile, "std::cout << \"Generated function called\\n\";\n");

    generate_code_cuda(genfile, cmd, ndims, ghost_depth, local_size, num_chares, ghost_fields);
    fprintf(genfile, "}\n");
    fclose(genfile);
    return graph_hash;
}


void generate_headers(FILE* genfile)
{
}


std::string generate_loop_rhs(FILE* genfile, char* &cmd, int ndims, uint32_t depth,
        std::vector<uint32_t> local_size, std::vector<uint32_t> num_chares, Slice& lhs_key);


void generate_code(FILE* genfile, char* &cmd, int ndims, 
        std::vector<uint32_t> ghost_depth,
        std::vector<uint32_t> local_size, 
        std::vector<uint32_t> num_chares,
        std::vector<uint8_t> &ghost_fields)
{
    OperandType operand_type = lookup_type(extract<uint8_t>(cmd));
    uint8_t opcode = extract<uint8_t>(cmd);
    Operation oper = lookup_operation(opcode);
    switch (oper)
    {
        case Operation::setitem:
        {
            uint8_t ftype = extract<uint8_t>(cmd);
            uint8_t fname_optype = extract<uint8_t>(cmd);
            uint8_t fname = extract<uint8_t>(cmd);
            Slice key = get_slice(cmd, ndims, num_chares, local_size);

            // FIXME
            uint32_t depth = 1; //ghost_depth[fname];

            fprintf(genfile, "Field& f%" PRIu8 " = fields[%" PRIu8 "];\n", fname, fname);
            fprintf(genfile, "int stop_idx[%i];\n", ndims);
            fprintf(genfile, "int step[%i];\n", ndims);

            // calculate stop index
            // FIXME calculate the range correctly
            for(int i = 0; i < ndims; i++)
                fprintf(genfile, "stop_idx[%i] = (idx%i == 0 ? lsz%i - 1 : "
                        "(idx%i == nchares%i - 1 ? %i + lsz%i - 1 :"
                        " %i + lsz%i - 1));\n",
                    i, i, i, i, i, key.index[i].stop + depth, i, depth, i);

            // calculate local sizes with depth
            for(int i = 0; i < ndims; i++)
                fprintf(genfile, "step[%i] = idx%i == "
                        "nchares%i - 1 || idx%i == 0 ? "
                        "%i + lsz%i : %i + lsz%i;\n", 
                    i, i, i, i, depth, i, 2 * depth, i);

            fprintf(genfile, "int count = 0;\n");
 
            // write the loops
            for (int i = ndims - 1; i >= 0; i--)
                fprintf(genfile, "for (int d%i = 0; d%i * %i"
                        " < stop_idx[%i]; d%i++) {\n", 
                        i, i, key.index[i].step, i, i);

            std::string index_str;

            if (ndims == 1)
                index_str = fmt::format("d0 * {} + {}", key.index[0].step, key.index[0].start);
            if (ndims == 2)
                index_str = fmt::format("d0 * {} + {} + step[0] * (d1 * {} + {})", 
                        key.index[0].step, key.index[0].start, key.index[1].step, key.index[1].step);
            if (ndims == 3)
                index_str = fmt::format(
                        "d0 * {} + {} + step[0] * (d1 * {} + {}) + step[0] * step[1] * (d2 * {} + {})", 
                        key.index[0].step, key.index[0].start, key.index[1].step, key.index[1].start,
                        key.index[2].step, key.index[2].start);

            //fprintf(genfile, "count++;\n");

            fprintf(genfile, "f%i.data[%s] = %s;\n", fname, index_str.c_str(), 
                    generate_loop_rhs(genfile, cmd, ndims, depth, local_size, num_chares, key).c_str());

            for (int i = 0; i < ndims; i++)
                fprintf(genfile, "}\n");

            //fprintf(genfile, "std::cout << \"Loop iterations = \" << count << std::endl;\n");
            break;
        }
        case Operation::norm:
        {}
        case Operation::exchange_ghosts:
        {
            // FIXME going to assume the first statement is ghost exchange 
            uint8_t num_ghost_fields = extract<uint8_t>(cmd);
            for (int i = 0; i < num_ghost_fields; i++)
            {
                uint8_t fname = extract<uint8_t>(cmd);
                ghost_fields.push_back(fname);
            }
            generate_code(genfile, cmd, ndims, ghost_depth, local_size, num_chares, ghost_fields);
            break;
        }
        default:
        {}
    }
}

void generate_code_cuda(FILE* genfile, char* &cmd, int ndims, 
        std::vector<uint32_t> ghost_depth,
        std::vector<uint32_t> local_size, 
        std::vector<uint32_t> num_chares,
        std::vector<uint8_t> &ghost_fields)
{
    OperandType operand_type = lookup_type(extract<uint8_t>(cmd));
    uint8_t opcode = extract<uint8_t>(cmd);
    Operation oper = lookup_operation(opcode);
    switch (oper)
    {
        case Operation::setitem:
        {
            uint8_t ftype = extract<uint8_t>(cmd);
            uint8_t fname_optype = extract<uint8_t>(cmd);
            uint8_t fname = extract<uint8_t>(cmd);
            Slice key = get_slice(cmd, ndims, num_chares, local_size);

            // FIXME
            uint32_t depth = 1; //ghost_depth[fname];

            fprintf(genfile, "int start_idx[%i];\n\t", ndims);
            fprintf(genfile, "int stop_idx[%i];\n\t", ndims);
            fprintf(genfile, "int step[%i];\n\t", ndims);

            char dims[3] = {'x', 'y', 'z'};

            for(int i = 0; i < ndims; i++)
                fprintf(genfile, "int d%i = threadIdx.%c + blockDim.%c * blockIdx.%c;\n\t", 
                    i, dims[i], dims[i], dims[i]);

            fprintf(genfile, "if (d0 == 0 && d1 == 0 && d2 == 0) printf(\"Test\\n\");\n\t");

            // calculate stop index
            // FIXME calculate the range correctly
            for(int i = 0; i < ndims; i++)
                fprintf(genfile, "start_idx[%i] = idx%i == 0 ? %i : %u;\n\t",
                    i, i, key.index[i].start + depth, depth);

            for(int i = 0; i < ndims; i++)
                fprintf(genfile, "stop_idx[%i] = idx%i == nchares%i - 1 ? lsz%i + %i : lsz%i + %u;\n\t",
                    i, i, i, i, key.index[i].stop + depth, i, depth);

            // calculate local sizes with depth
            for(int i = 0; i < ndims; i++)
                fprintf(genfile, "step[%i] = %i + lsz%i;\n\t", i, 2 * depth, i);

            // check for bounds
            if (ndims == 1)
                fprintf(genfile, "if(d0 * %i >= stop_idx[0]) ",
                    key.index[0].step);
            if (ndims == 2)
                fprintf(genfile, "if(d0 * %i >= stop_idx[0] || d1 * %i >= stop_idx[1]) ",
                    key.index[0].step, key.index[1].step);
            if (ndims == 3)
                fprintf(genfile, "if(d0 * %i >= stop_idx[0] || d1 * %i >= stop_idx[1] || d2 * %i >= stop_idx[2]) ",
                    key.index[0].step, key.index[1].step, key.index[2].step);
            fprintf(genfile, "return;\n\t");

            if (ndims == 1)
                fprintf(genfile, "if(d0 * %i < start_idx[0]) ",
                    key.index[0].step);
            if (ndims == 2)
                fprintf(genfile, "if(d0 * %i < start_idx[0] || d1 * %i < start_idx[1]) ",
                    key.index[0].step, key.index[1].step);
            if (ndims == 3)
                fprintf(genfile, "if(d0 * %i < start_idx[0] || d1 * %i < start_idx[1] || d2 * %i < start_idx[2]) ",
                    key.index[0].step, key.index[1].step, key.index[2].step);
            fprintf(genfile, "return;\n\t");

            std::string index_str;

            if (ndims == 1)
                index_str = fmt::format("d0 * {}", key.index[0].step);
            if (ndims == 2)
                index_str = fmt::format("d0 * {} + step[0] * d1 * {}", 
                        key.index[0].step, key.index[1].step);
            if (ndims == 3)
                index_str = fmt::format(
                        "d0 * {} + step[0] * d1 * {} + step[0] * step[1] * d2 * {}",
                        key.index[0].step, key.index[1].step, key.index[2].step);

            fprintf(genfile, "f%i[%s] = %s;\n\t", fname, index_str.c_str(), 
                    generate_loop_rhs(genfile, cmd, ndims, depth, local_size, num_chares, key).c_str());

            break;
        }
        case Operation::norm:
        {}
        case Operation::exchange_ghosts:
        {
            // FIXME going to assume the first statement is ghost exchange 
            uint8_t num_ghost_fields = extract<uint8_t>(cmd);
            for (int i = 0; i < num_ghost_fields; i++)
            {
                uint8_t fname = extract<uint8_t>(cmd);
                ghost_fields.push_back(fname);
            }
            generate_code_cuda(genfile, cmd, ndims, ghost_depth, local_size, num_chares, ghost_fields);
            break;
        }
        default:
        {}
    }
}

std::string generate_loop_rhs(FILE* genfile, char* &cmd, int ndims, uint32_t depth,
        std::vector<uint32_t> local_size, std::vector<uint32_t> num_chares, Slice& lhs_key)
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
                    res = fmt::format("f{}", fname);
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
                            num_chares, lhs_key);
            op2 = generate_loop_rhs(genfile, cmd, ndims, depth, local_size, 
                            num_chares, lhs_key);

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
                    genfile, cmd, ndims, depth, local_size, num_chares, lhs_key);

            Slice key = get_slice(cmd, ndims, num_chares, local_size);
            
            std::string index_str;

            if (ndims == 1)
                index_str = fmt::format("{} + d0 * {}", 
                        key.index[0].start - lhs_key.index[0].start, key.index[0].step);
            if (ndims == 2)
                index_str = fmt::format("step[0] * ({} + d1 * {}) + ({} + d0 * {})", 
                        key.index[1].start - lhs_key.index[1].start, key.index[1].step,
                        key.index[0].start - lhs_key.index[0].start, key.index[0].step);
            if (ndims == 3)
                index_str = fmt::format(
                        "step[0] * step[1] * ({} + d2 * {}) + step[0] * ({} + d1 * {}) + ({} + d0 * {})", 
                        key.index[2].start - lhs_key.index[2].start, key.index[2].step,
                        key.index[1].start - lhs_key.index[1].start, key.index[1].step,
                        key.index[0].start - lhs_key.index[0].start, key.index[0].step);

            std::string result_str = fmt::format("({})[{}]", f_str, index_str);

            return result_str;
        }
        default:
        {
            return "";
        }
    }
}*/
#endif // CODEGEN_HPP