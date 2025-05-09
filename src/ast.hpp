#ifndef AST_HPP
#define AST_HPP
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <string>
#define FMT_HEADER_ONLY
#include <fmt/format.h>
#include <cuda.h>
#include "utils.hpp"
#include "array.hpp"
#include "hapi.h"


class Kernel;

class Context
{
public:
    int active;
    int prev_active;
    Slice* lhs_slice;
    Kernel* active_kernel;

    Context();

    void set_active(int name);

    void set_lhs_slice(Slice* slice);

    void set_active_kernel(Kernel* kernel);

    void reset();

    void reset_active_kernel();

    std::string get_step();

    void register_output_slice(Slice& slice);
};

class ASTNode 
{
public:
    virtual std::string generate_code(Context* ctx) = 0;
};

class ArrayNode : public ASTNode
{
public:
    int arg_index;

    std::string generate_code(Context* ctx);

    std::string generate_bounds_check(Context* ctx);
};

class SliceNode : public ASTNode
{
public:
    Slice slice;

    std::string generate_code(Context* ctx);

    std::string generate_index_map(Context* ctx);
};

class TupleNode : public ASTNode
{
public:
    int value[2];

    std::string generate_code(Context* ctx);
};

class IntNode : public ASTNode
{
public:
    int value;

    std::string generate_code(Context* ctx);
};

class FloatNode : public ASTNode
{
public:
    float value;

    std::string generate_code(Context* ctx);
};

class OperationNode : public ASTNode
{
public:
    Operation operation;
    std::vector<ASTNode*> operands;

    std::string generate_code(Context* ctx);
};

void choose_optimal_grid(int* threads_per_block, int nx, int ny);

class Kernel
{
public:
    int kernel_id;
    std::vector<ASTNode*> nodes;
    int num_args;
    int num_outputs;
    std::vector<int> outputs;
    size_t hash;
    std::unordered_map<int, Slice> output_slices;

    void register_output_slice(int name, Slice& slice);

    Slice get_launch_bounds(int name, Array* array, int* chare_index);

    void get_launch_params(std::vector<Slice*> &bounds, int* threads_per_block,
         int* grid_dims);

    std::string generate_variable_declarations(Context* ctx);

    std::string generate_debug(Context* ctx);

    std::string generate_body(Context* ctx);

    std::string generate_arguments(Context* ctx);

    std::string generate_signature(Context* ctx);

    std::string generate_code(Context* ctx);
};

ASTNode* build_noop(char* &cmd);

ASTNode* build_ast(char* &cmd);

Kernel* build_kernel(char* &cmd);

#endif // AST_HPP