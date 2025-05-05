#include "ast.hpp"

Context::Context() : active(-1), prev_active(-1), lhs_slice(nullptr) {}

void Context::set_active(int name)
{
    prev_active = active;
    active = name;
}

void Context::set_lhs_slice(Slice* slice)
{
    lhs_slice = slice;
}

void Context::set_active_kernel(Kernel* kernel)
{
    active_kernel = kernel;
}

void Context::reset()
{
    active = prev_active;
    lhs_slice = nullptr;
}

void Context::reset_active_kernel()
{
    active_kernel = nullptr;
}

std::string Context::get_step()
{
    return fmt::format("(stride_{})", active); // Placeholder for actual step calculation
}

void Context::register_output_slice(Slice& slice)
{
    if (active_kernel)
    {
        active_kernel->register_output_slice(active, slice);
    }
}

std::string ArrayNode::generate_code(Context* ctx)
{
    return fmt::format("a_{}", arg_index);
}

std::string ArrayNode::generate_bounds_check(Context* ctx)
{
    return fmt::format("idx >= startx_{} && idx < stopx_{} && idy >= starty_{} && idy < stopy_{}", 
        arg_index, arg_index, arg_index, arg_index);
}

std::string SliceNode::generate_code(Context* ctx)
{
    // if this is a setitem operation
    if (ctx->lhs_slice == nullptr)
    {
        return fmt::format("IDX2D(idx, idy, {})", ctx->get_step());
    }
    else
    {
        Slice offset = slice.calculate_relative(*(ctx->lhs_slice));
        return fmt::format("IDX2D((idx + {}) * {}, (idy + {}) * {}, {})", offset.index[0].start, offset.index[0].step, 
            offset.index[1].start, offset.index[1].step, ctx->get_step());
    }
}

std::string SliceNode::generate_index_map(Context* ctx)
{
    std::string code = fmt::format("idx = startx_{} + {} * tx;\n", ctx->active, slice.index[0].step);
    code += fmt::format("idy = starty_{} + {} * ty;\n", ctx->active, slice.index[1].step);
    return code;
}

std::string TupleNode::generate_code(Context* ctx)
{
    // FIXME: This will only work on single GPU
    return fmt::format("IDX2D({}, {}, {})", value[0], value[1], ctx->get_step());
}

std::string IntNode::generate_code(Context* ctx)
{
    return fmt::format("({})", value);
}

std::string FloatNode::generate_code(Context* ctx)
{
    return fmt::format("({})", value);
}

std::string OperationNode::generate_code(Context* ctx)
{
    switch (operation)
    {
        case Operation::add:
        case Operation::sub:
        case Operation::mul:
        {
            std::string opstr = get_op_string(operation);
            return fmt::format("({} {} {})", operands[0]->generate_code(ctx), opstr, operands[1]->generate_code(ctx));
        }
        
        case Operation::getitem:
        {
            ctx->set_active(static_cast<ArrayNode*>(operands[0])->arg_index);
            return fmt::format("({}[{}])", operands[0]->generate_code(ctx), operands[1]->generate_code(ctx));
            ctx->reset();
        }

        case Operation::setitem:
        {
            ctx->set_active(static_cast<ArrayNode*>(operands[0])->arg_index);
            std::string lhs_slice_code = static_cast<SliceNode*>(operands[1])->generate_index_map(ctx);
            ctx->set_lhs_slice(&(static_cast<SliceNode*>(operands[1]))->slice);
            return fmt::format("if({})\n{\n{}[{}] = {}\n}\n", 
                static_cast<ArrayNode*>(operands[0])->generate_bounds_check(ctx),
                operands[0]->generate_code(ctx), lhs_slice_code, operands[2]->generate_code(ctx));
            ctx->reset();
        }

        default:
            break;
    }
}


void Kernel::register_output_slice(int name, Slice& slice)
{
    output_slices[name] = slice;
}

Slice Kernel::get_launch_bounds(int name, Array* array)
{
    Slice slice = output_slices[name];
    Slice bounds;
    bounds.index[0].start = slice.index[0].start;
    bounds.index[0].stop = array->shape[0] + slice.index[0].stop;
    //int stepx = slice.index[0].step;

    bounds.index[1].start = slice.index[1].start;
    bounds.index[1].stop = array->shape[1] + slice.index[1].stop;
    //int stepy = slice.index[1].step;

    return bounds;
}

void Kernel::get_launch_params(std::vector<Array*> &args, int* threads_per_block,
     int* grid_dims)
{
    int global_ntx = 0;
    int global_nty = 0;
    for(const auto& entry : output_slices)
    {
        int index = entry.first;
        Slice slice = entry.second;
        Array* array = args[index];
        int startx = slice.index[0].start;
        int stopx = array->shape[0] + slice.index[0].stop;
        int stepx = slice.index[0].step;

        int starty = slice.index[1].start;
        int stopy = array->shape[1] + slice.index[1].stop;
        int stepy = slice.index[1].step;

        int num_threadsx = (stopx - startx) / stepx;
        int num_threadsy = (stopy - starty) / stepy;
        global_ntx = std::max(global_ntx, num_threadsx);
        global_nty = std::max(global_nty, num_threadsy);
    }

    choose_optimal_grid(threads_per_block, global_ntx, global_nty);

    grid_dims[0] = (global_ntx + threads_per_block[0] - 1) / threads_per_block[0];
    grid_dims[1] = (global_nty + threads_per_block[1] - 1) / threads_per_block[1];
}

std::string Kernel::generate_variable_declarations(Context* ctx)
{
    std::string code = "";
    code += "int idx, idy;\n";
    code += "int tx = blockIdx.x * blockDim.x + threadIdx.x;\n";
    code += "int ty = blockIdx.y * blockDim.y + threadIdx.y;\n";
    return code;
}

std::string Kernel::generate_body(Context* ctx)
{
    std::string code = "";
    for (auto& node : nodes)
    {
        code += fmt::format("\n{\n{}\n}\n", node->generate_code(ctx));
    }
    return code;
}

std::string Kernel::generate_arguments(Context* ctx)
{
    std::string code = "";
    for (int i = 0; i < num_outputs; i++)
    {
        code += fmt::format("a_{}, startx_{}, stopx_{}, starty_{}, stopy_{}, stride_{}", i, i, i, i, i, i);
        if (i < num_inputs - 1)
            code += ", ";
    }

    for (int i = num_outputs; i < num_inputs; i++)
    {
        code += fmt::format("a_{}, stride_{}", i, i);
        if (i < num_inputs - 1)
            code += ", ";
    }
}

std::string Kernel::generate_signature(Context* ctx)
{
    return fmt::format("extern \"C\" __global__ void compute_func({})", generate_arguments(ctx));
}

std::string Kernel::generate_code(Context* ctx)
{
    ctx->set_active_kernel(this);
    return fmt::format("{}\n{\n{}\n{}\n}", generate_signature(ctx), generate_variable_declarations(ctx), generate_body(ctx));
    ctx->reset_active_kernel();
}

ASTNode* build_noop(char* cmd)
{
    OperandType operand_type = lookup_type(extract<uint8_t>(cmd));
    switch (operand_type)
    {
        case OperandType::int_t:
        {
            int value = extract<int>(cmd);
            IntNode* int_node = new IntNode();
            int_node->value = value;
            return int_node;
        }
        case OperandType::float_t:
        {
            float value = extract<float>(cmd);
            FloatNode* float_node = new FloatNode();
            float_node->value = value;
            return float_node;
        }
        case OperandType::array:
        {
            int arg_index = extract<int>(cmd);
            ArrayNode* array_node = new ArrayNode();
            array_node->arg_index = arg_index;
            return array_node;
        }
        case OperandType::tuple_int:
        {
            int value1 = extract<int>(cmd);
            int value2 = extract<int>(cmd);
            TupleNode* tuple_node = new TupleNode();
            tuple_node->value[0] = value1;
            tuple_node->value[1] = value2;
            return tuple_node;
        }
        case OperandType::tuple_slice:
        {
            Slice key = get_slice(cmd);
            SliceNode* slice_node = new SliceNode();
            slice_node->slice = key;
            return slice_node;
        }
        default:
        {
            CkAbort("Invalid operand type");
        }
    }
}

ASTNode* build_ast(char* cmd)
{
    Operation oper = lookup_operation(extract<uint8_t>(cmd));
    
    switch (oper)
    {
        case Operation::noop:
        {
            return build_noop(cmd);
        }
        case Operation::getitem:
        case Operation::add:
        case Operation::sub:
        case Operation::mul:
        case Operation::setitem:
        {
            OperationNode* op_node = new OperationNode();
            op_node->operation = oper;
            op_node->operands.push_back(build_ast(cmd));
            return op_node;
        }
        default:
        {
            CkAbort("Invalid operation");
        }
    }

    return nullptr;
}

Kernel* build_kernel(char* cmd)
{
    Kernel* kernel = new Kernel();
    kernel->kernel_id = extract<int>(cmd);
    int len_cmd = extract<int>(cmd);
    std::string hash_string(cmd, len_cmd);
    kernel->hash = std::hash<std::string>{}(hash_string);
    kernel->num_inputs = extract<int>(cmd);
    kernel->num_outputs = extract<int>(cmd);
    int num_nodes = extract<int>(cmd);
    for (int i = 0; i < num_nodes; i++)
    {
        ASTNode* node = build_ast(cmd);
        kernel->nodes.push_back(node);
    }
    return kernel;
}

void choose_optimal_grid(int* &threads_per_block, int nx, int ny)
{
    // FIXME - use a better heuristic later
    threads_per_block[0] = 32;
    threads_per_block[1] = 32;
}