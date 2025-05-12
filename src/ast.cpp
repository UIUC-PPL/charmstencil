#include "ast.hpp"

#define GET_START(slice, shape, i) ((slice).index[i].start < 0 ? ((shape)[i] + (slice).index[i].start) : (slice).index[i].start)
#define GET_STOP(slice, shape, i) (std::min((slice).index[i].stop <= 0 ? ((shape)[i] + (slice).index[i].stop) : (slice).index[i].stop, (shape)[i]))

Context::Context() : lhs_slice(nullptr) {}

void Context::set_active(int name)
{
    active_stack.push(name);
}

int Context::get_active()
{
    return active_stack.top();
}

void Context::set_lhs_slice(Slice* slice)
{
    lhs_slice = slice;
}

void Context::set_active_kernel(Kernel* kernel)
{
    active_kernel = kernel;
}

void Context::reset_active()
{
    active_stack.pop();
}

void Context::reset_slice()
{
    lhs_slice = nullptr;
}

void Context::reset_active_kernel()
{
    active_kernel = nullptr;
}

std::string Context::get_step()
{
    if (is_shmem(get_active())) 
        return fmt::format("blockDim.x + 2 * {}", active_kernel->ghost_info[get_active()]);
    else 
        return fmt::format("stride_{}", get_active());
}

void Context::register_output_slice(Slice& slice)
{
    if (active_kernel)
    {
        active_kernel->register_output_slice(get_active(), slice);
    }
}

void Context::register_shared_memory_access(int argname)
{
    shmem_info.insert(argname);
}

bool Context::is_shmem(int argname)
{
    return shmem_info.find(argname) != shmem_info.end();
}

std::string ArrayNode::generate_code(Context* ctx)
{
    if (ctx->is_shmem(arg_index))
        return fmt::format("sa_{}", arg_index);
    else
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
        return fmt::format("IDX2D(idy, idx, {})", ctx->get_step());
    }
    else
    {
        // this is a getitem operation
        Slice offset = slice.calculate_relative(*(ctx->lhs_slice));
        ctx->register_shared_memory_access(ctx->get_active());
        //return fmt::format("IDX2D((idy + {}) * {}, (idx + {}) * {}, {})", offset.index[1].start, offset.index[1].step, 
        //    offset.index[0].start, offset.index[0].step, ctx->get_step());
        std::string idy = ctx->is_shmem(ctx->get_active()) ? "s_idy" : "idy";
        std::string idx = ctx->is_shmem(ctx->get_active()) ? "s_idx" : "idx";
        return fmt::format("IDX2D(({} + {}) * {}, ({} + {}) * {}, {})", 
            idy, offset.index[1].start, offset.index[1].step, 
            idx, offset.index[0].start, offset.index[0].step, 
            ctx->get_step());
    }
}

std::string SliceNode::generate_index_map(Context* ctx)
{
    std::string code = fmt::format("idx = startx_{} + {} * tx;\n", ctx->get_active(), slice.index[0].step);
    code += fmt::format("idy = starty_{} + {} * ty;\n", ctx->get_active(), slice.index[1].step);

    // shared memory access
    //if (ctx->is_shmem(ctx->get_active()))
    //{
    code += fmt::format("s_idx = {} * threadIdx.x + {};\n", slice.index[0].step, 
        ctx->active_kernel->ghost_info[ctx->get_active()]);
    code += fmt::format("s_idy = {} * threadIdx.y + {};\n", slice.index[1].step, 
        ctx->active_kernel->ghost_info[ctx->get_active()]);
    //}
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
            std::string code = fmt::format("({}[{}])", operands[0]->generate_code(ctx), operands[1]->generate_code(ctx));
            ctx->reset_active();
            return code;
        }

        case Operation::setitem:
        {
            ctx->set_active(static_cast<ArrayNode*>(operands[0])->arg_index);
            ctx->register_output_slice(static_cast<SliceNode*>(operands[1])->slice);
            std::string index_map_code = static_cast<SliceNode*>(operands[1])->generate_index_map(ctx);
            
            std::string lhs_index = operands[1]->generate_code(ctx);
            ctx->set_lhs_slice(&(static_cast<SliceNode*>(operands[1]))->slice);
            std::string code = fmt::format("{}\nif({})\n{{\n{}[{}] = {};\n}}\n", index_map_code,
                static_cast<ArrayNode*>(operands[0])->generate_bounds_check(ctx),
                operands[0]->generate_code(ctx), lhs_index, operands[2]->generate_code(ctx));
            ctx->reset_slice();
            ctx->reset_active();
            return code;
        }

        default:
            break;
    }
}


void Kernel::register_output_slice(int name, Slice& slice)
{
    output_slices[name] = slice;
}

Slice Kernel::get_launch_bounds(int name, Array* array, int* chare_index)
{
    Slice slice = output_slices[name];
    Slice global_bounds;
    //DEBUG_PRINT("Calculating global_bounds on: (%i : %i), (%i : %i)\n", slice.index[0].start, slice.index[0].stop,
    //    slice.index[1].start, slice.index[1].stop);
    global_bounds.index[0].start = GET_START(slice, array->global_shape, 0);
    global_bounds.index[0].stop = GET_STOP(slice, array->global_shape, 0);
    //int stepx = slice.index[0].step;

    global_bounds.index[1].start = GET_START(slice, array->global_shape, 1);
    global_bounds.index[1].stop = GET_STOP(slice, array->global_shape, 1);
    //int stepy = slice.index[1].step;

    // now find what indices of the global bound belong to this chare
    // FIXME assumption all arrays are square

    int chare_startx = chare_index[0] * array->local_shape[0];
    int chare_stopx = (chare_index[0] + 1) * array->local_shape[0];
    int chare_starty = chare_index[1] * array->local_shape[1];
    int chare_stopy = (chare_index[1] + 1) * array->local_shape[1];

    //DEBUG_PRINT("DEBUG AST> (%i, %i) > (%i, %i) > (%i, %i)\n", chare_index[0], chare_index[1], chare_startx, chare_stopx, chare_starty, chare_stopy);

    Slice local_bounds;
    local_bounds.index[0].start = std::max(global_bounds.index[0].start, chare_startx);
    local_bounds.index[0].stop = std::max(std::min(global_bounds.index[0].stop, chare_stopx), local_bounds.index[0].start);
    local_bounds.index[0].step = slice.index[0].step;

    local_bounds.index[0].start += (array->ghost_depth - chare_startx);
    local_bounds.index[0].stop += (array->ghost_depth - chare_startx);

    local_bounds.index[1].start = std::max(global_bounds.index[1].start, chare_starty);
    local_bounds.index[1].stop = std::max(std::min(global_bounds.index[1].stop, chare_stopy), local_bounds.index[1].start);
    local_bounds.index[1].step = slice.index[1].step;

    local_bounds.index[1].start += (array->ghost_depth - chare_starty);
    local_bounds.index[1].stop += (array->ghost_depth - chare_starty);

    return local_bounds;
}

void Kernel::get_launch_params(std::vector<Slice*> &bounds, int* threads_per_block,
     int* grid_dims)
{
    int global_ntx = 0;
    int global_nty = 0;
    for(const auto& bound : bounds)
    {
        int num_threadsx = (bound->index[0].stop - bound->index[0].start) / bound->index[0].step;
        int num_threadsy = (bound->index[1].stop - bound->index[1].start) / bound->index[1].step;
        global_ntx = std::max(global_ntx, num_threadsx);
        global_nty = std::max(global_nty, num_threadsy);
    }

    choose_optimal_grid(threads_per_block, global_ntx, global_nty);

    grid_dims[0] = (global_ntx + threads_per_block[0] - 1) / threads_per_block[0];
    grid_dims[1] = (global_nty + threads_per_block[1] - 1) / threads_per_block[1];
}

std::string Kernel::generate_shared_memory_declarations(Context* ctx)
{
    std::string code = "extern __shared__ float shmem_data[];\n"
                      "int offset = 0;\n";
    
    for (const auto& argname : ctx->shmem_info)
    {
        code += fmt::format("float* sa_{} = &(shmem_data[offset]);\n", argname);
        code += fmt::format("offset += (blockDim.x + 2 * {0}) * (blockDim.y + 2 * {0});\n", 
            ghost_info[argname]);
    }

    return code;
}

std::string Kernel::generate_shared_memory_population(Context* ctx)
{
    std::string code = "";

    // local data
    for (const auto& argname : ctx->shmem_info)
    {
        DEBUG_PRINT("argname = %i, ghost = %i\n", argname, ghost_info[argname]);

        code += fmt::format("sa_{0}[IDX2D(threadIdx.y + {1}, threadIdx.x + {1}, blockDim.x + 2 * {1})]"
            " = a_{0}[IDX2D(starty_{2} + ty, startx_{2} + tx, stride_{0})];\n", 
            argname, ghost_info[argname], ctx->active_kernel->outputs[0]);

        DEBUG_PRINT("code = %s\n", code.c_str());
    }

    // ghost data
    // FIXME - fix the start, stop indices
    for (const auto& argname : ctx->shmem_info)
    {
        code += fmt::format("if (threadIdx.x < {}) {{\n", ghost_info[argname]);
        
        code += fmt::format("sa_{0}[IDX2D(threadIdx.y + {1}, threadIdx.x, blockDim.x + 2 * {1})]"
            " = a_{0}[IDX2D(starty_{2} + ty, startx_{2} + tx - {1}, stride_{0})];\n", 
            argname, ghost_info[argname], ctx->active_kernel->outputs[0]);

        code += fmt::format("sa_{0}[IDX2D(threadIdx.y + {1}, threadIdx.x + {1} + blockDim.x, blockDim.x + 2 * {1})]"
            " = a_{0}[IDX2D(starty_{2} + ty, startx_{2} + tx + blockDim.x, stride_{0})];\n", 
            argname, ghost_info[argname], ctx->active_kernel->outputs[0]);

        code += "}\n\n";

        code += fmt::format("if (threadIdx.y < {}) {{\n", ghost_info[argname]);
        
        code += fmt::format("sa_{0}[IDX2D(threadIdx.y, threadIdx.x + {1}, blockDim.x + 2 * {1})]"
            " = a_{0}[IDX2D(starty_{2} + ty - {1}, startx_{2} + tx, stride_{0})];\n", 
            argname, ghost_info[argname], ctx->active_kernel->outputs[0]);

        code += fmt::format("sa_{0}[IDX2D(threadIdx.y + {1} + blockDim.y, threadIdx.x + {1}, blockDim.x + 2 * {1})]"
            " = a_{0}[IDX2D(starty_{2} + ty + blockDim.y, startx_{2} + tx, stride_{0})];\n", 
            argname, ghost_info[argname], ctx->active_kernel->outputs[0]);

        code += "}\n\n";

        //DEBUG_PRINT("code = %s\n", code.c_str());
    }

    code += "__syncthreads();";
    return code;
}

std::string Kernel::generate_variable_declarations(Context* ctx)
{
    std::string code = "";
    code += "int idx, idy, s_idx, s_idy;\n";
    code += "int tx = blockIdx.x * blockDim.x + threadIdx.x;\n";
    code += "int ty = blockIdx.y * blockDim.y + threadIdx.y;\n";
    return code;
}

std::string Kernel::generate_body(Context* ctx)
{
    std::string code = "";
    for (auto& node : nodes)
    {
        code += fmt::format("\n{{\n{}\n{}\n}}\n", node->generate_code(ctx), generate_debug(ctx));
    }
    return code;
}

std::string Kernel::generate_debug(Context* ctx)
{
    //std::string code = "if(tx == 0 && ty == 0) printf(\"Kernel called\");\n";
    //std::string code = "printf(\"%%i, %%i, %%i, %%i\\n\", IDX2D((s_idy + 0) * 1, (s_idx + -1) * 1, stride_0), IDX2D((s_idy + 0) * 1, (s_idx + 1) * 1, stride_0), IDX2D((s_idy + -1) * 1, (s_idx + 0) * 1, stride_0), IDX2D((s_idy + 1) * 1, (s_idx + 0) * 1, stride_0));\n";
    //return code;
    return "";
}

std::string Kernel::generate_arguments(Context* ctx)
{
    std::ostringstream oss;

    for (int i = 0; i < num_args; i++)
    {
        oss << fmt::format("float* a_{0}, int stride_{0}, ", i);
    }

    for (int i = 0; i < num_outputs; i++)
    {
        oss << fmt::format("int startx_{0}, int stopx_{0}, int starty_{0}, int stopy_{0}", outputs[i]);
        if (i < num_outputs - 1)
            oss << ", ";
    }

    return oss.str();
}

std::string Kernel::generate_signature(Context* ctx)
{
    return fmt::format("extern \"C\" __global__ void compute_func({})", generate_arguments(ctx));
}

std::string Kernel::generate_code(Context* ctx)
{
    ctx->set_active_kernel(this);
    return fmt::format("{}\n{{\n{}\n{}\n{}\n{}\n}}", 
        generate_signature(ctx), 
        generate_variable_declarations(ctx),
        generate_shared_memory_declarations(ctx),
        generate_shared_memory_population(ctx),
        generate_body(ctx));
    ctx->reset_active_kernel();
}

ASTNode* build_noop(char* &cmd)
{
    OperandType operand_type = lookup_type(extract<uint8_t>(cmd));
    switch (operand_type)
    {
        case OperandType::int_t:
        {
            int value = extract<int>(cmd);
            IntNode* int_node = new IntNode();
            int_node->value = value;
            DEBUG_PRINT("IntNode: %i\n", int_node->value);
            return int_node;
        }
        case OperandType::float_t:
        {
            float value = extract<float>(cmd);
            FloatNode* float_node = new FloatNode();
            float_node->value = value;
            DEBUG_PRINT("FloatNode: %f\n", float_node->value);
            return float_node;
        }
        case OperandType::array:
        {
            int arg_index = extract<int>(cmd);
            ArrayNode* array_node = new ArrayNode();
            array_node->arg_index = arg_index;
            DEBUG_PRINT("ArrayNode: %i\n", array_node->arg_index);
            return array_node;
        }
        case OperandType::tuple_int:
        {
            int value1 = extract<int>(cmd);
            int value2 = extract<int>(cmd);
            TupleNode* tuple_node = new TupleNode();
            tuple_node->value[0] = value1;
            tuple_node->value[1] = value2;
            DEBUG_PRINT("TupleNode: (%i, %i)\n", tuple_node->value[0], tuple_node->value[1]);
            return tuple_node;
        }
        case OperandType::tuple_slice:
        {
            Slice key = get_slice(cmd);
            SliceNode* slice_node = new SliceNode();
            slice_node->slice = key;
            DEBUG_PRINT("SliceNode: (%i:%i:%i, %i:%i:%i)\n", 
                slice_node->slice.index[0].start,
                slice_node->slice.index[0].stop,
                slice_node->slice.index[0].step,
                slice_node->slice.index[1].start,
                slice_node->slice.index[1].stop,
                slice_node->slice.index[1].step);
            return slice_node;
        }
        default:
        {
            CkAbort("Invalid operand type");
        }
    }
}

ASTNode* build_ast(char* &cmd)
{
    Operation oper = lookup_operation(extract<uint8_t>(cmd));
    int num_operands = extract<int>(cmd);
    
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
            DEBUG_PRINT("OperationNode: %s\n", get_op_string(op_node->operation).c_str());
            for (int i = 0; i < num_operands; i++)
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

Kernel* build_kernel(char* &cmd)
{
    Kernel* kernel = new Kernel();
    kernel->kernel_id = extract<int>(cmd);
    DEBUG_PRINT("=============== Building new kernel ===============\n");
    DEBUG_PRINT("Kernel ID: %i\n", kernel->kernel_id);
    int len_cmd = extract<int>(cmd);
    DEBUG_PRINT("Command length: %i\n", len_cmd);
    std::string hash_string(cmd, len_cmd);
    kernel->hash = std::hash<std::string>{}(hash_string);
    DEBUG_PRINT("Hash: %zu\n", kernel->hash);
    kernel->num_args = extract<int>(cmd);
    DEBUG_PRINT("Num inputs: %i\n", kernel->num_args);
    for (int i = 0; i < kernel->num_args; i++)
    {
        kernel->ghost_info[i] = 1;
        //DEBUG_PRINT("Input %i: %i\n", i, input);
    }
    kernel->num_outputs = extract<int>(cmd);
    DEBUG_PRINT("Num outputs: %i\n", kernel->num_outputs);
    for (int i = 0; i < kernel->num_outputs; i++)
    {
        int output = extract<int>(cmd);
        kernel->outputs.push_back(output);
        //DEBUG_PRINT("Output %i: %i\n", i, output);
    }
    int num_nodes = extract<int>(cmd);
    for (int i = 0; i < num_nodes; i++)
    {
        ASTNode* node = build_ast(cmd);
        kernel->nodes.push_back(node);
    }
    return kernel;
}

void choose_optimal_grid(int* threads_per_block, int nx, int ny)
{
    // FIXME - use a better heuristic later
    threads_per_block[0] = std::min(16, nx);
    threads_per_block[0] = std::pow(2, std::floor(std::log2(threads_per_block[0])));
    threads_per_block[1] = std::min(256 / threads_per_block[0], ny);
    threads_per_block[1] = std::pow(2, std::floor(std::log2(threads_per_block[1])));
    //threads_per_block[1] = std::min(1024 / threads_per_block[0], ny);
}