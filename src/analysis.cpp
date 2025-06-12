#include "analysis.hpp"

void traverse_ast(ASTNode* node, Context* ctx)
{
    if (node == nullptr || ctx == nullptr) return;

    if (auto* op_node = dynamic_cast<OperationNode*>(node)) 
    {
        auto operation = op_node->operation;
        auto& operands = op_node->operands;

        switch (operation)
        {
            case Operation::add:
            case Operation::sub:
            case Operation::mul:
            {
                traverse_ast(operands[0], ctx);
                traverse_ast(operands[1], ctx);
                break;
            }
            
            case Operation::getitem:
            {
                ctx->set_active(static_cast<ArrayNode*>(operands[0])->arg_index);
                //ctx->register_shared_memory_access(ctx->get_active());
                traverse_ast(operands[0], ctx);
                traverse_ast(operands[1], ctx);
                ctx->reset_active();
                break;
            }

            case Operation::setitem:
            {
                ctx->set_active(static_cast<ArrayNode*>(operands[0])->arg_index);
                ctx->register_output_slice(static_cast<SliceNode*>(operands[1])->slice);                
                ctx->set_lhs_slice(&(static_cast<SliceNode*>(operands[1]))->slice);
                traverse_ast(operands[2], ctx);
                ctx->reset_slice();
                ctx->reset_active();
                break;
            }

            default:
                break;
        }
    }
    else if (auto* slice_node = dynamic_cast<SliceNode*>(node)) 
    {
        if (ctx->lhs_slice != nullptr)
        {
            // this is a getitem operation
            ctx->register_access(slice_node->slice);
        }
    }
    else if (auto* array_node = dynamic_cast<ArrayNode*>(node)) 
    {
        return;
    }
    else if (auto* int_node = dynamic_cast<IntNode*>(node)) 
    {
        // Handle integer nodes if needed
        return;
    }
    else if (auto* float_node = dynamic_cast<FloatNode*>(node)) 
    {
        // Handle float nodes if needed
        return;
    }
    else if (auto* tuple_node = dynamic_cast<TupleNode*>(node)) 
    {
        // Handle tuple nodes if needed
        return;
    }
}

void finalize_analysis(Context* ctx)
{
    if (ctx == nullptr) return;

    for (auto it : ctx->active_kernel->mem_access_info)
    {
        int argname = it.first;
        auto& accesses = it.second;

        if (accesses.size() > 1)
        {
            // If there are multiple accesses, we need to register shared memory access
            ctx->register_shared_memory_access(argname);
        }
    }
}