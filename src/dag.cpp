#include "dag.hpp"

DAGNode::DAGNode()
{
    is_leaf = false;
    status = NodeStatus::UnVisited;
    done = false;
}

std::vector<DAGNode*> build_dag(char* &cmd, std::unordered_map<int, DAGNode*>& node_cache,
    std::unordered_map<int, Kernel*> &kernels, std::unordered_map<int, int> &ghost_info)
{
    // this is done in 2 steps
    // first build the node cache
    // then read the dependencies and build the dag

    DEBUG_PRINT("========================= Building DAG =========================\n");
    int num_nodes = extract<int>(cmd);
    DEBUG_PRINT("Num nodes: %i\n", num_nodes);

    for (int i = 0; i < num_nodes; i++)
    {
        int node_type = extract<int>(cmd);
        DEBUG_PRINT("Node type: %i\n", node_type);
        if (node_type == static_cast<int>(DAGNodeType::Array))
        {
            ArrayDAGNode* node = new ArrayDAGNode();
            node->node_id = extract<int>(cmd);
            node->name = extract<int>(cmd);
            int ndims = extract<int>(cmd);
            assert(ndims == 2);
            for (int j = 0; j < ndims; j++)
                node->shape.push_back(extract<int>(cmd));
            node_cache[node->node_id] = node;
            DEBUG_PRINT("Array node: %i, %i\n", node->name, node->node_id);
        }
        else if (node_type == static_cast<int>(DAGNodeType::Kernel))
        {
            KernelDAGNode* node = new KernelDAGNode();
            node->node_id = extract<int>(cmd);
            node->kernel_id = extract<int>(cmd);
            int num_inputs = extract<int>(cmd);
            for (int j = 0; j < num_inputs; j++)
            {
                node->inputs.push_back(extract<int>(cmd));
                if (ghost_info.find(node->inputs[j]) == ghost_info.end())
                    ghost_info[node->inputs[j]] = kernels[node->kernel_id]->ghost_info[j]; // FIXME this is hardcoded to 0 for now
                ghost_info[node->inputs[j]] = std::max(ghost_info[node->inputs[j]], kernels[node->kernel_id]->ghost_info[j]);
            }
            node_cache[node->node_id] = node;
            DEBUG_PRINT("Kernel node: %i, %i\n", node->kernel_id, node->node_id);
        }
    }

    // now read the dependencies
    int num_edges = extract<int>(cmd);
    for (int i = 0; i < num_edges; i++)
    {
        int src = extract<int>(cmd);
        int dst = extract<int>(cmd);
        static_cast<KernelDAGNode*>(node_cache[dst])->dependencies.push_back(node_cache[src]);
    }

    int num_goals = extract<int>(cmd);
    std::vector<DAGNode*> goals;
    for (int i = 0; i < num_goals; i++)
    {
        int goal_id = extract<int>(cmd);
        goals.push_back(node_cache[goal_id]);
    }
    return goals;
}