#include "dag.hpp"

DAGNode::DAGNode()
{
    is_leaf = false;
    status = NodeStatus::UnVisited;
    future = CkLocalFuture();
}

std::vector<DAGNode*> build_dag(char* cmd, std::unordered_map<int, DAGNode*>& node_cache)
{
    // this is done in 2 steps
    // first build the node cache
    // then read the dependencies and build the dag

    int num_nodes = extract<int>(cmd);

    for (int i = 0; i < num_nodes; i++)
    {
        int node_type = extract<int>(cmd);
        if (node_type == static_cast<int>(DAGNodeType::Array))
        {
            ArrayDAGNode* node = new ArrayDAGNode();
            node->node_id = extract<int>(cmd);
            node->name = extract<int>(cmd);
            int ndims = extract<int>(cmd);
            for (int j = 0; j < ndims; j++)
                node->shape.push_back(extract<int>(cmd));
            node_cache[node->node_id] = node;
        }
        else if (node_type == static_cast<int>(DAGNodeType::Kernel))
        {
            KernelDAGNode* node = new KernelDAGNode();
            node->node_id = extract<int>(cmd);
            node->kernel_id = extract<int>(cmd);
            int num_inputs = extract<int>(cmd);
            for (int j = 0; j < num_inputs; j++)
                node->inputs.push_back(extract<int>(cmd));
            node_cache[node->node_id] = node;
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