#ifndef DAG_HPP
#define DAG_HPP
#include "codegen.hpp"

enum class DAGNodeType : int
{
    Array = 0,
    Kernel = 1,
};

enum class NodeStatus : bool
{
    Visited = 0,
    UnVisited = 1,
};

class DAGNode
{
public:
    int node_id;
    bool is_leaf;
    NodeStatus status;
    CkLocalFuture future;

    virtual ~DAGNode() = default;

    DAGNode();
};

class ArrayDAGNode : public DAGNode
{
public:
    int name;
    std::vector<int> shape;

    ArrayDAGNode() : DAGNode()
    {}
};

class KernelDAGNode : public DAGNode
{
public:
    int kernel_id;
    // dag nodes that this node depends on
    std::vector<DAGNode*> dependencies;
    // array input names to this kernel
    std::vector<int> inputs;

    KernelDAGNode() : DAGNode()
    {}
};

std::vector<DAGNode*> build_dag(char* cmd, std::unordered_map<int, DAGNode*>& node_cache);

#endif // DAG_HPP