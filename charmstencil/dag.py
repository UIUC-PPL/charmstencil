from charmstencil.interface import to_bytes
import charmstencil.kernel as kernel
import charmstencil.array as array

gid = 0
fusion_enabled = True
max_depth = 100

def set_max_depth(depth):
    global max_depth
    max_depth = depth

def get_max_depth():
    """
    Returns the current maximum depth for the DAG.

    Returns:
        int: The current maximum depth.
    """
    global max_depth
    return max_depth

def is_fusion_enabled():
    """
    Checks if fusion of kernel nodes in the DAG is enabled.

    Returns:
        bool: True if fusion is enabled, False otherwise.
    """
    global fusion_enabled
    return fusion_enabled

def disable_fusion():
    """
    Disables fusion of kernel nodes in the DAG.
    """
    global fusion_enabled
    fusion_enabled = False

def get_gid():
    """
    Returns a unique global ID for the DAG node.

    Returns:
        int: A unique global ID.
    """
    global gid
    gid += 1
    return gid

class DAGNodeType(object):
    Array = 0
    Kernel = 1

class DAGNode(object):
    """
    A class representing a node in a directed acyclic graph (DAG).
    """

    def __init__(self, name):
        """
        Initializes the DAG node with its name.

        Args:
            name (str): Name of the node.
        """
        self.name = name
        self.gid = get_gid()
        self.children = set()
        self.dependency_list = set()

    def update_dependency_list(self, node):
        self.dependency_list.add(node)
        self.dependency_list.update(node.dependency_list)

    def propogate_dependency_list(self, deps):
        """
        Propagates the dependency list to the children of this node.

        Args:
            deps (set): A set of dependencies to be added.
        """
        for dep in deps:
            self.update_dependency_list(dep)
        for child in self.children:
            child.propogate_dependency_list(deps)

    def add_edge(self, dependency):
        """
        Adds a dependency edge to the array node.

        Args:
            dependency (DAGNode): The dependency node to be added.
        """
        self.children.add(dependency)
        #print(f'Adding edge from {self.name} to {dependency.name}')

    def remove_edge(self, child):
        #print(f'Removing edge from {self.name} to {child.name}')
        if child in self.children:
            self.children.remove(child)
        #print(f'Edges after removal: {[c.name for c in self.children]}')

    def clear(self):
        """
        Clears the node by resetting its children.
        """
        self.children = set()

    def fill_plot(self, G, node_map={}, parent=None):
        """
        Fills the plot with the DAG node and its children.

        Args:
            G (networkx.Graph): The graph to be filled.
            node_map (dict): A mapping of node IDs to node names.
            next_id (int): The next available ID for the node.
            parent (DAGNode): The parent node in the graph.
        """
        #print(self.children)
        if not G.has_node(self.gid):
            #print(self.node_type, self.gid, self.name)
            if self.node_type == DAGNodeType.Array:
                G.add_node(self.gid, color='lightcoral')
            else:
                G.add_node(self.gid, color='skyblue')
            node_map[self.gid] = self.name

        if parent is not None:
            G.add_edge(parent, self.gid)

        for dep in self.children:
            dep.fill_plot(G, node_map=node_map, parent=self.gid)


class ArrayDAGNode(DAGNode):
    """
    A class representing an array node in a directed acyclic graph (DAG).
    """

    def __init__(self, name, array):
        """
        Initializes the array node with its name and associated array graph.

        Args:
            name (str): Name of the array.
            array_graph (ArrayGraph): The array graph associated with this node.
        """
        super().__init__(name)
        self.array = array
        self.node_type = DAGNodeType.Array


class KernelDAGNode(DAGNode):
    """
    A class representing a kernel node in a directed acyclic graph (DAG).
    """

    def __init__(self, name, kernel_id, inputs, output_shape=None, fusion_nodes=None):
        """
        Initializes the kernel node with its name and associated kernel graph.

        Args:
            name (str): Name of the kernel.
            kernel_graph (KernelGraph): The kernel graph associated with this node.
        """
        super().__init__(name)
        self.kernel_id = kernel_id
        self.inputs = inputs
        self.output_shape = output_shape
        self.node_type = DAGNodeType.Kernel
        if fusion_nodes is None:
            self.fusion_nodes = [self]
        else:
            self.fusion_nodes = fusion_nodes

    def check_dependencies(self, other):
        for node in self.dependency_list:
            if node in other.fusion_nodes:
                return False
            
        for node in other.dependency_list:
            if node in self.fusion_nodes:
                return False

        return True
        #return self not in other.dependency_list and other not in self.dependency_list

        # self_outputs = kernel.get_kernel_graph_set().get_graph(self.kernel_id).get_outputs(self.inputs)
        # other_outputs = kernel.get_kernel_graph_set().get_graph(other.kernel_id).get_outputs(other.inputs)

        # self_out_names = [o.name for o in self_outputs]
        # other_out_names = [o.name for o in other_outputs]

        # self_ro_inps = [inp.name for inp in self.inputs if inp.name not in self_outputs]
        # other_ro_inps = [inp.name for inp in other.inputs if inp.name not in other_outputs]

        # #print(self_out_names, other_ro_inps)

        # # check if there are any read-after-write (RAW) dependencies
        # for inp in self_ro_inps:
        #     if inp in other_out_names:
        #         return False
            
        # # check if there are any write-after-read (WAR) dependencies
        # for inp in other_ro_inps:
        #     if inp in self_out_names:
        #         return False
            
        # return True

    def check_fusion(self, other):
        # check RAW and WAR dependencies
        return self.check_dependencies(other) and self.output_shape == other.output_shape
    
    def fuse(self, other):
        """
        Fuses this kernel node with another kernel node.

        Args:
            other (KernelDAGNode): The other kernel node to be fused with this one.
        """

        # merge inputs and outputs
        fused_inputs = self.inputs + other.inputs

        # now make a new kernel graph
        self_graph = kernel.get_kernel_graph_set().get_graph(self.kernel_id)
        other_graph = kernel.get_kernel_graph_set().get_graph(other.kernel_id)

        fused_graph = self_graph.fuse(other_graph)
        kernel.get_kernel_graph_set().add_graph(fused_graph)

        # create a new node
        new_node = KernelDAGNode(
            name=f'{self.name}_{other.name}',
            kernel_id=fused_graph.kernel_id,
            inputs=fused_inputs,
            output_shape=self.output_shape,
            fusion_nodes=self.fusion_nodes + [other]
        )
        new_node.dependency_list = self.dependency_list.union(other.dependency_list)

        # clear the children of the old nodes
        self.clear()
        other.clear()
        
        return new_node


class DAG(object):
    """
    A class representing a directed acyclic graph (DAG) for task scheduling.
    """

    def __init__(self):
        """
        Initializes the DAG with tasks and their children.
        """
        # leaf nodes are the independent ArrayDAGNodes
        self.leaf_nodes = set()
        # goal nodes are the nodes without children
        self.goal_nodes = set()
        # all nodes are the nodes in the DAG
        self.all_nodes = set()

        self.nodes_seq = []

        self.edges = set()

        self.access_info = {}

    def add_kernel_call(self, graph, inputs, output_shape=None):
        kernel_node = KernelDAGNode(f"knl{graph.kernel_id}", graph.kernel_id,
                                    inputs, output_shape=output_shape)
        self.add_node(kernel_node)

        outputs = graph.get_outputs(inputs)
        # inputs are the Array objects
        for inp in inputs:
            if isinstance(inp, array.Array):
                self.add_edge(inp.dag_node, kernel_node)
                if inp not in outputs:
                    if (inp.name, inp.generation) not in self.access_info:
                        self.access_info[(inp.name, inp.generation)] = []
                    self.access_info[(inp.name, inp.generation)].append(kernel_node)

        for out in outputs:
            if (out.name, out.generation) in self.access_info:
                for node in self.access_info[(out.name, out.generation)]:
                    self.add_edge(node, kernel_node)
            out.inc_generation(kernel_node)

    def clear(self):
        """
        Clears the DAG by resetting all nodes and edges.
        """
        from charmstencil.kernel import get_kernel_graphs
        self.leaf_nodes = set()
        for node in self.all_nodes:
            if isinstance(node, KernelDAGNode):
                kernels = get_kernel_graphs()
                for knl in kernels.values():
                    if knl.kernel_id == node.kernel_id:
                        outputs = knl.get_outputs(node.inputs)
                        for output in outputs:
                            output.clear()
                            self.leaf_nodes.add(output.dag_node)
        self.goal_nodes = set()
        self.all_nodes = set()
        self.edges = set()
        self.access_info.clear()

    def serialize(self):
        # first add node information
        #print(self.all_nodes)
        
        #print("Edges list = ", [(e[0].gid, e[1].gid) for e in self.edges])
        cmd = to_bytes(len(self.all_nodes), 'i')
        for node in self.all_nodes:
            cmd += to_bytes(node.node_type, 'i')
            cmd += to_bytes(node.gid, 'i')
            if node.node_type == DAGNodeType.Array:
                cmd += to_bytes(node.name, 'i')
                cmd += to_bytes(len(node.array.shape), 'i')
                for dim in node.array.shape:
                    cmd += to_bytes(dim, 'i')
            else:
                cmd += to_bytes(node.kernel_id, 'i')
                cmd += to_bytes(len(node.inputs), 'i')
                for out in node.inputs:
                    cmd += to_bytes(out.name, 'i')

        # now add the edges
        cmd += to_bytes(len(self.edges), 'i')
        for edge in self.edges:
            cmd += to_bytes(edge[0].gid, 'i')
            cmd += to_bytes(edge[1].gid, 'i')

        # finally add the goal nodes
        cmd += to_bytes(len(self.goal_nodes), 'i')
        for node in self.goal_nodes:
            cmd += to_bytes(node.gid, 'i')

        return cmd

    def add_node(self, node):
        """
        Adds a node to the DAG.

        Args:
            node (DAGNode): The node to be added.
        """
        #print(f'Adding node {node.name} to DAG')
        #if isinstance(node, ArrayDAGNode):
        #    self.leaf_nodes.append(node)
        self.leaf_nodes.add(node)
        self.all_nodes.add(node)
        self.goal_nodes.add(node)

        self.nodes_seq.append(node)

    def replace_node(self, node, new_node):
        #print(f'Replacing node {node.name} with {new_node.name}')
        self.all_nodes.remove(node)
        if node in self.leaf_nodes:
            self.leaf_nodes.remove(node)
            self.leaf_nodes.add(new_node)
        if node in self.goal_nodes:
            self.goal_nodes.remove(node)
            self.goal_nodes.add(new_node)
            
        self.all_nodes.add(new_node)
        # remove all edges to this node
        for i, edge in enumerate(list(self.edges)):
            if edge[0] == node:
                #edge = (new_node, edge[1])
                #print(f"Replacing edge {edge[0].name} -> {edge[1].name} with {new_node.name} -> {edge[1].name}")
                new_node.add_edge(edge[1])
                edge[0].remove_edge(edge[1])
                self.edges.remove(edge)
                self.edges.add((new_node, edge[1]))
            if edge[1] == node:
                #edge = (edge[0], new_node)
                #print(f"Replacing edge {edge[0].name} -> {edge[1].name} with {edge[0].name} -> {new_node.name}")
                edge[0].add_edge(new_node)
                edge[0].remove_edge(edge[1])
                self.edges.remove(edge)
                self.edges.add((edge[0], new_node))
        #print(self.edges)

    def add_edge(self, from_node, to_node):
        """
        Adds a directed edge from one node to another in the DAG.

        Args:
            from_node (DAGNode): The source node.
            to_node (DAGNode): The destination node.
        """
        #print(f'Adding edge from {from_node.name} to {to_node.name}')
        if from_node in self.goal_nodes:
            self.goal_nodes.remove(from_node)
        if to_node in self.leaf_nodes:
            self.leaf_nodes.remove(to_node)
        self.goal_nodes.add(to_node)
        self.edges.add((from_node, to_node))
        #print(f"Adding edge from {from_node.name} (gid={from_node.gid}) to {to_node.name} (gid={to_node.gid})")
        from_node.add_edge(to_node)
        to_node.update_dependency_list(from_node)

    def fuse(self):
        n = len(self.nodes_seq)
        i = 0
        while i < n - 1:
            node1 = self.nodes_seq[i]
            node2 = self.nodes_seq[i + 1] if i + 1 < n else None

            if isinstance(node1, KernelDAGNode) and isinstance(node2, KernelDAGNode):
                if node1.check_fusion(node2):
                    fused_node = node1.fuse(node2)
                    self.replace_node(node1, fused_node)
                    self.replace_node(node2, fused_node)
                    
                    # Replace the two nodes with the new fused node
                    self.nodes_seq.pop(i + 1)
                    self.nodes_seq[i] = fused_node
                    
                    n -= 1  # Decrease the count of nodes
                else:
                    i += 1
            else:
                i += 1

        used_kernels = set()
        for node in self.all_nodes:
            if isinstance(node, KernelDAGNode):
                # check which kernels are actually used
                used_kernels.add(node.kernel_id)
        
        # remove unused kernels
        graphs = kernel.get_kernel_graph_set()
        for knl_id in list(graphs.graphs.keys()):
            if knl_id not in used_kernels:
                print(f'Removing unused kernel {knl_id}')
                graphs.remove_graph(knl_id)

    def plot(self):
        """
        Plots the DAG using networkx and matplotlib.
        """
        import networkx as nx
        from networkx.drawing.nx_agraph import graphviz_layout
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        node_map = {}
        next_id = 0

        for node in self.leaf_nodes:
            node.fill_plot(G, node_map=node_map, parent=None)

        pos = graphviz_layout(G, prog='dot')
        colors = [G.nodes[n].get('color', 'grey') for n in G.nodes()]
        plt.figure(figsize=(4.2, 9))
        nx.draw(G, pos, labels=node_map, node_size=600, font_size=14, node_color=colors)
        #plt.show()
        plt.savefig('dag.pdf', bbox_inches='tight')
        plt.close()


active_dag = DAG()

def get_active_dag():
    """
    Returns the active DAG.
    """
    global active_dag
    return active_dag


def show_dag():
    """
    Displays the DAG.
    """
    print("Plotting DAG")
    if is_fusion_enabled():
        get_active_dag().fuse()
    get_active_dag().plot()