from charmstencil.interface import to_bytes

gid = 0

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
        self.children = []

    def add_edge(self, dependency):
        """
        Adds a dependency edge to the array node.

        Args:
            dependency (DAGNode): The dependency node to be added.
        """
        self.children.append(dependency)
        #print(f'Adding edge from {self.name} to {dependency.name}')

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
            G.add_node(self.gid)
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

    def __init__(self, name, kernel_id, inputs):
        """
        Initializes the kernel node with its name and associated kernel graph.

        Args:
            name (str): Name of the kernel.
            kernel_graph (KernelGraph): The kernel graph associated with this node.
        """
        super().__init__(name)
        self.kernel_id = kernel_id
        self.inputs = inputs
        self.node_type = DAGNodeType.Kernel


class DAG(object):
    """
    A class representing a directed acyclic graph (DAG) for task scheduling.
    """

    def __init__(self):
        """
        Initializes the DAG with tasks and their children.
        """
        # leaf nodes are the independent ArrayDAGNodes
        self.leaf_nodes = []
        # goal nodes are the nodes without children
        self.goal_nodes = set()
        # all nodes are the nodes in the DAG
        self.all_nodes = set()

        self.edges = []

    def serialize(self):
        # first add node information
        #print(self.all_nodes)
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
        if isinstance(node, ArrayDAGNode):
            self.leaf_nodes.append(node)
        self.all_nodes.add(node)
        self.goal_nodes.add(node)

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
        self.goal_nodes.add(to_node)
        self.edges.append((from_node, to_node))
        from_node.add_edge(to_node)

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
        nx.draw(G, pos, labels=node_map, node_size=600, font_size=10)
        plt.show()


active_dag = DAG()

def get_active_dag():
    """
    Returns the active DAG.
    """
    global active_dag
    return active_dag

def compute():
    get_active_dag().compute()

def show_dag():
    """
    Displays the DAG.
    """
    get_active_dag().plot()