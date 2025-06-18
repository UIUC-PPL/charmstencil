from charmstencil.ast import KernelGraph, KernelParameter, get_next_kernel_id, get_parameter_state
from charmstencil.dag import get_active_dag, get_gid, KernelDAGNode


class KernelGraphSet(object):
    def __init__(self):
        self.graphs = {}
        self.identifiers = {}

    def add_graph(self, graph):
        from charmstencil.array import Array
        if not isinstance(graph, KernelGraph):
            raise ValueError('Graph must be a KernelGraph')
        graph.args = [array.get_kernel_parameter() for array in get_parameter_state().arrays]
        if graph.identifier not in self.identifiers:
            self.identifiers[graph.identifier] = graph
            graph.kernel_id = get_next_kernel_id()
            self.graphs[graph.kernel_id] = graph
        else:
            graph.kernel_id = self.identifiers[graph.identifier].kernel_id

        dag = get_active_dag()
        kernel_node = KernelDAGNode(f"knl{graph.kernel_id}", graph.kernel_id,
                                    get_parameter_state().arrays)
        dag.add_node(kernel_node)

        # inputs are the Array objects
        for inp in get_parameter_state().arrays:
            if isinstance(inp, Array):
                dag.add_edge(inp.dag_node, kernel_node)

        outputs = graph.get_outputs(get_parameter_state().arrays)
        for out in outputs:
            out.inc_generation(kernel_node)
        
        get_parameter_state().reset()
        reset_active_kernel_graph()
        
    def get_graph(self, knl_id):
        if knl_id not in self.graphs:
            raise ValueError(f'Kernel graph {knl_id} not found')
        return self.graphs[knl_id]
    

kernel_graphs = KernelGraphSet()
active_graph = None

def get_kernel_graph_set():
    """Get the kernel graph set."""
    global kernel_graphs
    return kernel_graphs

def get_kernel_graphs():
    """Get the kernel graphs."""
    global kernel_graphs
    return kernel_graphs.graphs

def get_kernel_graph(name):
    """Get the kernel graph by name."""
    global kernel_graphs
    if name not in kernel_graphs:
        raise ValueError(f'Kernel graph {name} not found')
    return kernel_graphs[name]

def get_active_kernel_graph():
    """Get the active kernel graph."""
    global active_graph
    if active_graph is None:
        active_graph = KernelGraph()
    return active_graph

def reset_active_kernel_graph():
    """Set the active kernel graph."""
    global active_graph
    active_graph = None

def plot_kernel_graphs():
    """Plot the kernel graph."""
    global kernel_graphs

    for graph in kernel_graphs.graphs.values():
        graph.plot()