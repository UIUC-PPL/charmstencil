from charmstencil.ast import KernelGraph, KernelParameter, get_next_kernel_id, get_parameter_state

class KernelGraphSet(object):
    def __init__(self):
        self.graphs = {}
        self.identifiers = {}

    def add_graph(self, graph):
        from charmstencil.array import Array
        if not isinstance(graph, KernelGraph):
            raise ValueError('Graph must be a KernelGraph')
        if graph.identifier not in self.identifiers:
            self.identifiers[graph.identifier] = graph
            graph.kernel_id = get_next_kernel_id()
            self.graphs[graph.kernel_id] = graph
        else:
            graph.kernel_id = self.identifiers[graph.identifier].kernel_id
        
    def get_graph(self, knl_id):
        if knl_id not in self.graphs:
            raise ValueError(f'Kernel graph {knl_id} not found')
        return self.graphs[knl_id]
    
    def remove_graph(self, knl_id):
        if knl_id in self.graphs:
            del self.graphs[knl_id]
            for identifier, graph in list(self.identifiers.items()):
                if graph.kernel_id == knl_id:
                    del self.identifiers[identifier]
                    break
    

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