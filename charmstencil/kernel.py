from charmstencil.ast import KernelGraph, KernelParameter
from charmstencil.dag import get_active_dag, KernelDAGNode

kernel_graphs = {}
active_graph = None

def get_kernel_graphs():
    """Get the kernel graphs."""
    global kernel_graphs
    return kernel_graphs

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
        raise ValueError('No active kernel graph')
    return active_graph

def capture_kernel_graph(f, args):
    """Capture the kernel graph from the function."""
    global active_graph
    active_graph = KernelGraph(f.__name__)
    active_graph.args = [KernelParameter(idx) for idx, arg in enumerate(args)]
    f(*active_graph.args)
    kernel_graphs[f.__name__] = active_graph
    active_graph = None

def kernel(f):
    """Decorator to mark a function as a kernel."""
    f.is_kernel = True
    
    def wrapper(*args):
        # make a new graph for each kernel
        global kernel_graphs
        global active_graph
        if f.__name__ not in kernel_graphs:
            capture_kernel_graph(f, args)
        active_graph = kernel_graphs[f.__name__]
        outputs = active_graph.get_outputs(args)
        # now find the inputs and outputs and add call to the DAG
        dag = get_active_dag()
        kernel_node = KernelDAGNode(f.__name__, active_graph.kernel_id, args)
        dag.add_node(kernel_node)
        # inputs are the Array objects
        for inp in args:
            dag.add_edge(inp.dag_node, kernel_node)

        for out in outputs:
            out.inc_generation(kernel_node)

        active_graph = None
    return wrapper

def plot_kernel_graphs():
    """Plot the kernel graph."""
    global kernel_graphs

    for graph in kernel_graphs.values():
        graph.plot()