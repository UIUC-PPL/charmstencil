from charmstencil.kernel import get_active_kernel_graph, get_kernel_graph_set, reset_active_kernel_graph
from charmstencil.dag import get_active_dag, ArrayDAGNode
from charmstencil.ast import KernelGraph, KernelParameter, ParamOperationNode, get_parameter_state

next_name = 0

def get_next_name():
    global next_name
    name = next_name
    next_name += 1
    return name

class Array(object):
    def __init__(self, fname, shape, **kwargs):
        self.name = fname
        self.shape = shape
        if isinstance(shape, int):
            self._key_type = int
            self._slice_type = slice
        else:
            self._key_type = tuple
            self._slice_type = tuple
        self.slice_key = kwargs.pop('slice_key', None)
        # for a new array this dag node is the independent ArrayDAGNode
        # after this array is written to by a kernel, this is the KernelDAGNode
        # that wrote to this array
        self.dag_node = ArrayDAGNode(self.name, self)
        self.base_node = self.dag_node
        self.kernel_param = None
        self.generation = 0
        get_active_dag().add_node(self.dag_node)
        # TODO get ghost data from kwargs

    def reset_kernel_parameter(self):
        """
        Reset the kernel parameter for this array.
        """
        self.kernel_param = None

    def get_kernel_parameter(self):
        """
        Get the kernel parameter for this array.
        If it does not exist, create it.
        """
        if self.kernel_param is None:
            self.kernel_param = KernelParameter()
            get_parameter_state().add_array(self)
            get_active_kernel_graph().args.add(self.kernel_param)
        return self.kernel_param

    def binop(self, op, other):
        if isinstance(other, Array):
            node = ParamOperationNode(op, [self.get_kernel_parameter(),
                                           other.get_kernel_parameter()])
            return KernelParameter(index=self.kernel_param.index, graph=node)
        elif isinstance(other, KernelParameter):
            node = ParamOperationNode(op, [self.get_kernel_parameter(),
                                           other.get_kernel_parameter()])
            return KernelParameter(index=self.kernel_param.index, graph=node)
        else:
            node = ParamOperationNode(op, [self.get_kernel_parameter(),
                                           ParamOperationNode('noop', [other.get_kernel_parameter()])])
            return KernelParameter(index=self.kernel_param.index, graph=node)

    def __getitem__(self, key):
        node = ParamOperationNode('getitem', [ParamOperationNode('noop', [self.get_kernel_parameter()]), 
                                              ParamOperationNode('noop', [key])])
        return KernelParameter(index=self.kernel_param.index, slice_key=key, graph=node)

    def __setitem__(self, key, value):
        if isinstance(value, KernelParameter):
            value_node = value.graph
        elif isinstance(value, float) or isinstance(value, int):
            value_node = ParamOperationNode('noop', [value])
        else:
            raise TypeError('Value must be an array slice, int, or float')
        node = ParamOperationNode('setitem', [ParamOperationNode('noop', [self.get_kernel_parameter()]), 
                                              ParamOperationNode('noop', [key]), 
                                              value_node])
        active_graph = get_active_kernel_graph()
  
        active_graph.insert(node)
        active_graph.add_output(self.get_kernel_parameter())
        # FIXME check if this works

        # now reset kernel parameter for me and everyone in the value
        get_kernel_graph_set().add_graph(active_graph, get_parameter_state().arrays, output_shape=key)
        get_active_dag().add_kernel_call(active_graph, get_parameter_state().arrays, output_shape=key)
        get_parameter_state().reset()
        reset_active_kernel_graph()

    def __add__(self, other):
        return self.binop('+', other)
    
    def __radd__(self, other):
        return self.binop('+', other)

    def __sub__(self, other):
        return self.binop('-', other)

    def __mul__(self, other):
        return self.binop('*', other)

    def __rmul__(self, other):
        return self.binop('*', other)

    def __div__(self, other):
        return self.binop('/', other)

    def get(self, interface):
        return interface.get(self.name)
    
    def inc_generation(self, kernel_node):
        self.dag_node = kernel_node
        self.generation += 1

    def clear(self):
        self.dag_node = self.base_node
        self.dag_node.clear()

def create_array(shape=None, **kwargs):
    """
    Create an array with the given shape and optional parameters.
    
    Args:
        shape (tuple or int): Shape of the array. If int, creates a 1D array.
        **kwargs: Additional parameters for the array.
    
    Returns:
        Array: An instance of the Array class.
    """
    if shape is None:
        raise ValueError('Shape must be provided')
    elif len(shape) != 2:
        raise ValueError('Only 2D arrays currently supported')
    return Array(get_next_name(), shape, **kwargs)

def create_array_like(array, **kwargs):
    """
    Create an array with the same shape as the given array.
    
    Args:
        array (Array): The array to copy the shape from.
        **kwargs: Additional parameters for the new array.
    
    Returns:
        Array: An instance of the Array class with the same shape as the input array.
    """
    return create_array(array.shape, **kwargs)