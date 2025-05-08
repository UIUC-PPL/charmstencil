from charmstencil.kernel import get_active_kernel_graph
from charmstencil.dag import get_active_dag, ArrayDAGNode

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
        self.generation = kwargs.pop('generation', 0)
        # for a new array this dag node is the independent ArrayDAGNode
        # after this array is written to by a kernel, this is the KernelDAGNode
        # that wrote to this array
        self.dag_node = ArrayDAGNode(self.name, self)
        get_active_dag().add_node(self.dag_node)
        # TODO get ghost data from kwargs

    def __getitem__(self, key):
        raise RuntimeError('This operation outside of a kernel is not allowed')

    def __setitem__(self, key, value):
        raise RuntimeError('This operation outside of a kernel is not allowed')

    def __add__(self, other):
        raise RuntimeError('This operation outside of a kernel is not allowed')

    def __sub__(self, other):
        raise RuntimeError('This operation outside of a kernel is not allowed')

    def __mul__(self, other):
        raise RuntimeError('This operation outside of a kernel is not allowed')

    def __rmul__(self, other):
        raise RuntimeError('This operation outside of a kernel is not allowed')

    def __div__(self, other):
        raise RuntimeError('This operation outside of a kernel is not allowed')

    def get(self, interface):
        return interface.get(self.name)
    
    def inc_generation(self, kernel_node):
        self.dag_node = kernel_node
        self.generation += 1

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