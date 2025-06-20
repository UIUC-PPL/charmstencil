import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
from ctypes import c_long
from charmstencil.interface import to_bytes
from copy import deepcopy


kernel_id = 0

def get_next_kernel_id():
    global kernel_id
    id = kernel_id
    kernel_id += 1
    return id


OPCODES = {'noop': 0, 'create': 1, '+': 2, '-': 3, '*': 4, 'norm': 5,
           'getitem': 6, 'setitem': 7, 'exchange_ghosts': 8}


INV_OPCODES = {v: k for k, v in OPCODES.items()}


OPERAND_TYPES = {'array': 0, 'tuple_slice': 1, 'slice': 2, 'tuple_int': 3, 'int': 4, 'float': 5}


def slice_to_bytes(key):
    start = 0 if key.start is None else key.start
    stop = np.iinfo(np.int32).max if key.stop is None else key.stop
    step = 1 if key.step is None else key.step
    cmd = to_bytes(start, 'i')
    cmd += to_bytes(stop, 'i')
    cmd += to_bytes(step, 'i')
    return cmd

def slice_to_str(key):
    start = '0' if key.start is None else key.start
    stop = '' if key.stop is None else key.stop
    step = '1' if key.step is None else key.step
    return f'{start}:{stop}:{step}'

def tuple_to_str(key):
    return '(' + ','.join([slice_to_str(k) if isinstance(k, slice) else str(k) for k in key]) + ')'


def int_to_slice(i):
    return slice(i, i + 1, 1)


class KernelParameterState(object):
    def __init__(self):
        self.param_index = 0
        self.arrays = []

    def add_array(self, arg):
        if arg not in self.arrays:
            self.arrays.append(arg)

    def get_index(self):
        print(f'Getting index {self.param_index} for array')
        index = self.param_index
        self.param_index += 1
        return index
    
    def reset(self):
        for array in self.arrays:
            array.reset_kernel_parameter()
        self.param_index = 0
        self.arrays = []


parameter_state = KernelParameterState()


def get_parameter_state():
    """Get the current parameter state."""
    global parameter_state
    return parameter_state


class KernelParameter(object):
    def __init__(self, **kwargs):
        self.index = kwargs.pop('index', None)
        if self.index is None:
            self.index = get_parameter_state().get_index()
        self.slice_key = kwargs.pop('slice_key', None)
        self.graph = kwargs.pop('graph', ParamOperationNode('noop', [self]))
        # for a new array this dag node is the independent ArrayDAGNode
        # after this array is written to by a kernel, this is the KernelDAGNode
        # that wrote to this array
        # TODO get ghost data from kwargs

    def __getitem__(self, key):
        from charmstencil.kernel import get_active_kernel_graph
        node = ParamOperationNode('getitem', [ParamOperationNode('noop', [self]), 
                                              ParamOperationNode('noop', [key])])
        return KernelParameter(index=self.index, slice_key=key, graph=node)

    def __setitem__(self, key, value):
        raise ValueError("Setting items in a view not allowed")

    def binop(self, op, other):
        if isinstance(other, KernelParameter):
            other_node = other.graph
        else:
            other_node = ParamOperationNode('noop', [other])
        node = ParamOperationNode(op, [self.graph, other_node])
        return KernelParameter(index=self.index, graph=node)

    def __add__(self, other):
        return self.binop('+', other)
    
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self.binop('-', other)

    def __mul__(self, other):
        return self.binop('*', other)

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        return self * (1 / other)

    def get(self):
        raise NotImplementedError
    

class ParamOperationNode(object):
    def __init__(self, operation, operands):
        self.opcode = OPCODES.get(operation)
        self.operands = operands
        print(f'Creating ParamOperationNode with {operation} and operands {self.operands}')

    def serialize(self):
        cmd = to_bytes(self.opcode, 'B')
        cmd += to_bytes(len(self.operands), 'i')
        for op in self.operands:
            if isinstance(op, KernelParameter):
                cmd += to_bytes(OPERAND_TYPES.get('array'), 'B')
                cmd += to_bytes(op.index, 'i')
            elif isinstance(op, int):
                cmd += to_bytes(OPERAND_TYPES.get('int'), 'B')
                cmd += to_bytes(op, 'i')
            elif isinstance(op, float):
                cmd += to_bytes(OPERAND_TYPES.get('float'), 'B')
                cmd += to_bytes(op, 'f')
            elif isinstance(op, tuple):
                tuple_type = OPERAND_TYPES.get('tuple_int')
                for k in op:
                    if isinstance(k, slice):
                        tuple_type = OPERAND_TYPES.get('tuple_slice')

                #cmd += to_bytes(len(op), 'B')
                cmd += to_bytes(tuple_type, 'B')
                for k in op:
                    if isinstance(k, slice):
                        cmd += slice_to_bytes(k)
                    elif isinstance(k, int) and tuple_type == OPERAND_TYPES.get('tuple_int'):
                        cmd += to_bytes(k, 'i')
                    elif isinstance(k, int) and tuple_type == OPERAND_TYPES.get('tuple_slice'):
                        cmd += slice_to_bytes(int_to_slice(k))
                    else:
                        raise ValueError('unrecognized operation')
            elif isinstance(op, ParamOperationNode):
                cmd += op.serialize()
            else:
                raise ValueError('unrecognized operation')
        return cmd

    def fill_plot(self, G, node_map={}, next_id=0, parent=None):
        if self.opcode == OPCODES['noop']:
            if isinstance(self.operands[0], KernelParameter):
                node_map[next_id] = 'a' + str(self.operands[0].index)
                G.add_node(next_id)
                if parent is not None:
                    G.add_edge(parent, next_id)
                return next_id + 1
            elif isinstance(self.operands[0], float) or isinstance(self.operands[0], int) or \
                isinstance(self.operands[0], slice):
                G.add_node(next_id)
                G.add_edge(parent, next_id)
                node_map[next_id] = str(self.operands[0])
                return next_id + 1
            elif isinstance(self.operands[0], tuple):
                G.add_node(next_id)
                G.add_edge(parent, next_id)
                node_map[next_id] = tuple_to_str(self.operands[0])
                return next_id + 1
        else:
            opnode = next_id
            G.add_node(next_id)
            if parent is not None:
                G.add_edge(parent, next_id)
            node_map[next_id] = INV_OPCODES.get(self.opcode, '?')
            next_id += 1

            for op in self.operands:
                # an operand can also be a double
                next_id = op.fill_plot(G, node_map=node_map, next_id=next_id,
                                      parent=opnode)
        return next_id

    def plot(self):
        G = nx.Graph()
        node_map = {}
        self.fill_plot(G, node_map=node_map)
        pos = graphviz_layout(G, prog='dot')
        nx.draw(G, pos, labels=node_map, node_size=600, font_size=10)
        plt.show()


class KernelGraph(object):
    def __init__(self):
        self.graph = []
        #self.kernel_id = get_next_kernel_id()
        self.kernel_id = None
        self.args = set()
        self.outputs = set()
        self._identifier = None

    def get_outputs(self, args):
        #print(args)
        #print([o.index for o in self.outputs])
        return [args[output.index] for output in self.outputs]

    def is_empty(self):
        return len(self.graph) == 0

    def insert(self, node):
        self.graph.append(node)

    def add_output(self, output):
        self.outputs.add(output)

    def fuse(self, other):
        """
        Fuse another KernelGraph into this one.
        """
        print(f"Fusing {self.kernel_id} with {other.kernel_id}")
        new_graph = KernelGraph()
        #new_graph.graph = self.graph + other.graph
        last_param_index = max([p.index for p in self.args]) if self.args else 0
        other_kernel = deepcopy(other)
        #print(f"Last param index: {last_param_index}")
        for p in other_kernel.args:
            p.index += (last_param_index + 1)
        new_graph.graph = self.graph + other_kernel.graph
        new_graph.args = self.args.union(other_kernel.args)
        new_graph.outputs = self.outputs.union(other_kernel.outputs)
        #print(f"new graph operands = {new_graph.graph[0].operands[0].operands[0]}, {new_graph.graph[1].operands[0].operands[0]}")
        #get_kernel_graph_set().add_graph(new_graph)
        return new_graph

    @property
    def identifier(self):
        #self.reindex()
        #cmd = to_bytes(self.kernel_id, 'i')
        if self._identifier != None:
            return self._identifier

        gcmd = to_bytes(len(self.args), 'i')
        gcmd += to_bytes(len(self.outputs), 'i')
        outputs = sorted(self.outputs, key=lambda x: x.index)
        for out in outputs:
            gcmd += to_bytes(out.index, 'i')
        gcmd += to_bytes(len(self.graph), 'i')
        for g in self.graph:
            gcmd += g.serialize()
        
        cmd = to_bytes(len(gcmd), 'i')
        cmd += gcmd
        self._identifier = cmd
        return cmd
    
    def serialize(self):
        cmd = to_bytes(self.kernel_id, 'i')
        cmd += self.identifier
        return cmd

    def fill_plot(self, G, node_map={}, next_id=0, parent=None):
        for g in self.graph:
            next_id = g.fill_plot(G, node_map=node_map, next_id=next_id,
                                  parent=parent)
        return next_id

    def plot(self):
        G = nx.Graph()
        plt.title(f"Kernel Graph: {self.kernel_id}", loc='center')
        node_map = {}
        next_id = 0
        self.fill_plot(G, node_map=node_map, next_id=next_id)
        pos = graphviz_layout(G, prog='dot')
        nx.draw(G, pos, labels=node_map, node_size=600, font_size=10)
        plt.show()
