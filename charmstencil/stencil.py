import sys
import warnings
import numpy as np
from charmstencil.interface import DummyInterface
from charmstencil.ast import StencilGraph, IterateGraph, BoundaryGraph, \
    FieldOperationNode, CreateFieldNode


try:
    from typing import final
except ImportError:
    final = lambda f: f


next_stencil_name = 0


def get_stencil_name():
    global next_stencil_name
    name = next_stencil_name
    next_stencil_name += 1
    return name


class Field(object):
    def __init__(self, fname, shape, stencil, **kwargs):
        self.name = fname
        self.shape = shape
        if isinstance(shape, int):
            self._key_type = int
            self._slice_type = slice
        else:
            self._key_type = tuple
            self._slice_type = tuple
        self.stencil = stencil
        self.slice_key = kwargs.pop('slice_key', None)
        self.graph = kwargs.pop('graph', FieldOperationNode('noop', [self]))
        self.ghost_depth = kwargs.pop('ghost_depth', 1)
        # TODO get ghost data from kwargs

    def __getitem__(self, key):
        if isinstance(key, self._slice_type) or isinstance(key, self._key_type):
            node = FieldOperationNode('getitem', [self, key])
            return Field(self.name, self.shape, self.stencil, slice_key=key,
                         graph=node)

    def __setitem__(self, key, value):
        if isinstance(key, self._slice_type) or isinstance(key, self._key_type):
            node = FieldOperationNode('setitem', [self, key, value])
            self.stencil.active_graph.insert(node)

    def __add__(self, other):
        node = FieldOperationNode('+', [self, other])
        #print("returning")
        return Field(self.name, self.shape, self.stencil,
                     graph=node)

    def __sub__(self, other):
        node = FieldOperationNode('-', [self, other])
        return Field(self.name, self.shape, self.stencil,
                     graph=node)

    def __mul__(self, other):
        #print("MUL")
        node = FieldOperationNode('*', [self, other])
        return Field(self.name, self.shape, self.stencil,
                     graph=node)

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        return self * (1 / other)

    def get(self):
        if not self.stencil.active_graph.is_empty():
            self.stencil.active_graph.insert(self.graph)
        self.stencil.evaluate()
        self.graph.save = True
        self.stencil.interface.get(self.stencil.name, self.name)

    def flush(self):
        self.graph = FieldOperationNode('noop', [self])


class Stencil(object):
    def __init__(self, shape, num_fields, **kwargs):
        """Stencil base class
        """
        self.initialize(shape, num_fields, **kwargs)

    def __del__(self):
        #if self._allocated:
        self.interface.delete_stencil(self.name)

    def _call_iterate(self, *args, **kwargs):
        #print("CHECK")
        self.active_graph, prev_graph = self._iterate_graph, self.active_graph
        #print("DONE")
        ret = self.iterate(*args, **kwargs)
        #print("DONE2")
        self._iterate_graph = IterateGraph()
        if not self.active_graph.is_empty():
            self.stencil_graph.insert(self.active_graph)
        self.active_graph = prev_graph
        return ret

    def _call_boundary(self, *args, **kwargs):
        prev_graph = self.active_graph
        if self.active_graph == self.stencil_graph:
            self.active_graph = self._boundary_graph
            ret = self.boundary(*args, **kwargs)
            self._boundary_graph = BoundaryGraph()
            if not self.active_graph.is_empty():
                self.stencil_graph.insert(self.active_graph)
        elif self.active_graph == self._iterate_graph:
            ret = self.boundary(*args, **kwargs)
        else:
            raise RuntimeError("apply_boundary cannot be called here")
        self.active_graph = prev_graph
        return ret

    def sync(self):
        self.evaluate()
        self.interface.sync_stencil(self)

    def initialize(self, shape, num_fields, **kwargs):
        self.interface = kwargs.pop('interface', DummyInterface())
        max_epochs = kwargs.pop('max_epochs', 10)
        self.odf = kwargs.pop('odf', 4)
        self.name = get_stencil_name()
        self.num_fields = num_fields
        self.shape = shape
        self._next_field_id = 0
        self.stencil_graph = StencilGraph(self, max_epochs)
        self._boundary_graph = BoundaryGraph()
        self._iterate_graph = IterateGraph()
        self._fields = []
        self._allocated = False
        self.active_graph = self.stencil_graph
        self._fields = self._create_fields(num_fields, **kwargs)
        self.interface.initialize_stencil(self)
        if len(self._fields) == 1:
            return self._fields[0]
        else:
            return self._fields

    def get_field_name(self):
        field_name = self._next_field_id
        self._next_field_id += 1
        return field_name

    @final
    def _create_field(self, **kwargs):
        if self.active_graph != self.stencil_graph:
            raise RuntimeError("fields cannot be created in iterate or "
                               "boundary functions")
        name = self.get_field_name()
        f = Field(name, self.shape, self, **kwargs)
        self.stencil_graph.insert(CreateFieldNode(name, self.shape, **kwargs))
        self._fields.append(f)
        return f
        #raise NotImplementedError("Use create_fields()")
    
    @final
    def _create_fields(self, num_fields, **kwargs):
        ghost_depths = kwargs.pop('ghost_depths', 
                                  [1] * num_fields)
        fields = []
        for i in range(num_fields):
            name = self.get_field_name()
            f = Field(name, self.shape, self, 
                      ghost_depth=ghost_depths[i])
            fields.append(f)
        return fields

    #args would be num_args and field_names right?
    @final
    def exchange_ghosts(self, *args):
        if self.active_graph == self._iterate_graph:
            self.active_graph.insert(FieldOperationNode('exchange_ghosts', args))
        else:
            raise RuntimeError("exchange_ghosts should be called from iterate")

    @final
    def apply_boundary(self, *args, **kwargs):
        self._call_boundary(*args, **kwargs)

    @final
    def solve(self, *args, **kwargs):
        while(self._call_iterate(*args, **kwargs)):
            pass
        self.evaluate()

    def iterate(self, *args, **kwargs):
        raise NotImplementedError("iterate function not defined")

    def boundary(self):
        raise NotImplementedError("boundary called without definition")

    def evaluate(self):
        if self.active_graph == self._iterate_graph and \
                not self.active_graph.is_empty():
            self.stencil_graph.insert(self.active_graph)
            self._iterate_graph = IterateGraph()
            self.active_graph = self._iterate_graph
        elif self.active_graph == self._boundary_graph and \
                not self.active_graph.is_empty():
            self.stencil_graph.insert(self.active_graph)
            self._boundary_graph = BoundaryGraph()
            self.active_graph = self._boundary_graph
        if not self.stencil_graph.is_empty():
            self.interface.evaluate_stencil(self)
            self.flush()

    def flush(self):
        for f in self._fields:
            f.flush()
        self.stencil_graph.flush()

class StencilNumpy(Stencil):
    def __init__(self, shape, num_fields, **kwargs):
        """Stencil base class for numpy
        """
        self.initialize(shape, num_fields, **kwargs)

    def __del__(self):
        pass

    def _call_iterate(self, *args, **kwargs):
        return self.iterate(*args, **kwargs)

    def _call_boundary(self, *args, **kwargs):
        return self.boundary(*args, **kwargs)

    def sync(self):
        pass

    def initialize(self, shape, num_fields, **kwargs):
        max_epochs = kwargs.pop('max_epochs', 10)
        self.odf = kwargs.pop('odf', 4)
        self.name = get_stencil_name()
        self.num_fields = num_fields
        self.shape = shape
        self._next_field_id = 0
        self._fields = []
        self._allocated = False
        self._fields = self._create_fields(num_fields, **kwargs)
        if len(self._fields) == 1:
            return self._fields[0]
        else:
            return self._fields

    def get_field_name(self):
        field_name = self._next_field_id
        self._next_field_id += 1
        return field_name

    @final
    def _create_field(self, **kwargs):
        name = self.get_field_name()
        f = np.zeros(self.shape, dtype=np.float64)
        self._fields.append(f)
        return f
    
    @final
    def _create_fields(self, num_fields, **kwargs):
        fields = []
        for i in range(num_fields):
            name = self.get_field_name()
            f = np.zeros(self.shape, dtype=np.float64)
            fields.append(f)
        return fields

    #args would be num_args and field_names right?
    @final
    def exchange_ghosts(self, *args):
        pass

    def evaluate(self):
        pass

    def flush(self):
        pass

