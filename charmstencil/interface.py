import struct
from pyccs import Server


def to_bytes(value, dtype='I'):
    return struct.pack(dtype, value)


def from_bytes(bvalue, dtype='I'):
    return struct.unpack(dtype, bvalue)[0]


class Handlers(object):
    connection_handler = b'aum_connect'
    disconnection_handler = b'aum_disconnect'
    operation_handler = b'aum_operation'
    fetch_handler = b'aum_fetch'
    delete_handler = b'aum_delete'
    exit_handler = b'aum_exit'
    sync_handler = b'aum_sync'
    create_handler = b'aum_create'


class Interface(object):
    def __init__(self):
        pass

    def delete_stencil(self, name):
        raise NotImplementedError('delete_stencil called from base class')

    def evaluate_stencil(self, stencil):
        raise NotImplementedError('evaluate_stencil called from base class')

    def get(self, stencil_name, field_name):
        raise NotImplementedError('get called from base class')


class DummyInterface(Interface):
    def __init__(self):
        pass

    def delete_stencil(self, name):
        pass

    def evaluate_stencil(self, stencil):
        pass

    def get(self, stencil_name, field_name):
        return None


class DebugInterface(Interface):
    def __init__(self):
        pass

    def initialize_stencil(self, stencil):
        pass

    def delete_stencil(self, name):
        pass

    def evaluate_stencil(self, stencil):
        stencil.stencil_graph.plot()

    def get(self, stencil_name, field_name):
        return None


class CCSInterface(Interface):
    def __init__(self, server_ip, server_port):
        self.server = Server(server_ip, server_port)
        self.server.connect()
        self.client_id = self.send_command(Handlers.connection_handler, '')
        print(self.client_id)

    def __del__(self):
        self.disconnect()

    def disconnect(self):
        cmd = to_bytes(self.client_id, 'B')
        #self.send_command_async(Handlers.disconnection_handler, cmd)

    def sync_stencil(self, stencil):
        cmd = to_bytes(stencil.name, 'B')
        cmd += to_bytes(stencil.stencil_graph.epoch, 'I')
        res = self.send_command(Handlers.sync_handler, cmd)

    def delete_stencil(self, name):
        cmd = to_bytes(name, 'B')
        self.send_command_async(Handlers.delete_handler, cmd)

    def initialize_stencil(self, stencil):
        '''
        stencil name
        ndims
        dims
        odf
        num_fields
        for each field
            ghost depth for the field
        '''
        cmd = to_bytes(stencil.name, 'B')
        cmd += to_bytes(len(stencil.shape), 'B')
        for dim in stencil.shape:
            cmd += to_bytes(dim, 'I')
        cmd += to_bytes(stencil.odf, 'B')
        cmd += to_bytes(len(stencil._fields), 'B')
        for i in range(len(stencil._fields)):
            cmd += to_bytes(stencil._fields[i].ghost_depth, 'I')
        for b in stencil.boundary:
            cmd += to_bytes(b, 'd')
        self.send_command(Handlers.create_handler, cmd)

    def evaluate_stencil(self, stencil):
        '''
        stencil name
        1. epoch
        cmd size
        2. number of new unique graphs
        for each new graph
            1. size of graph
            2. graph
        size of graph epochs section of cmd
        number of graph epochs
        3. array of graph index
        '''
        stencil_graph = stencil.stencil_graph
        cmd = to_bytes(stencil_graph.stencil.name, 'B')
        cmd += to_bytes(stencil_graph.epoch, 'I')
        gcmd = to_bytes(len(stencil_graph.unique_graphs) - \
                        stencil_graph.next_graph, 'B')
        for g in stencil_graph.unique_graphs[stencil_graph.next_graph:]:
            cmd_graph = g.get_identifier()
            gcmd += to_bytes(len(cmd_graph), 'I')
            gcmd += cmd_graph
        #print(len(stencil_graph.graphs))
        gcmd += to_bytes(len(stencil_graph.graphs), 'I')
        for graph in stencil_graph.graphs:
            gcmd += to_bytes(graph, 'B')
        cmd += to_bytes(len(gcmd), 'I')
        cmd += gcmd
        stencil_graph.next_graph = len(stencil_graph.unique_graphs)
        self.send_command_async(Handlers.operation_handler, cmd)

    def send_command_raw(self, handler, msg, reply_size):
        self.server.send_request(handler, 0, msg)
        return self.server.receive_response(reply_size)

    def send_command(self, handler, msg, reply_size=1, reply_type='B'):
        return from_bytes(self.send_command_raw(handler, msg, reply_size), reply_type)

    def send_command_async(self, handler, msg):
        self.server.send_request(handler, 0, msg)

    def get(self, stencil_name, field_name):
        pass

