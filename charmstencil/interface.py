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


class Interface(object):
    def __init__(self):
        pass

    def delete_stencil(self, name):
        raise NotImplementedError('delete_stencil called from base class')

    def evaluate_stencil(self, stencil_graph):
        raise NotImplementedError('evaluate_stencil called from base class')

    def get(self, stencil_name, field_name):
        raise NotImplementedError('get called from base class')


class DummyInterface(Interface):
    def __init__(self):
        pass

    def delete_stencil(self, name):
        pass

    def evaluate_stencil(self, stencil_graph):
        pass

    def get(self, stencil_name, field_name):
        return None


class DebugInterface(Interface):
    def __init__(self):
        pass

    def delete_stencil(self, name):
        pass

    def evaluate_stencil(self, stencil_graph):
        stencil_graph.plot()

    def get(self, stencil_name, field_name):
        return None


class CCSInterface(Interface):
    def __init__(self, server_ip, server_port):
        self.server = Server(server_ip, server_port)
        self.server.connect()
        self.client_id = self.send_command(Handlers.connection_handler, '')

    def __del__(self):
        self.disconnect()

    def disconnect(self):
        cmd = to_bytes(self.client_id, 'B')
        self.send_command_async(Handlers.disconnection_handler, cmd)

    def delete_stencil(self, name):
        cmd = to_bytes(name, 'B')
        self.send_command_async(Handlers.delete_handler, cmd)

    def evaluate_stencil(self, stencil_graph):
        '''
        1. epoch
        2. number of new unique graphs
        for each new graph
            1. size of graph
            2. graph
        3. array of graph index
        '''
        cmd = to_bytes(stencil_graph.stencil.name, 'B')
        cmd += to_bytes(stencil_graph.epoch, 'I')
        gcmd = to_bytes(len(stencil_graph.unique_graphs) - \
                        stencil_graph.next_graph, 'B')
        for g in stencil_graph.unique_graphs[stencil_graph.next_graph:]:
            gcmd += to_bytes(len(g.identifier), 'I')
            gcmd += g.identifier
        gcmd += to_bytes(len(stencil_graph.graphs), 'I')
        for graph in stencil_graph.graphs:
            gcmd += to_bytes(graph, 'B')
        cmd += to_bytes(len(gcmd), 'I')
        cmd += gcmd
        stencil_graph.next_graph = len(stencil_graph.unique_graphs)
        send_command_async(Handlers.operation_handler, cmd)

    def send_command_raw(self, handler, msg, reply_size):
        self.server.send_request(handler, 0, msg)
        return self.server.receive_response(reply_size)

    def send_command(self, handler, msg, reply_size=1, reply_type='B'):
        return from_bytes(self.send_command_raw(handler, msg, reply_size), reply_type)

    def send_command_async(self, handler, msg):
        self.server.send_request(handler, 0, msg)

    def get(self, stencil_name, field_name):
        pass

