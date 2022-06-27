import struct
from pyccs import Server


def to_bytes(value, dtype='I'):
    return struct.pack(dtype, value)


def from_bytes(bvalue, dtype='I'):
    return struct.unpack(dtype, bvalue)[0]


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

    def __del__(self):
        self.disconnect()

    def disconnect(self):
        cmd = to_bytes(self.client_id, 'B')
        self.send_command_async(Handlers.disconnection_handler, cmd)

    def delete_stencil(self, name):
        cmd = to_bytes(name, 'B')
        self.send_command_async(Handlers.delete_handler, cmd)

    def evaluate_stencil(self, stencil):
        pass

    def send_command_raw(self, handler, msg, reply_size):
        self.server.send_request(handler, 0, msg)
        return self.server.receive_response(reply_size)

    def send_command(self, handler, msg, reply_size=1, reply_type='B'):
        return from_bytes(self.send_command_raw(handler, msg, reply_size), reply_type)

    def send_command_async(self, handler, msg):
        self.server.send_request(handler, 0, msg)

    def get(self, stencil_name, field_name):
        pass

