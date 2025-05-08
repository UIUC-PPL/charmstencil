import struct
import numpy as np
from pyccs import Server


def to_bytes(value, dtype='I'):
    return struct.pack(dtype, value)


def from_bytes(bvalue, dtype='I'):
    return struct.unpack(dtype, bvalue)[0]


class Handlers(object):
    connection_handler = b'connect'
    disconnection_handler = b'disconnect'
    operation_handler = b'operation'
    fetch_handler = b'fetch'


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

    def execute(self):
        from charmstencil.kernel import get_kernel_graphs
        from charmstencil.dag import get_active_dag
        kernel_graphs = get_kernel_graphs()
        for kernel_graph in kernel_graphs:
            kernel_graph.plot()
        get_active_dag().plot()

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
    def __init__(self, server_ip, server_port, odf=4):
        self.server = Server(server_ip, server_port)
        self.server.connect()
        cmd = to_bytes(odf, 'i')
        self.send_command(Handlers.connection_handler, cmd)
        print("Connected to server")

    def __del__(self):
        self.disconnect()

    def execute(self):
        """
        Send the DAG and kernel graphs to backend for execution.
        """
        from charmstencil.kernel import get_kernel_graphs
        from charmstencil.dag import get_active_dag
        kernel_graphs = get_kernel_graphs()
        print(f"Sending {len(kernel_graphs)} kernel graphs to server")
        cmd = to_bytes(len(kernel_graphs), 'i')
        for kernel_graph in kernel_graphs.values():
            cmd += kernel_graph.serialize()
        dag = get_active_dag().serialize()
        dag_msg = to_bytes(len(dag), 'i')
        dag_msg += dag
        msg = to_bytes(len(cmd) + len(dag_msg), 'i')
        msg += cmd
        msg += dag_msg
        self.send_command(Handlers.operation_handler, msg)

    def disconnect(self):
        self.send_command(Handlers.disconnection_handler, '')

    def send_command_raw(self, handler, msg, reply_size):
        self.server.send_request(handler, 0, msg)
        return self.server.receive_response(reply_size)

    def send_command_raw_var(self, handler, msg):
        self.server.send_request(handler, 0, msg)
        return self.server.receive_response_message()

    def send_command(self, handler, msg, reply_size=1):
        return self.send_command_raw(handler, msg, reply_size)

    def send_command_async(self, handler, msg):
        self.server.send_request(handler, 0, msg)

    def get(self, name):
        cmd = to_bytes(name, 'i')
        buf = self.send_command_raw_var(Handlers.fetch_handler, cmd)
        return np.frombuffer(buf, dtype=np.float32)

