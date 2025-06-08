import socket
import pickle
import struct
from phi3_partial.con.abstract_client import AbstractClient


class PartialForwardClient(AbstractClient):
    def send_partial_forward(self, input_data):
        data = input_data 
        for host, port in self.servers:
            with socket.create_connection((host, port)) as s:
                serialized = pickle.dumps(data)

                # Prefixa com 4 bytes o tamanho
                s.sendall(struct.pack('!I', len(serialized)))
                s.sendall(serialized)

                # Lê primeiro o tamanho
                size_data = s.recv(4)
                expected_size = struct.unpack('!I', size_data)[0]
                # Agora lê exatamente expected_size bytes
                response = b''
                while len(response) < expected_size:
                    packet = s.recv(4096)
                    response += packet
                data = pickle.loads(response)
        return data
