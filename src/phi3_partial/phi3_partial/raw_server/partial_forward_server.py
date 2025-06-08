import socket
import pickle
import threading
import struct
from phi3_partial.con.abstract_server import AbstractServer


class PartialForwardServer(AbstractServer):
    def __init__(self, host, port, partial_model):
        self.host = host
        self.port = port
        self.partial_model = partial_model

    def start(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.host, self.port))
        server.listen()

        print(f'Server running on {self.host}:{self.port}')

        while True:
            conn, addr = server.accept()
            threading.Thread(target=self.handle_client, args=(conn,)).start()

    def handle_client(self, conn):
        try:
            # Lê os 4 bytes iniciais para saber o tamanho
            size_data = conn.recv(4)
            expected_size = struct.unpack('!I', size_data)[0]

            data = b''
            while len(data) < expected_size:
                packet = conn.recv(4096)
                data += packet
            input_data = pickle.loads(data)

            output_data = self.partial_forward(input_data)
            serialized = pickle.dumps(output_data)

            # Envia também o tamanho da resposta
            conn.sendall(struct.pack('!I', len(serialized)))
            conn.sendall(serialized)
        finally:
            conn.close()

    def partial_forward(self, input_data):
        return self.partial_model.partial_forward(input_data)
