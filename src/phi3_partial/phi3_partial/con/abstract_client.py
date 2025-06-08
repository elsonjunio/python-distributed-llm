from abc import ABC, abstractmethod


class AbstractClient(ABC):
    def __init__(self, servers):
        self.servers = servers  # List of (host, port) tuples in order

    @abstractmethod
    def send_partial_forward(self, input_data):
        pass
