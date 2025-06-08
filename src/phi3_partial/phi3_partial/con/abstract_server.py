from abc import ABC, abstractmethod

class AbstractServer(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def partial_forward(self, input_data):
        pass