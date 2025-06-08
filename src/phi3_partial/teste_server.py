from phi3_partial.raw_server.partial_forward_server import PartialForwardServer


class Phi3PartialModel:
    def __init__(self):
        pass

    def partial_forward(self, input_data):
        print(input_data)
        return input_data


if __name__ == '__main__':

    partial_model = Phi3PartialModel()
    server = PartialForwardServer('0.0.0.0', 9000, partial_model)
    server.start()
