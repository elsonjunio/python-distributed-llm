import pickle


class CommunicationProtocol:
    @staticmethod
    def serialize(data):
        return pickle.dumps(data)

    @staticmethod
    def deserialize(data):
        return pickle.loads(data)
