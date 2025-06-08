from phi3_partial.raw_server.partial_forward_server import PartialForwardServer
from phi3_partial.phi3_partial_model import Phi3PartialModel, Phi3PartialForCausalLM
from transformers import Phi3Model, Phi3Config

MODEL_PATH = './Phi-3-mini-4k-instruct'


if __name__ == '__main__':

    model = Phi3PartialForCausalLM.from_pretrained_partial(MODEL_PATH, server_index=1, total_servers=2)
    server = PartialForwardServer('0.0.0.0', 9002, model)
    server.start()
