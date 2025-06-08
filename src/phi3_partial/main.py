from transformers import Phi3Model, Phi3Config, Phi3ForCausalLM
from transformers import AutoTokenizer, Phi3ForCausalLM

from phi3_partial.phi3_partial_model import Phi3PartialForCausalLM
from phi3_partial.raw_server.partial_forward_client import PartialForwardClient

MODEL_PATH = './Phi-3-mini-4k-instruct'

#configuration = Phi3Config.from_pretrained(MODEL_PATH)
#
#configuration._attn_implementation_autoset = True
#configuration.torch_dtype = "float32"
#configuration.use_cache = False


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

client = PartialForwardClient([('127.0.0.1', 9001), ('127.0.0.1', 9002)])

#model = Phi3PartialForCausalLM.from_pretrained(MODEL_PATH)

model = Phi3PartialForCausalLM.from_pretrained_partial(MODEL_PATH, client=client)

prompt = "Once upon a time"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
