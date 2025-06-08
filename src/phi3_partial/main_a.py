from transformers import Phi3Model, Phi3Config, Phi3ForCausalLM
from transformers import AutoTokenizer, Phi3ForCausalLM

MODEL_PATH = './Phi-3-mini-4k-instruct'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = Phi3ForCausalLM.from_pretrained(MODEL_PATH)

prompt = "Once upon a time"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
