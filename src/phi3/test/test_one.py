import torch
from transformers import AutoTokenizer, Phi3ForCausalLM
from transformers.cache_utils import DynamicCache
from phi3_partial import Phi3PartialForCausalLM

from torch.distributions import Categorical

temperature = 0.0  # ajuste aqui conforme desejado

MODEL_PATH = 'microsoft/Phi-3.5-mini-instruct'

# Carrega modelo completo (master) só para acessar partes como embeddings e config
master = Phi3PartialForCausalLM.from_pretrained_partial(
    MODEL_PATH, handle_section_index=0, total_sections=1
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
prompt = (
    '<|system|>\nVocê é um assistente de IA, responda de forma sucinta e objetiva. Quando terminar, encerre sua resposta.\n'
    '<|user|>\nQual o nome oficial do Brasil?\n'
    '<|assistant|>\n'
)
inputs = tokenizer(prompt, return_tensors='pt')

generated = master.generate_using_partial_forward(inputs, temperature)

# Decodifica a sequência gerada
result = tokenizer.decode(
    generated['generated_only'], skip_special_tokens=True
)

print(result)
