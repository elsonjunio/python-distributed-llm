import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained('microsoft/Phi-3.5-mini-instruct')
tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3.5-mini-instruct')

messages = [
    {'role': 'system', 'content': 'Você é um assistente de IA'},
    {'role': 'user', 'content': 'Qual o nome oficial do Brasil?'},
]

pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    'max_new_tokens': 500,
    'return_full_text': False,
    'temperature': 0.0,
    'do_sample': False,
}

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
