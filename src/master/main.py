# master/main.py

import torch
import requests
import base64
import io
import time
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Endpoints
WORKER1_URL = 'http://localhost:8001/forward'
WORKER2_URL = 'http://localhost:8002/forward'

# Tokenizer e lm_head
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
lm_head = model.lm_head
lm_head.eval()


def encode_tensor(tensor: torch.Tensor) -> dict:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return {
        'data': encoded,
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
    }


def decode_tensor(data: dict) -> torch.Tensor:
    byte_data = base64.b64decode(data['data'])
    buffer = io.BytesIO(byte_data)
    return torch.load(buffer)


def infer(prompt: str, max_tokens: int = 20):
    print(f'\nPrompt: {prompt}')
    total_start = time.time()

    # Tokenização
    t0 = time.time()
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    print(f'[tokenizer] {time.time() - t0:.3f} s')

    for step in range(max_tokens):
        print(f'\n🧠 Gerando token {step+1}/{max_tokens}')
        step_start = time.time()

        input_ids_list = input_ids[0].tolist()

        # Worker 1
        t1 = time.time()
        payload1 = {'input_ids': input_ids_list, 'hidden_states': None}
        response1 = requests.post(WORKER1_URL, json=payload1)
        h1 = decode_tensor(response1.json())
        print(f'[worker1] {time.time() - t1:.3f} s')

        # Worker 2
        t2 = time.time()
        payload2 = {'hidden_states': encode_tensor(h1)}
        response2 = requests.post(WORKER2_URL, json=payload2)
        h2 = decode_tensor(response2.json())
        print(f'[worker2] {time.time() - t2:.3f} s')

        # lm_head
        t3 = time.time()
        with torch.no_grad():
            logits = lm_head(h2)
        print(f'[lm_head] {time.time() - t3:.3f} s')

        # Escolher próximo token
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

        print(f'[token total] {time.time() - step_start:.3f} s')

        # Parar se for token especial
        if next_token_id.item() in tokenizer.all_special_ids:
            print('[parado] token especial detectado.')
            break

    result = tokenizer.decode(input_ids[0])
    print(f'\n📝 Resposta: {result}')
    print(f'[tempo total] {time.time() - total_start:.3f} s')


if __name__ == '__main__':
    while True:
        prompt = input("\nDigite um prompt (ou 'sair'): ")
        if prompt.strip().lower() == 'sair':
            break
        infer(prompt)
