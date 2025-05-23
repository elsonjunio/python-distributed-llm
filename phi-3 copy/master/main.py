import sys
import asyncio
import httpx
import torch
from transformers import AutoTokenizer, Phi3ForCausalLM

# Configurações de endereços dos Workers
WORKER1_URL = 'http://localhost:8001/forward'
WORKER2_URL = 'http://localhost:8002/forward'

# Função para enviar requisições ao Worker
async def send_to_worker(url: str, model_input: torch.Tensor):
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=model_input.tolist())
        response.raise_for_status()
        return response.json()


# Função principal de processamento
async def process_prompt(prompt: str):
    # Usar AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'microsoft/Phi-3-mini-4k-instruct'
    )

    # Não carrega o modelo no master
    # Apenas tokeniza
    inputs = tokenizer(prompt, return_tensors='pt')
    model_input = inputs['input_ids']

    # Enviar para Worker 1
    intermediate_output = await send_to_worker(WORKER1_URL, model_input)

    # Enviar resultado intermediário para Worker 2
    final_output = await send_to_worker(
        WORKER2_URL, torch.tensor(intermediate_output['intermediate_output'])
    )

    # Decodificar a resposta final
    output_ids = torch.tensor(final_output['output_ids'])
    decoded_output = tokenizer.decode(
        output_ids.squeeze(), skip_special_tokens=True
    )

    return decoded_output


if __name__ == '__main__':
    while True:
        prompt = input("\nDigite um prompt (ou 'sair'): ")
        if prompt.strip().lower() == 'sair':
            break

        result = asyncio.run(process_prompt(prompt))
        print(f"Resposta do modelo: {result}")