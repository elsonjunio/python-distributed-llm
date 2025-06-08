# master/main.py
import requests
from transformers import AutoTokenizer

WORKER1_URL = "http://127.0.0.1:8001/forward"
WORKER2_URL = "http://127.0.0.1:8002/forward"

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

def main():
    while True:
        prompt = input("Digite o prompt (ou 'sair' para encerrar): ")
        if prompt.lower() == 'sair':
            break

        input_ids = tokenizer.encode(prompt)

        MAX_TOKENS = 50
        for _ in range(MAX_TOKENS):
            # Envia para Worker1
            response1 = requests.post(WORKER1_URL, json={"input": input_ids})
            if response1.status_code != 200:
                print(f"Erro Worker1: {response1.status_code} {response1.text}")
                break

            intermediate_output = response1.json().get("intermediate_output")

            # Envia para Worker2
            response2 = requests.post(WORKER2_URL, json={"intermediate_output": intermediate_output})
            if response2.status_code != 200:
                print(f"Erro Worker2: {response2.status_code} {response2.text}")
                break

            next_token_id = response2.json().get("next_token_id")
            if next_token_id is None:
                print("Resposta inválida do Worker2")
                break

            input_ids.append(next_token_id)

            # Parar se for token de parada
            if next_token_id == tokenizer.eos_token_id:
                break

        output_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"Resposta do modelo: {output_text}")

if __name__ == "__main__":
    main()
