### ============================
### MASTER
### ============================

from tokenizers import Tokenizer
import requests

tokenizer = Tokenizer.from_file("./phi3_safetensors/tokenizer.json")

WORKER1_URL = "http://127.0.0.1:8001/forward"
WORKER2_URL = "http://127.0.0.1:8002/forward"

def main():
    while True:
        prompt = input("Digite o prompt: ")
        if prompt == "sair":
            break

        input_ids = tokenizer.encode(prompt).ids

        # Envia para Worker1
        response = requests.post(WORKER1_URL, json={"input_ids": input_ids})
        hidden = response.json()["hidden_state"]

        # Envia para Worker2
        response = requests.post(WORKER2_URL, json={"hidden_state": hidden})
        output_ids = response.json()["output_ids"]

        output_text = tokenizer.decode(output_ids)
        print(f"Resposta: {output_text}")

if __name__ == "__main__":
    main()
