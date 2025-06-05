from tokenizers import Tokenizer
import requests

tokenizer = Tokenizer.from_file("phi3_safetensors/tokenizer.json")

WORKER1_URL = "http://127.0.0.1:8001/forward"
WORKER2_URL = "http://127.0.0.1:8002/forward"

def main():
    while True:
        prompt = input("Usuário: ")
        if prompt == "sair":
            break

        input_ids = tokenizer.encode(prompt).ids
        generated_ids = input_ids.copy()

        for _ in range(50):  # Limite máximo de tokens gerados
            # Envia para Worker1
            response = requests.post(WORKER1_URL, json={"input_ids": generated_ids})
            hidden = response.json()["hidden_state"]

            # Envia para Worker2
            response = requests.post(WORKER2_URL, json={"hidden_state": hidden})
            output_ids = response.json()["output_ids"]

            next_token_id = output_ids[-1]
            generated_ids.append(next_token_id)

            next_token = tokenizer.decode([next_token_id])
            print(next_token, end='', flush=True)

            # Condição de parada: token de parada ou outro critério
            if next_token_id == tokenizer.token_to_id("<|endoftext|>") or len(generated_ids) > 512:
                break

        print("\n")  # Nova linha após resposta completa

if __name__ == "__main__":
    main()
