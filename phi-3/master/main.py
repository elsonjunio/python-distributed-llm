import requests
from transformers import AutoTokenizer

# Configura URLs dos Workers
WORKER1_URL = "http://127.0.0.1:8001/forward"
WORKER2_URL = "http://127.0.0.1:8002/forward"

# Carrega tokenizer - ajuste para o modelo Phi3 correspondente
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

def main():
    while True:
        # Recebe o prompt
        prompt = input("Digite o prompt (ou 'sair' para encerrar): ")
        if prompt.lower() == 'sair':
            break

        # Tokeniza
        input_ids = tokenizer.encode(prompt, return_tensors='pt').tolist()[0]

        # Envia para Worker1
        response1 = requests.post(WORKER1_URL, json={"input": input_ids})
        if response1.status_code != 200:
            print(f"Erro no Worker1: {response1.status_code} {response1.text}")
            continue

        intermediate_output = response1.json().get("intermediate_output")
        if intermediate_output is None:
            print("Resposta inválida do Worker1")
            continue

        # Envia para Worker2
        response2 = requests.post(WORKER2_URL, json={"intermediate_output": intermediate_output})
        if response2.status_code != 200:
            print(f"Erro no Worker2: {response2.status_code} {response2.text}")
            continue

        output_ids = response2.json().get("output_ids")
        if output_ids is None:
            print("Resposta inválida do Worker2")
            continue

        # Decodifica
        #output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        flat_output_ids = output_ids[0] if isinstance(output_ids[0], list) else output_ids
        output_text = tokenizer.decode(flat_output_ids, skip_special_tokens=True)

        # Mostra resultado
        print(f"Resposta do modelo: {output_text}")

if __name__ == "__main__":
    main()
