from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import Phi3ForCausalLM, AutoTokenizer

app = FastAPI()

# Carrega modelo e tokenizer
model = Phi3ForCausalLM.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')


class ModelInput(BaseModel):
    intermediate_output: list  # Resultado vindo do worker1


@app.post('/forward')
async def forward(data: ModelInput):
    intermediate_tensor = torch.tensor(data.intermediate_output)

    # Processamento simplificado: aplica camada de normalização
    with torch.no_grad():
        normed = model.transformer.ln_f(
            intermediate_tensor
        )  # camada final de normalização
        # Gera logits a partir da saída normalizada
        logits = model.lm_head(normed)

    # Faz argmax para obter output_ids
    output_ids = torch.argmax(logits, dim=-1)

    return {'output_ids': output_ids.tolist()}
