from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import Phi3ForCausalLM, AutoTokenizer
import uvicorn

app = FastAPI()

# Carrega modelo e tokenizer
model = Phi3ForCausalLM.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')


class ModelInput(BaseModel):
    input: list  # input_ids


@app.post('/forward')
async def forward(data: ModelInput):
    input_ids = torch.tensor(data.input).unsqueeze(
        0
    )  # Adiciona batch dimension

    # Processamento simplificado: gerar embeddings
    with torch.no_grad():
        outputs = model.transformer.wte(
            input_ids
        )  # Apenas embeddings iniciais

    # Converte para lista para serialização
    return {'intermediate_output': outputs.tolist()}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)