from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM
import torch
import uvicorn

app = FastAPI()

# Modelo - ajuste conforme o modelo baixado
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

class InputData(BaseModel):
    input: list

@app.post("/forward")
async def forward(data: InputData):
    input_ids = torch.tensor([data.input])

    # Processa parcialmente - por exemplo, só passa pela embedding
    with torch.no_grad():
        #outputs = model.transformer(input_ids)
        outputs = model.model(input_ids)
        # Aqui usamos apenas a última camada como exemplo
        intermediate_output = outputs.last_hidden_state.tolist()

    return {"intermediate_output": intermediate_output}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)