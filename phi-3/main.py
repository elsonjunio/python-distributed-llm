from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM
import torch
import uvicorn

app = FastAPI()

# Modelo - mesmo modelo
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

class InputData(BaseModel):
    intermediate_output: list

@app.post("/forward")
async def forward(data: InputData):
    # Converte intermediate_output de volta para tensor
    intermediate_tensor = torch.tensor(data.intermediate_output)

    # Passa pela cabeça de linguagem
    with torch.no_grad():
        logits = model.lm_head(intermediate_tensor)
        # Obtém o token mais provável
        predicted_ids = torch.argmax(logits, dim=-1).tolist()

    return {"output_ids": predicted_ids}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8002)