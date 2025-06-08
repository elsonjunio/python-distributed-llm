# worker2/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM
import torch
import uvicorn

app = FastAPI()

model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

class InputData(BaseModel):
    intermediate_output: list

@app.post("/forward")
async def forward(data: InputData):
    intermediate_tensor = torch.tensor(data.intermediate_output)

    with torch.no_grad():
        logits = model.lm_head(intermediate_tensor)
        predicted_id = torch.argmax(logits[:, -1, :], dim=-1).item()

    return {"next_token_id": predicted_id}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8002)

