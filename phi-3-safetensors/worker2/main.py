### ============================
### WORKER 2: Camadas finais + lm_head
### ============================

from fastapi import FastAPI
from pydantic import BaseModel
from safetensors.torch import safe_open
import torch
import uvicorn
import json
import os

app = FastAPI()


class HiddenData(BaseModel):
    hidden_state: list


MODEL_PATH = '../../phi3_safetensors'
INDEX_FILE = os.path.join(MODEL_PATH, 'model.safetensors.index.json')

LAYERS = list(range(16, 32))


def is_relevant(key):
    for i in LAYERS:
        if f"layers.{i}." in key:
            return True
    if "lm_head" in key or "model.norm" in key:
        return True
    return False


with open(INDEX_FILE) as f:
    index_data = json.load(f)

shard_files = set()
for key, meta in index_data['weight_map'].items():
    if is_relevant(key):
        shard_files.add(meta)

tensors = {}
for shard in shard_files:
    with safe_open(os.path.join(MODEL_PATH, shard), framework='pt') as f:
        for key in f.keys():
            if is_relevant(key):
                tensors[key] = f.get_tensor(key)


@app.post('/forward')
async def forward(data: HiddenData):
    hidden = torch.tensor(data.hidden_state)

    for i in LAYERS:
        key = f'layers.{i}.self_attn.q_proj.weight'
        if key in tensors:
            W = tensors[key]
            hidden = torch.nn.functional.linear(hidden, W)

    norm_weight = tensors.get("model.norm.weight", None)
    if norm_weight is not None:
        hidden = torch.nn.functional.layer_norm(hidden, hidden.shape[-1:], weight=norm_weight, bias=None)

    lm_head_weight = tensors[[k for k in tensors if 'lm_head.weight' in k][0]]
    output = torch.nn.functional.linear(hidden, lm_head_weight)

    output_ids = output.argmax(dim=-1)

    return {'output_ids': output_ids.tolist()}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8002)
