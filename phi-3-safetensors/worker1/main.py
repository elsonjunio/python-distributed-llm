### ============================
### WORKER 1: Embeddings + Camadas iniciais
### ============================

from fastapi import FastAPI
from pydantic import BaseModel
from safetensors.torch import safe_open
import torch
import uvicorn
import json
import os

app = FastAPI()


class InputData(BaseModel):
    input_ids: list


MODEL_PATH = '../../phi3_safetensors'
INDEX_FILE = os.path.join(MODEL_PATH, 'model.safetensors.index.json')

LAYERS = list(range(0, 16))


def is_relevant(key):
    if 'embed_tokens' in key:
        return True
    for i in LAYERS:
        if f'layers.{i}.' in key:
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
async def forward(data: InputData):
    input_ids = torch.tensor(data.input_ids)

    embed_weight = tensors[
        [k for k in tensors if 'embed_tokens.weight' in k][0]
    ]
    hidden = embed_weight[input_ids]

    for i in LAYERS:
        key = f'layers.{i}.self_attn.q_proj.weight'
        if key in tensors:
            W = tensors[key]
            hidden = torch.nn.functional.linear(hidden, W)

    return {'hidden_state': hidden.tolist()}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)
