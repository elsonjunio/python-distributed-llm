# worker2/main.py

from fastapi import FastAPI, Request
import torch
from transformers import GPT2Model
import uvicorn
import base64
import io

app = FastAPI()

# Carregar modelo completo
full_model = GPT2Model.from_pretrained('gpt2')


# Sem precisar importar GPT2Block diretamente
class BlockWrapper(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return self.block(x)[0]  # só pega hidden_states


# Carregar apenas camadas 6–11 do GPT2 com wrapper
full_model = GPT2Model.from_pretrained('gpt2')
wrapped_layers = [BlockWrapper(block) for block in full_model.h[6:]]
partial_layers = torch.nn.Sequential(*wrapped_layers)


# Selecionar apenas as camadas 6–11
# partial_layers = torch.nn.Sequential(*full_model.h[6:])

# Normalização final do GPT2
ln_f = full_model.ln_f

# Desabilita gradientes para inferência
for param in partial_layers.parameters():
    param.requires_grad = False
partial_layers.eval()
ln_f.eval()


def decode_tensor(data: dict) -> torch.Tensor:
    """Recebe dict com shape, dtype e base64 e converte em tensor PyTorch"""
    byte_data = base64.b64decode(data['data'])
    buffer = io.BytesIO(byte_data)
    tensor = torch.load(buffer)
    return tensor


def encode_tensor(tensor: torch.Tensor) -> dict:
    """Serializa tensor PyTorch para dict com base64"""
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return {
        'data': encoded,
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
    }


@app.post('/forward')
async def forward(request: Request):
    payload = await request.json()

    hidden_states = decode_tensor(payload['hidden_states'])

    with torch.no_grad():
        out = partial_layers(hidden_states)
        out = ln_f(out)  # Normalização final do modelo

    return encode_tensor(out)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8002)
