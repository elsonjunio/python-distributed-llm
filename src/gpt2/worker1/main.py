from fastapi import FastAPI, Request
import torch
from transformers import GPT2Model
#from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import uvicorn
import base64
import io

app = FastAPI()

# Wrapper para extrair apenas hidden_states
#class GPT2BlockWrapper(torch.nn.Module):
#    def __init__(self, block: GPT2Block):
#        super().__init__()
#        self.block = block
#
#    def forward(self, x):
#        return self.block(x)[0]  # Apenas hidden_states

# Sem precisar importar GPT2Block diretamente
class BlockWrapper(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return self.block(x)[0]  # só pega hidden_states


# Carregar apenas camadas 0–5 do GPT2 com wrapper
full_model = GPT2Model.from_pretrained('gpt2')
wrapped_layers = [BlockWrapper(block) for block in full_model.h[:6]]
partial_layers = torch.nn.Sequential(*wrapped_layers)

embedding = full_model.wte
position = full_model.wpe
ln_f = full_model.ln_f  # caso precise no futuro

# Desabilita gradientes e seta eval
for param in partial_layers.parameters():
    param.requires_grad = False
partial_layers.eval()


def decode_tensor(data: dict) -> torch.Tensor:
    byte_data = base64.b64decode(data['data'])
    buffer = io.BytesIO(byte_data)
    tensor = torch.load(buffer)
    return tensor


def encode_tensor(tensor: torch.Tensor) -> dict:
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

    hidden_states = None
    if payload['hidden_states'] is not None:
        hidden_states = decode_tensor(payload['hidden_states'])

    input_ids = torch.tensor(payload['input_ids']).unsqueeze(0)  # [1, seq_len]

    # Forward pass nas camadas 0–5
    with torch.no_grad():
        if hidden_states is None:
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len).unsqueeze(0)
            inputs_embeds = embedding(input_ids) + position(position_ids)
        else:
            inputs_embeds = hidden_states

        out = partial_layers(inputs_embeds)

    return encode_tensor(out)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)
