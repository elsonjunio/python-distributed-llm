from phi3_partial import Phi3PartialModel, Phi3PartialForCausalLM
from transformers import Phi3Model, Phi3Config

from transformers import Phi3Model, Phi3Config, Phi3ForCausalLM
from transformers import AutoTokenizer, Phi3ForCausalLM

import torch


# MODEL_PATH = './Phi-3-mini-4k-instruct'
MODEL_PATH = 'microsoft/Phi-3.5-mini-instruct'


# Carregue seus dois modelos parciais
model1 = Phi3PartialForCausalLM.from_pretrained_partial(
    MODEL_PATH, handle_section_index=0, total_sections=2
)

model2 = Phi3PartialForCausalLM.from_pretrained_partial(
    MODEL_PATH, handle_section_index=1, total_sections=2
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
prompt = 'Qual o nome oficial do Brasil?'
inputs = tokenizer(prompt, return_tensors='pt')

# Inicializa variáveis
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

generated_ids = input_ids.clone()

# Gera 10 tokens
for _ in range(50):
    with torch.no_grad():

        inputs_embeds = model1.model.embed_tokens(generated_ids)
        cache_position = torch.arange(inputs_embeds.shape[1])
        position_ids = cache_position.unsqueeze(0)
        rotary_emb = model1.model.rotary_emb(inputs_embeds, position_ids)

        causal_mask = model1.model._update_causal_mask(
            attention_mask=attention_mask,
            input_tensor=inputs_embeds,
            cache_position=cache_position,
            past_key_values=None,
            output_attentions=False,
        )

        hidden_states = inputs_embeds

        for layer in model1.model.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=rotary_emb,
            )[0]

        # hidden_states = model1.model.norm(hidden_states)
        # logits = model1.lm_head(hidden_states)
        # probs = torch.softmax(logits[:, -1], dim=-1)
        # next_token = torch.argmax(probs, dim=-1).unsqueeze(0)

        # Etapa 3: camadas do model2
        for layer in model2.model.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=rotary_emb,
            )[0]

        # Etapa 4: finalização
        hidden_states = model2.model.norm(hidden_states)
        logits = model2.lm_head(hidden_states)
        probs = torch.softmax(logits[:, -1], dim=-1)
        next_token = torch.argmax(probs, dim=-1).unsqueeze(0)

        # Adiciona token à sequência
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

# Decodifica a sequência final
result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(result)
