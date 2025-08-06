import torch
from transformers import AutoTokenizer, Phi3ForCausalLM
from transformers.cache_utils import DynamicCache
from phi3_partial import Phi3PartialForCausalLM

from torch.distributions import Categorical

temperature = 0.0  # ajuste aqui conforme desejado

MODEL_PATH = 'microsoft/Phi-3.5-mini-instruct'

# Carrega modelo completo (master) só para acessar partes como embeddings e config
master = Phi3PartialForCausalLM.from_pretrained_partial(
    MODEL_PATH, handle_section_index=0, total_sections=0
)

# Modelos particionados
model1 = Phi3PartialForCausalLM.from_pretrained_partial(
    MODEL_PATH, handle_section_index=0, total_sections=2
)

model2 = Phi3PartialForCausalLM.from_pretrained_partial(
    MODEL_PATH, handle_section_index=1, total_sections=2
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# prompt = "Qual é o nome oficial do Brasil?"
# prompt = "<|system|>\nVocê é um assistente de IA, responda de forma sucinta e objetiva. Quando terminar, encerre sua resposta.\n<|user|>\nQual o nome oficial do Brasil?\n<|assistant|>\n"
prompt = (
    '<|system|>\nVocê é um assistente de IA, responda de forma sucinta e objetiva. Quando terminar, encerre sua resposta.\n'
    '<|user|>\nQual o nome oficial do Brasil?\n'
    '<|assistant|>\n'
)
inputs = tokenizer(prompt, return_tensors='pt')

# -- Dentro


input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Tokens gerados começam com o prompt
generated_ids = input_ids.clone()
start_ids = input_ids.clone()

# Cache dinâmico para armazenar past_key_values
past_key_values = DynamicCache()
output_attentions = master.config.output_attentions

# Identifica o token de parada usado pelo modelo
eos_token_id = tokenizer.eos_token_id

# Geração de 50 tokens
for i in range(4096):
    with torch.no_grad():

        if i == 0:
            # Primeira iteração: usa o prompt completo
            input_ids_step = generated_ids
            cache_position = torch.arange(generated_ids.shape[1]).unsqueeze(0)
            # position_ids = cache_position
        else:
            # Iterações seguintes: usa apenas o último token
            input_ids_step = generated_ids[:, -1:]
            cache_position = torch.tensor([[generated_ids.shape[1] - 1]])
            # position_ids = cache_position

        # Gera apenas o próximo token, usando o último token como entrada
        # input_ids_step = generated_ids[:, -1:]
        attention_mask = torch.ones_like(generated_ids)

        inputs_embeds = master.model.embed_tokens(input_ids_step)
        seq_len = generated_ids.shape[1]
        # cache_position = torch.tensor([seq_len - 1]).unsqueeze(0)
        position_ids = cache_position

        # Rotary embedding
        rotary_emb = master.model.rotary_emb(inputs_embeds, position_ids)

        # Máscara causal
        causal_mask = master.model._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        hidden_states = inputs_embeds

        # Passa pelas camadas do modelo 1
        for layer in model1.model.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=rotary_emb,
            )[0]

        # Passa pelas camadas do modelo 2
        for layer in model2.model.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=rotary_emb,
            )[0]

        # Normalização e cabeça de linguagem
        hidden_states = master.model.norm(hidden_states)
        logits = master.lm_head(hidden_states)

        # Seleção do próximo token
        next_token_logits = logits[:, -1, :]
        if temperature == 0:
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)
        else:
            # Aplica temperatura (quanto menor, mais determinístico)
            scaled_logits = next_token_logits / temperature
            # Softmax para obter probabilidades
            probs = torch.softmax(scaled_logits, dim=-1)
            # Amostra o próximo token de forma estocástica
            next_token = Categorical(probs).sample().unsqueeze(-1)

        # Interrompe se gerar o token de parada
        if next_token.item() == eos_token_id:
            break

        # Adiciona o novo token à sequência
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)


generated_only = generated_ids[0, start_ids.shape[1] :]

# Decodifica a sequência gerada
result = tokenizer.decode(generated_only, skip_special_tokens=True)

print(result)
