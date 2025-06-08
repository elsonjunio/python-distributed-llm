from transformers import Phi3Model, Phi3Config, Phi3ForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import can_return_tuple, logging
from transformers.models.phi3.modeling_phi3 import (
    Phi3DecoderLayer,
    Phi3RMSNorm,
    Phi3RotaryEmbedding,
)
from functools import partial
import torch
from torch import nn
from typing import Callable, Optional, Tuple, Union
from phi3_partial.con.abstract_client import AbstractClient

import pickle
import struct

logger = logging.get_logger(__name__)


def get_layer_range(server_index, total_servers, total_layers):
    base = total_layers // total_servers
    remainder = total_layers % total_servers

    start = 0
    for i in range(server_index):
        start += base
        if i < remainder:
            start += 1  # distribuir sobra

    end = start + base

    if server_index < remainder:
        end += 1

    return start, end


# Guarda o original
Phi3Model_original_init = Phi3Model.__init__

# Define novo __init__
def phi3_model_partial_init(self, config, *args, **kwargs):
    Phi3Model.__bases__[0].__init__(self, config)

    print('Partial init called')


Phi3Model.__init__ = phi3_model_partial_init


class Phi3PartialModel(Phi3Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Phi3DecoderLayer`]

    Args:
        config: Phi3Config
        server_index: Índice do servidor (0 = master)
        total_servers: Total de servidores (0 = master)
    """

    def __init__(
        self,
        config: Phi3Config,
        server_index: int = 0,
        total_servers: int = 0,
        client: AbstractClient = None,
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

        self.server_index = server_index
        self.total_servers = total_servers

        self.master = False

        if self.server_index == 0 and self.total_servers == 0:
            # Master: não carrega camadas
            self.layers = nn.ModuleList([])
            self.master = True
        else:

            start_layer, end_layer = get_layer_range(
                server_index, total_servers, config.num_hidden_layers
            )

            self.assigned_layers = list(range(start_layer, end_layer))

            self.layers = nn.ModuleList(
                [
                    Phi3DecoderLayer(config, layer_idx)
                    for layer_idx in self.assigned_layers
                ]
            )

        self.norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Phi3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.client = client

        # Initialize weights and apply final processing
        self.post_init()

    def serialize(self, data):
        serialized = pickle.dumps(data)
        return pickle.loads(serialized)

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                'You must specify exactly one of input_ids or inputs_embeds'
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.'
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError(
                'The `past_key_values` should be either a `Cache` object or `None`.'
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length()
                if past_key_values is not None
                else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        #
        ##
        #

        param = {
            'all_hidden_states': all_hidden_states,
            'hidden_states': hidden_states,
            'causal_mask': causal_mask,
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'output_attentions': output_attentions,
            'use_cache': use_cache,
            'cache_position': cache_position,
            'position_embeddings': position_embeddings,
            'flash_attn_kwargs': flash_attn_kwargs,
            'all_self_attns': all_self_attns,
            'output_hidden_states': output_hidden_states,
        }

        new_param = self.client.send_partial_forward(param)

        all_hidden_states = new_param['all_hidden_states']
        hidden_states = new_param['hidden_states']
        causal_mask = new_param['causal_mask']
        position_ids = new_param['position_ids']
        past_key_values = new_param['past_key_values']
        output_attentions = new_param['output_attentions']
        use_cache = new_param['use_cache']
        cache_position = new_param['cache_position']
        position_embeddings = new_param['position_embeddings']
        flash_attn_kwargs = new_param['flash_attn_kwargs']
        all_self_attns = new_param['all_self_attns']
        output_hidden_states = new_param['output_hidden_states']
        #
        ##
        #

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def partial_forward(self, data):

        all_hidden_states = data['all_hidden_states']
        hidden_states = data['hidden_states']
        causal_mask = data['causal_mask']
        position_ids = data['position_ids']
        past_key_values = data['past_key_values']
        output_attentions = data['output_attentions']
        use_cache = data['use_cache']
        cache_position = data['cache_position']
        position_embeddings = data['position_embeddings']
        flash_attn_kwargs = data['flash_attn_kwargs']
        all_self_attns = data['all_self_attns']
        output_hidden_states = data['output_hidden_states']

        for layer_idx, decoder_layer in enumerate(
            self.layers[: self.config.num_hidden_layers]
        ):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        param = {
            'all_hidden_states': all_hidden_states,
            'hidden_states': hidden_states,
            'causal_mask': causal_mask,
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'output_attentions': output_attentions,
            'use_cache': use_cache,
            'cache_position': cache_position,
            'position_embeddings': position_embeddings,
            'flash_attn_kwargs': flash_attn_kwargs,
            'all_self_attns': all_self_attns,
            'output_hidden_states': output_hidden_states,
        }

        return param


# Guarda o original
original_init = Phi3ForCausalLM.__init__

# Define novo __init__
def partial_init(self, config, *args, **kwargs):
    Phi3ForCausalLM.__bases__[0].__init__(self, config)

    print('Partial init called')


Phi3ForCausalLM.__init__ = partial_init


class Phi3PartialForCausalLM(Phi3ForCausalLM):
    client: AbstractClient = (None,)
    server_index = 0
    total_servers = 0

    def __init__(self, config):
        super().__init__(config)
        self.model = Phi3PartialModel(
            config,
            client=self.client,
            server_index=self.server_index,
            total_servers=self.total_servers,
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_pretrained_partial(
        cls,
        pretrained_model_name_or_path,
        client=None,
        config=None,
        server_index: int = 0,
        total_servers: int = 0,
    ):
        cls.client = client
        cls.server_index = server_index
        cls.total_servers = total_servers

        if config is not None:
            return cls.from_pretrained(
                pretrained_model_name_or_path, config=config
            )
        else:
            return cls.from_pretrained(pretrained_model_name_or_path)
        
    def partial_forward(self, input_data):
        return self.model.partial_forward(input_data)
