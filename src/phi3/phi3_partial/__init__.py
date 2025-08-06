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
from transformers.processing_utils import Unpack

from torch.distributions import Categorical
from typing import List

logger = logging.get_logger(__name__)


class DummyDecoderLayer(nn.Module):
    """
    A placeholder decoder layer that returns input as output without modification.

    This class is typically used to occupy unused layers in a distributed model
    or for testing purposes when no computation is required.

    Methods:
        forward(hidden_states, ...): Returns the input hidden states unchanged.
    """

    def __init__(self, config: Phi3Config, layer_idx: int):
        """
        Initializes a dummy decoder layer.

        Args:
            config (Phi3Config): Configuration object (unused in dummy layer).
            layer_idx (int): Index of the layer (unused, but kept for compatibility).
        """
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> List[torch.Tensor]:
        """
        Returns the hidden states unchanged.

        This method is a no-op and is used to bypass actual computation in
        transformer layers.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
            attention_mask (Optional[torch.Tensor]): Not used.
            position_ids (Optional[torch.LongTensor]): Not used.
            past_key_value (Optional[Cache]): Not used.
            output_attentions (Optional[bool]): Not used.
            use_cache (Optional[bool]): Not used.
            cache_position (Optional[torch.LongTensor]): Not used.
            position_embeddings (Optional[Tuple[torch.Tensor, torch.Tensor]]): Not used.
            **kwargs (FlashAttentionKwargs): Additional keyword arguments (ignored).

        Returns:
            List[torch.Tensor]: A single-element list containing the input `hidden_states`.
        """
        return [hidden_states]


def create_layer(
    config: Phi3Config, layer_idx: int, assigned_layers: list[int]
) -> Union[Phi3DecoderLayer, DummyDecoderLayer]:
    if layer_idx in assigned_layers:
        return Phi3DecoderLayer(config, layer_idx)
    else:
        return DummyDecoderLayer(config, layer_idx)


class Phi3PartialModel(Phi3Model):
    """
    A partially distributed transformer decoder model based on Phi3Model.

    This class splits the transformer layers across multiple nodes or processes.
    Each instance handles only a subset of layers, determined by its section index.
    The master node (section index 0 and total sections 0) holds no layers and is
    responsible for orchestration or coordination.

    Attributes:
        padding_idx (int): Token ID used for padding.
        vocab_size (int): Size of the model's vocabulary.
        embed_tokens (nn.Embedding): Embedding layer for token input.
        handle_section_index (int): Index of the section this model handles. 0 indicates master.
        total_sections (int): Total number of sections across which layers are distributed.
        master (bool): Whether this instance is the master (no layers).
        start_layer (int): Index of the first layer handled by this instance.
        end_layer (int): Index of the last layer (exclusive) handled by this instance.
        assigned_layers (List[int]): List of layer indices handled.
        layers (nn.ModuleList): List of transformer decoder layers (partially or fully populated).
        norm (nn.Module): Final normalization layer.
        rotary_emb (Phi3RotaryEmbedding): Rotary positional embeddings.
        gradient_checkpointing (bool): Whether gradient checkpointing is enabled.
    """

    def __init__(
        self,
        config: Phi3Config,
        handle_section_index: int = 0,
        total_sections: int = 0,
    ):
        """
        Initializes a partially distributed Phi3 transformer model.

        Args:
            config (Phi3Config): Configuration object for the model.
            handle_section_index (int, optional): Index of the section this instance handles.
                Use 0 for the master node. Defaults to 0.
            total_sections (int, optional): Total number of sections (including master).
                Use 0 to indicate master mode. Defaults to 0.
        """

        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

        self.handle_section_index = handle_section_index
        self.total_sections = total_sections

        self.master = False

        if self.handle_section_index == 0 and self.total_sections == 0:
            # Master: não carrega camadas
            self.layers = nn.ModuleList([])
            self.master = True
        else:

            start_layer, end_layer = self.get_layer_range(
                handle_section_index, total_sections, config.num_hidden_layers
            )

            self.start_layer = start_layer
            self.end_layer = end_layer

            self.assigned_layers = list(range(start_layer, end_layer))

            self.layers = nn.ModuleList(
                [
                    create_layer(config, layer_idx, self.assigned_layers)
                    for layer_idx in range(config.num_hidden_layers)
                ]
            )

        self.norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Phi3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def post_init(self):
        super().post_init()

    @staticmethod
    def get_layer_range(handle_section_index, total_sections, total_layers):
        """
        Calculates the range of transformer layers assigned to a given section index.

        The distribution ensures that extra layers (from remainder) are spread among
        the first sections.

        Args:
            handle_section_index (int): Index of the current section.
            total_sections (int): Total number of sections.
            total_layers (int): Total number of transformer layers.

        Returns:
            Tuple[int, int]: A tuple containing the start (inclusive) and end (exclusive)
            layer indices assigned to this section.
        """
        base = total_layers // total_sections
        remainder = total_layers % total_sections

        start = 0
        for i in range(handle_section_index):
            start += base
            if i < remainder:
                start += 1  # distribuir sobra

        end = start + base

        if handle_section_index < remainder:
            end += 1

        return start, end


class Phi3PartialForCausalLM(Phi3ForCausalLM):
    """
    A partially distributed version of Phi3ForCausalLM, allowing splitting the model
    across multiple workers. This class manages model partitioning and generation
    across sections of transformer layers.

    Attributes:
        handle_section_index (int): The index of the section this instance handles.
        total_sections (int): The total number of sections the model is split into.
        vocab_size (int): Size of the vocabulary used for language modeling.
        lm_head (nn.Linear): Linear layer projecting hidden states to vocabulary logits.
    """

    handle_section_index = 0
    total_sections = 0

    def __init__(self, config):
        """
        Initializes the Phi3PartialForCausalLM model.

        Args:
            config (PretrainedConfig): The configuration object containing model parameters.
        """
        super().__init__(config)
        self.model = Phi3PartialModel(
            config,
            handle_section_index=self.handle_section_index,
            total_sections=self.total_sections,
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        self.post_init()

    @classmethod
    def from_pretrained_partial(
        cls,
        pretrained_model_name_or_path,
        config=None,
        handle_section_index: int = 0,
        total_sections: int = 0,
    ):
        """
        Instantiates the model from a pre-trained checkpoint, while configuring
        the section of the model to handle in distributed mode.

        Args:
            pretrained_model_name_or_path (str): Path or identifier of the pre-trained model.
            config (Optional[PretrainedConfig]): Optional configuration to use instead of loading from file.
            handle_section_index (int): Index of the section this instance will manage.
            total_sections (int): Total number of sections into which the model is split.

        Returns:
            Phi3PartialForCausalLM: A model instance initialized for distributed inference.
        """
        cls.handle_section_index = handle_section_index
        cls.total_sections = total_sections

        if config is not None:
            return cls.from_pretrained(
                pretrained_model_name_or_path, config=config
            )
        else:
            return cls.from_pretrained(pretrained_model_name_or_path)

    def run_local_layers_forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values,
        cache_position,
        rotary_emb,
    ):
        """
        Runs the forward pass through locally assigned layers.

        Args:
            hidden_states (torch.Tensor): Current hidden states to pass through layers.
            attention_mask (torch.Tensor): Causal attention mask.
            position_ids (torch.LongTensor): Position IDs.
            past_key_values (Cache): Cache for KV.
            cache_position (torch.LongTensor): Position in cache.
            rotary_emb (Tuple[torch.Tensor, torch.Tensor]): Rotary embeddings.

        Returns:
            torch.Tensor: Output hidden states after local layer forward.
        """
        for layer in self.model.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=rotary_emb,
            )[0]

        response = {
            'hidden_states': hidden_states,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'cache_position': cache_position,
            'rotary_emb': rotary_emb,
        }

        return response

    def generate_using_partial_forward(
        self,
        inputs,
        temperature,
        max_tokens=4096,
        eos_token_id=32000,
        distributed_layer_forward=None,
    ):
        """
        Generates text using a partially executed forward pass over distributed sections of the model.

        Args:
            inputs (Dict[str, torch.Tensor]): A dictionary containing input tensors. Must include
                'input_ids' and 'attention_mask'.
            temperature (float): Sampling temperature. If 0, uses greedy decoding.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 4096.
            eos_token_id (int, optional): Token ID that indicates end of sequence. Defaults to 32000.
            distributed_layer_forward (Callable, optional): Custom function for distributed forward
                execution. If not provided, executes locally on this instance's section.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with:
                - 'all': Full sequence of generated token IDs.
                - 'generated_only': Portion of generated IDs excluding the input prefix.
        """
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        generated_ids = input_ids.clone()
        start_ids = input_ids.clone()

        past_key_values = DynamicCache()
        output_attentions = self.model.config.output_attentions

        eos_token_id = eos_token_id

        for i in range(max_tokens):
            with torch.no_grad():

                if i == 0:
                    input_ids_step = generated_ids
                    cache_position = torch.arange(
                        generated_ids.shape[1]
                    ).unsqueeze(0)
                else:
                    input_ids_step = generated_ids[:, -1:]
                    cache_position = torch.tensor(
                        [[generated_ids.shape[1] - 1]]
                    )

                attention_mask = torch.ones_like(generated_ids)

                inputs_embeds = self.model.embed_tokens(input_ids_step)
                position_ids = cache_position
                rotary_emb = self.model.rotary_emb(inputs_embeds, position_ids)

                causal_mask = self.model._update_causal_mask(
                    attention_mask,
                    inputs_embeds,
                    cache_position,
                    past_key_values,
                    output_attentions,
                )

                hidden_states = inputs_embeds

                if distributed_layer_forward:
                    response = distributed_layer_forward(
                        hidden_states=hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        rotary_emb=rotary_emb,
                    )
                else:
                    response = self.run_local_layers_forward(
                        hidden_states=hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        rotary_emb=rotary_emb,
                    )
                # --
                hidden_states = response['hidden_states']
                causal_mask = response['attention_mask']
                position_ids = response['position_ids']
                past_key_values = response['past_key_values']
                cache_position = response['cache_position']
                rotary_emb = response['rotary_emb']

                # --

                hidden_states = self.model.norm(hidden_states)
                logits = self.lm_head(hidden_states)

                next_token_logits = logits[:, -1, :]
                if temperature == 0:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)
                else:
                    scaled_logits = next_token_logits / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    next_token = Categorical(probs).sample().unsqueeze(-1)

                if next_token.item() == eos_token_id:
                    break

                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                generated_only = generated_ids[0, start_ids.shape[1] :]

        return {'all': generated_ids, 'generated_only': generated_only}
