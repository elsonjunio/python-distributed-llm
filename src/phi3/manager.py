"""Manager process that coordinates distributed inference across Phi-3 workers.

This script loads a Phi3PartialForCausalLM as the master model (handling no
local layers) and delegates all forward passes to remote workers via
PartialForwardClient.

Usage:
    python manager.py
"""

from transformers import AutoTokenizer
from phi3_partial import Phi3PartialForCausalLM
from partial_forward_client import PartialForwardClient

# Model and generation settings
MODEL_PATH = "microsoft/Phi-3.5-mini-instruct"
TEMPERATURE = 0.0

# Ordered list of workers (host, port) that will process layers sequentially.
WORKERS = [
    ("127.0.0.1", 9001),
    ("127.0.0.1", 9002),
]

# Create a client to communicate with the workers
client = PartialForwardClient(WORKERS)


def distributed_layer_forward(**kwargs):
    """Send partial forward requests sequentially through configured workers.

    Args:
        **kwargs: Arbitrary keyword arguments representing the forward pass state
            (hidden_states, attention_mask, position_ids, etc.).

    Returns:
        Any: Output returned from the final worker in the chain.
    """
    return client.send_partial_forward(kwargs)


def main() -> None:
    """Run a distributed inference request using remote workers.

    Loads the Phi-3 model in master mode (no layers locally), sends the
    forward pass through remote workers, and prints the generated text.

    Returns:
        None
    """
    master = Phi3PartialForCausalLM.from_pretrained_partial(
        MODEL_PATH, handle_section_index=0, total_sections=0
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt = (
        "<|system|>\nVocê é um assistente de IA, responda de forma sucinta e objetiva. "
        "Quando terminar, encerre sua resposta.\n"
        "<|user|>\nQual o nome oficial do Brasil?\n"
        "<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    generated = master.generate_using_partial_forward(
        inputs, TEMPERATURE, distributed_layer_forward=distributed_layer_forward
    )

    result = tokenizer.decode(
        generated["generated_only"], skip_special_tokens=True
    )
    print(result)


if __name__ == "__main__":
    main()
