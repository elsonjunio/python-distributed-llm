"""Worker process that hosts a partial Phi-3 model and serves partial forwards.

This module instantiates a Phi3PartialForCausalLM configured to handle a section
of the model (specified by `handle_section_index` and `total_sections`) and
starts a PartialForwardServer that listens for remote partial-forward requests.

Usage:
    python worker1.py
"""

from phi3_partial import Phi3PartialForCausalLM
from partial_forward_server import PartialForwardServer

# Network settings for the worker server.
HOST = "0.0.0.0"
PORT = 9001

# Local path / HF identifier of the Phi-3 model. Adjust as needed.
MODEL_PATH = "microsoft/Phi-3.5-mini-instruct"


def main() -> None:
    """Instantiate the partial model and start the partial-forward server.

    The model is loaded using `from_pretrained_partial` with the section index
    and total sections appropriate for this worker. The server then listens on
    (HOST, PORT) and handles incoming partial-forward requests.

    Returns:
        None
    """
    worker_1 = Phi3PartialForCausalLM.from_pretrained_partial(
        MODEL_PATH, handle_section_index=0, total_sections=2
    )

    server = PartialForwardServer(HOST, PORT, worker_1)
    server.start()


if __name__ == "__main__":
    main()
