"""Client for sending partial-forward requests to a sequence of worker servers.

This module provides `PartialForwardClient`, a simple synchronous client that
sends a Python-serialized payload to each configured worker in sequence and
returns the final transformed payload received from the last worker.

The wire protocol used is:
  1. client -> server: 4-byte big-endian length prefix followed by pickle payload
  2. server -> client: 4-byte big-endian length prefix followed by pickle payload

Note:
  - This implementation uses blocking sockets and `pickle` for simplicity. For
    production use consider message framing, stronger serialization (e.g. msgpack
    or protobuf), authentication, timeouts and error handling.
"""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Any, Iterable, Tuple


class PartialForwardClient:
    """Sends partial-forward payloads through an ordered list of worker servers.

    The client sends the provided `input_data` to the first server, receives a
    response, forwards that response to the next server, and so on until all
    servers have been called. The final response is returned to the caller.

    Attributes:
        servers (Iterable[Tuple[str, int]]): Iterable of (host, port) pairs in
            the order they should be invoked.
    """

    def __init__(self, servers: Iterable[Tuple[str, int]]) -> None:
        """Initialize the client.

        Args:
            servers: An iterable of (host, port) tuples describing the worker
                endpoints in the order they should be called.
        """
        self.servers = list(servers)

    def send_partial_forward(self, input_data: Any) -> Any:
        """Send `input_data` through the configured servers sequentially.

        For each server:
          1. Connect to the server.
          2. Send a 4-byte big-endian length prefix followed by the pickled payload.
          3. Read a 4-byte big-endian length prefix from the server and then read
             the exact number of bytes indicated.
          4. Unpickle the server response and pass it as the input to the next server.

        Args:
            input_data: Arbitrary Python object serializable by `pickle`. This
                object will be transformed by each worker and the final object
                will be returned.

        Returns:
            The object returned by the last server after sequential processing.

        Raises:
            OSError: On socket-level errors (connection refused, reset, etc.).
            pickle.PickleError: If pickling/unpickling fails.
            struct.error: If framing bytes are malformed.
        """
        data = input_data
        for host, port in self.servers:
            with socket.create_connection((host, port)) as s:
                serialized = pickle.dumps(data)

                # Send length prefix (4 bytes, network order) followed by payload
                s.sendall(struct.pack("!I", len(serialized)))
                s.sendall(serialized)

                # Read 4-byte length prefix for the response
                size_data = s.recv(4)
                if len(size_data) < 4:
                    raise OSError("Connection closed while reading response length prefix")
                expected_size = struct.unpack("!I", size_data)[0]

                # Read exactly expected_size bytes
                response = b""
                while len(response) < expected_size:
                    packet = s.recv(min(4096, expected_size - len(response)))
                    if not packet:
                        raise OSError("Connection closed while reading response body")
                    response += packet

                data = pickle.loads(response)

        return data
