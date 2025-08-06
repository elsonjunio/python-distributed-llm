"""Simple TCP server that receives partial-forward requests and runs local layers.

This module implements `PartialForwardServer`, a minimal blocking TCP server that
accepts framed, pickled requests, forwards the payload to a provided model
object via `run_local_layers_forward(**payload)` and returns the pickled result.
The wire protocol is:

  1. client -> server: 4-byte big-endian length prefix followed by pickle payload
  2. server -> client: 4-byte big-endian length prefix followed by pickle payload

Notes:
  - This implementation is intended for testing and debugging. For production use,
    consider adding authentication, timeouts, robust error handling, retries and
    a safer serialization format (e.g. msgpack/protobuf) instead of pickle.
  - The `partial_model` passed must expose a callable `run_local_layers_forward`
    that accepts the deserialized payload as keyword arguments and returns a
    picklable object.
"""

from __future__ import annotations

import pickle
import socket
import struct
import threading
from typing import Any, Callable, Tuple


class PartialForwardServer:
    """A small threaded TCP server to handle partial-forward requests.

    The server accepts connections and spawns a thread for each client. Each
    client request is expected to be a pickled object preceded by a 4-byte
    big-endian length prefix. The server deserializes the payload and calls
    `partial_model.run_local_layers_forward(**payload)` to produce a response,
    which is then pickled and sent back with a length prefix.

    Attributes:
        host (str): Host or IP address to bind to.
        port (int): TCP port to listen on.
        partial_model (Any): Object implementing `run_local_layers_forward(**kwargs)`.
    """

    def __init__(self, host: str, port: int, partial_model: Any) -> None:
        """Initialize the server.

        Args:
            host: Host or IP address to bind to.
            port: TCP port to listen on.
            partial_model: Model-like object that exposes `run_local_layers_forward`.
        """
        self.host = host
        self.port = port
        self.partial_model = partial_model

    def start(self) -> None:
        """Start listening for incoming connections and handle clients in threads.

        This method blocks the current thread and runs forever until the process
        is terminated.
        """
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen()

        print(f"Server running on {self.host}:{self.port}")

        try:
            while True:
                conn, _addr = server.accept()
                thread = threading.Thread(target=self.handle_client, args=(conn,))
                thread.daemon = True
                thread.start()
        finally:
            server.close()

    def handle_client(self, conn: socket.socket) -> None:
        """Handle a single client connection.

        Reads the framed request, deserializes it, calls the model's
        `run_local_layers_forward(**payload)`, and returns the framed response.

        Args:
            conn: Connected socket object.
        """
        try:
            # Read 4-byte length prefix
            size_data = self._recv_exact(conn, 4)
            if not size_data:
                raise ConnectionError("Client closed connection before sending size prefix.")
            expected_size = struct.unpack("!I", size_data)[0]

            # Read the exact payload size
            payload_bytes = self._recv_exact(conn, expected_size)
            if payload_bytes is None:
                raise ConnectionError("Client closed connection while sending payload.")

            # Deserialize request
            input_data = pickle.loads(payload_bytes)

            # Call model's handler
            if not hasattr(self.partial_model, "run_local_layers_forward"):
                raise AttributeError(
                    "partial_model must implement run_local_layers_forward(**kwargs)"
                )

            # Expect input_data to be a mapping of keyword args (dict-like)
            if isinstance(input_data, dict):
                output_data = self.partial_model.run_local_layers_forward(**input_data)
            else:
                # If payload is not a dict, attempt to pass it as a single arg
                output_data = self.partial_model.run_local_layers_forward(input_data)

            # Serialize and send response with length prefix
            serialized = pickle.dumps(output_data)
            conn.sendall(struct.pack("!I", len(serialized)))
            conn.sendall(serialized)
        except Exception as exc:  # Keep broad catch for debug/test server
            # For testing, print exception; in production use logging
            print(f"Error processing client: {exc}")
            try:
                # Attempt to return an error payload to client
                err_payload = pickle.dumps({"error": str(exc)})
                conn.sendall(struct.pack("!I", len(err_payload)))
                conn.sendall(err_payload)
            except Exception:
                # If even sending the error fails, silently continue to close
                pass
        finally:
            conn.close()

    @staticmethod
    def _recv_exact(conn: socket.socket, nbytes: int) -> bytes | None:
        """Receive exactly `nbytes` bytes from `conn` or return None if closed.

        Args:
            conn: Socket to read from.
            nbytes: Number of bytes to read.

        Returns:
            bytes object with the received data or None if connection closed early.
        """
        buf = bytearray()
        remaining = nbytes
        while remaining > 0:
            chunk = conn.recv(min(4096, remaining))
            if not chunk:
                return None
            buf.extend(chunk)
            remaining -= len(chunk)
        return bytes(buf)
