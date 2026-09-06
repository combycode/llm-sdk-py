"""A real MCP server over a WebSocket, for tests that should not mock a socket.

Runs on the stdlib alone: a `socket` accept loop plus the RFC 6455 handshake and
just enough framing to carry text both ways. That is a deliberate choice over
pulling in a server library -- the point is to exercise OUR client against a
socket that is genuinely a socket, and a fixture that dragged in a framework
would make the test about the framework.

Only what the protocol needs: single-frame text messages, masked client frames
unmasked, close handled. No fragmentation, no compression, no binary. A server
this small is readable in one sitting, which is the other half of the point.

Started by `serve()` on a background thread; the port it bound is returned so a
test never guesses one.
"""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import struct
import threading
from collections.abc import Callable
from typing import Any

#: The constant RFC 6455 makes every server concatenate before hashing.
_GUID = "258EAFA5-E914-47DA-95CA-5AB0DC85B11C"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "add",
        "description": "Add two numbers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
    {
        "name": "echo",
        "description": "Return the text it was given.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
]


def _accept_key(key: str) -> str:
    digest = hashlib.sha1(f"{key}{_GUID}".encode()).digest()
    return base64.b64encode(digest).decode()


def _handshake(conn: socket.socket) -> bool:
    """Answer the upgrade. False when the client sent something else."""
    request = b""
    while b"\r\n\r\n" not in request:
        chunk = conn.recv(4096)
        if not chunk:
            return False
        request += chunk
    key = ""
    for line in request.decode("latin-1").split("\r\n"):
        if line.lower().startswith("sec-websocket-key:"):
            key = line.split(":", 1)[1].strip()
    if not key:
        return False
    conn.sendall(
        b"HTTP/1.1 101 Switching Protocols\r\n"
        b"Upgrade: websocket\r\n"
        b"Connection: Upgrade\r\n"
        b"Sec-WebSocket-Accept: " + _accept_key(key).encode() + b"\r\n\r\n"
    )
    return True


def _recv_exactly(conn: socket.socket, count: int) -> bytes | None:
    out = b""
    while len(out) < count:
        chunk = conn.recv(count - len(out))
        if not chunk:
            return None
        out += chunk
    return out


def _read_frame(conn: socket.socket) -> str | None:
    """One text frame, or None when the peer closed."""
    header = _recv_exactly(conn, 2)
    if header is None:
        return None
    opcode = header[0] & 0x0F
    masked = bool(header[1] & 0x80)
    length = header[1] & 0x7F
    if length == 126:
        extended = _recv_exactly(conn, 2)
        if extended is None:
            return None
        length = struct.unpack(">H", extended)[0]
    elif length == 127:
        extended = _recv_exactly(conn, 8)
        if extended is None:
            return None
        length = struct.unpack(">Q", extended)[0]
    mask = _recv_exactly(conn, 4) if masked else b""
    if mask is None:
        return None
    payload = _recv_exactly(conn, length) if length else b""
    if payload is None:
        return None
    if masked:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    if opcode == 0x8:  # close
        return None
    if opcode == 0x9:  # ping -> pong, so a keep-alive does not kill the socket
        _send_frame(conn, payload, opcode=0xA)
        return _read_frame(conn)
    if opcode != 0x1:  # not text
        return _read_frame(conn)
    return payload.decode("utf-8", "replace")


def _send_frame(conn: socket.socket, payload: bytes, opcode: int = 0x1) -> None:
    header = bytes([0x80 | opcode])
    length = len(payload)
    if length < 126:
        header += bytes([length])
    elif length < 65536:
        header += bytes([126]) + struct.pack(">H", length)
    else:
        header += bytes([127]) + struct.pack(">Q", length)
    conn.sendall(header + payload)


def _send_text(conn: socket.socket, text: str) -> None:
    _send_frame(conn, text.encode("utf-8"))


def _result(method: str, params: dict[str, Any], mode: str) -> Any:
    if method == "initialize":
        return {
            "protocolVersion": params.get("protocolVersion", "2025-11-25"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "ws-fixture", "version": "0.0.1"},
        }
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name == "add":
            total = arguments.get("a", 0) + arguments.get("b", 0)
            return {"content": [{"type": "text", "text": str(total)}]}
        if name == "echo":
            return {"content": [{"type": "text", "text": str(arguments.get("text", ""))}]}
        return {"content": [{"type": "text", "text": f"no such tool: {name}"}], "isError": True}
    if method == "ping":
        return {}
    return {}


def _handle(conn: socket.socket, mode: str) -> None:
    replies: list[dict[str, Any]] = []
    while True:
        text = _read_frame(conn)
        if text is None:
            return
        try:
            message = json.loads(text)
        except ValueError:
            continue
        if not isinstance(message, dict):
            continue

        method = message.get("method")
        request_id = message.get("id")

        # A reply to one of OUR requests carries an id and no method.
        if method is None and request_id is not None:
            replies.append(
                {"id": request_id, "ok": "result" in message,
                 "error": (message.get("error") or {}).get("message")}
            )
            continue

        if method == "server/discover":
            # Every mode but `modern` refuses, so the fallback path is what a
            # connection normally exercises.
            if mode != "modern":
                _send_text(
                    conn,
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {"code": -32601, "message": "no discover here"},
                        }
                    ),
                )
                continue
            _send_text(
                conn,
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "capabilities": {"tools": {}},
                            "_meta": {
                                "io.modelcontextprotocol/serverInfo": {
                                    "name": "ws-fixture",
                                    "version": "0.0.1",
                                }
                            },
                        },
                    }
                ),
            )
            continue

        if method == "notifications/initialized":
            continue

        # Received and deliberately never answered, so a test can have a
        # request genuinely IN FLIGHT when it kills the socket. Every other
        # method answers, which would leave nothing to be woken.
        if method == "fixture/silence":
            continue

        # `fixture/push` makes the server send an unsolicited notification, and
        # `fixture/ask` makes it send US a request -- the direction only a duplex
        # connection has.
        if method == "fixture/push":
            _send_text(
                conn,
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/message",
                        "params": {"level": "info", "data": "pushed"},
                    }
                ),
            )
        elif method == "fixture/ask":
            _send_text(
                conn,
                json.dumps({"jsonrpc": "2.0", "id": 9001, "method": "roots/list", "params": {}}),
            )
        elif method == "fixture/replies":
            _send_text(
                conn,
                json.dumps({"jsonrpc": "2.0", "id": request_id, "result": {"replies": replies}}),
            )
            continue

        if request_id is None:
            continue
        _send_text(
            conn,
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": _result(str(method), message.get("params") or {}, mode),
                }
            ),
        )


def serve(
    mode: str = "normal", events: list[str] | None = None
) -> tuple[int, Callable[[], None]]:
    """Bind a port and serve on a thread. Returns the port and a stopper.

    `events` collects lifecycle notes -- `"open"` and `"close"` per
    connection. A client closing its socket is only observable from THIS
    side, so a test that wants to know the client really closed has to ask
    the server.
    """
    log = events if events is not None else []
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    port = listener.getsockname()[1]
    stopping = threading.Event()

    def loop() -> None:
        while not stopping.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            with conn:
                try:
                    if _handshake(conn):
                        log.append("open")
                        _handle(conn, mode)
                        log.append("close")
                except OSError:
                    log.append("close")

    thread = threading.Thread(target=loop, name="mcp-ws-fixture", daemon=True)
    thread.start()

    def stop() -> None:
        stopping.set()
        listener.close()
        thread.join(timeout=5)

    return port, stop


if __name__ == "__main__":
    import sys

    chosen = sys.argv[1] if len(sys.argv) > 1 else "normal"
    bound, _stop = serve(chosen)
    print(f"listening on {bound}", flush=True)
    threading.Event().wait()
