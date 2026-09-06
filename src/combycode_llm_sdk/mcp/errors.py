"""What a JSON-RPC failure is, and what it is not.

The distinction this file exists to hold: a transport or protocol failure is an
EXCEPTION, and a tool that ran and failed is a normal result carrying
`isError: true`. Collapsing the two would make "the server is unreachable" and
"the model asked for a file that does not exist" the same event, and only one of
those is worth retrying or reporting to an operator.

Transposed from `unified-library-ts/src/plugins/mcp/jsonrpc.ts`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final


class McpErrorCode:
    """JSON-RPC and MCP error codes.

    A class of constants rather than an `IntEnum`: a server may send a code
    nobody here has heard of, and an enum would make that unrepresentable.
    """

    CONNECTION_CLOSED: Final = -32000
    REQUEST_TIMEOUT: Final = -32001
    PARSE_ERROR: Final = -32700
    INVALID_REQUEST: Final = -32600
    METHOD_NOT_FOUND: Final = -32601
    INVALID_PARAMS: Final = -32602
    INTERNAL_ERROR: Final = -32603

    # -- 2026-07-28 revision (verified against mcp-py 2.0.0 `mcp_types/jsonrpc.py`)
    #: A routing header disagrees with the request body.
    HEADER_MISMATCH: Final = -32020
    #: The server requires a client capability this client did not declare.
    MISSING_REQUIRED_CLIENT_CAPABILITY: Final = -32021
    #: The server does not speak the requested revision; `data["supported"]` lists
    #: the ones it does. The one error carrying era information, which is why
    #: negotiation reads it specifically.
    UNSUPPORTED_PROTOCOL_VERSION: Final = -32022


class McpError(RuntimeError):
    """A transport- or protocol-level failure.

    NOT raised for a tool that ran and reported a problem: that arrives as an
    ordinary result with `isError` set, and is shown to the model as text.
    """

    def __init__(self, message: str, *, code: int = McpErrorCode.INTERNAL_ERROR, data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data

    @staticmethod
    def of(error: Mapping[str, Any]) -> McpError:
        """Build one from a JSON-RPC `error` object."""
        return McpError(
            str(error.get("message") or "MCP error"),
            code=int(error.get("code") or McpErrorCode.INTERNAL_ERROR),
            data=error.get("data"),
        )

    def as_wire(self) -> dict[str, Any]:
        """Back into a JSON-RPC `error` object, for a reply to the server."""
        wire: dict[str, Any] = {"code": self.code, "message": str(self)}
        if self.data is not None:
            wire["data"] = self.data
        return wire

    def __repr__(self) -> str:
        return f"McpError({str(self)!r}, code={self.code})"


__all__ = ["McpError", "McpErrorCode"]
