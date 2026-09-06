"""The two MCP rules a spec cannot express as data.

Every other named spec rule lives in `llm/wire_transforms.py`. These two cannot:
that module belongs to the LLM layer, MCP is a plugin, and a shared registry
reaching down into a plugin is the edge this port does not draw. So the
transport composes its own registry from the shared one instead.

Both are genuinely code rather than data:

`mcpModern` -- the era is set only AFTER discovery succeeds, so a request has to
be judged by the version it DECLARES as well. Keying on era alone leaves the
`server/discover` probe itself half-modern, which a modern server rejects.

`mcpNameHeader` -- the subject lives under a different parameter per method
(`name` for tools/call and prompts/get, `uri` for resources/read), so this is a
lookup followed by a read at the key that lookup returned. A template can
express a fixed path, not a computed one.

Transposed from `unified-library-ts/src/plugins/mcp/wire-rules.ts`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from ..wire.interpreter import Ctx, Registry
from ..wire.registry import WIRE_SPECS
from .protocol import is_modern_mcp_version

#: Methods whose subject goes in the `Mcp-Name` routing header, and the
#: parameter it comes from. Lets a gateway route and authorize without parsing
#: the body.
MCP_NAME_BEARING_METHODS: Mapping[str, str] = {
    "tools/call": "name",
    "prompts/get": "name",
    "resources/read": "uri",
}

_ASCII_PRINTABLE = re.compile(r"^[\x20-\x7E]*$")


def encode_mcp_header_value(value: str) -> str:
    """A header value that a runtime will actually accept.

    Header values must be ASCII and a tool name need not be, so anything outside
    the printable range is percent-encoded rather than emitted raw and rejected.
    """
    from urllib.parse import quote

    return value if _ASCII_PRINTABLE.match(value) else quote(value, safe="")


def mcp_spec(spec_id: str) -> dict[str, Any]:
    """One MCP wire spec, with its inheritance chain applied."""
    from ..wire.inherit import resolve_spec

    if spec_id == "mcp/http.base":
        raise ValueError("mcp/http.base is a base spec and cannot build a request on its own")
    resolved: dict[str, Any] = resolve_spec(spec_id, WIRE_SPECS)
    return resolved


def mcp_wire_registry(base: Registry) -> Registry:
    """The shared registry plus the two rules only MCP has. Leaves `base` alone."""

    def mcp_modern(ctx: Ctx) -> bool:
        request = ctx.req if isinstance(ctx.req, Mapping) else {}
        return request.get("era") == "modern" or is_modern_mcp_version(
            str(request.get("protocolVersion") or "")
        )

    def mcp_name_header(_value: Any, ctx: Ctx) -> Any:
        request = ctx.req if isinstance(ctx.req, Mapping) else {}
        key = MCP_NAME_BEARING_METHODS.get(str(request.get("method")))
        if not key:
            return None
        params = request.get("params")
        subject = params.get(key) if isinstance(params, Mapping) else None
        return encode_mcp_header_value(subject) if isinstance(subject, str) else None

    return replace(
        base,
        predicates={**base.predicates, "mcpModern": mcp_modern},
        transforms={**base.transforms, "mcpNameHeader": mcp_name_header},
    )


__all__ = [
    "MCP_NAME_BEARING_METHODS",
    "encode_mcp_header_value",
    "mcp_spec",
    "mcp_wire_registry",
]
