"""Which revision of MCP a session speaks, and how to ask.

MCP has two ERAS, not merely two versions:

- **handshake** (2024-11-05 ... 2025-11-25) -- `initialize` +
  `notifications/initialized`, a session id, and a server->client back-channel.
- **modern** (2026-07-28+) -- no handshake and no session id. One
  `server/discover`, then every request states its own identity in `_meta`.

Neither is preferred. Real servers overwhelmingly still speak 2025-11-25, so the
handshake path must keep working exactly as it does.

Transposed from `unified-library-ts/src/plugins/mcp/protocol-version.ts`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

#: Every released revision, oldest to newest. Verified against mcp-py 2.0.0
#: (`mcp_types/version.py`, KNOWN_PROTOCOL_VERSIONS).
MCP_KNOWN_PROTOCOL_VERSIONS: Final = (
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
    "2025-11-25",
    "2026-07-28",
)

#: Revisions reachable through the `initialize` handshake.
MCP_HANDSHAKE_PROTOCOL_VERSIONS: Final = (
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
    "2025-11-25",
)

#: Revisions that use the stateless per-request envelope (`server/discover`).
MCP_MODERN_PROTOCOL_VERSIONS: Final = ("2026-07-28",)

#: Newest handshake revision -- what `initialize` offers.
MCP_LATEST_HANDSHAKE_VERSION: Final = "2025-11-25"

#: Newest per-request-envelope revision -- what the discover probe asks for.
MCP_LATEST_MODERN_VERSION: Final = "2026-07-28"

McpEra = Literal["handshake", "modern"]


def mcp_era_of(version: str) -> McpEra:
    """Which wire a negotiated version implies.

    Version strings are an ENUMERATED SET, not an ordered scalar. Released
    revisions happen to be dates that sort lexicographically, but a future
    identifier need not be date-shaped, and an unrecognised peer string must
    compare conservatively rather than accidentally -- `"zzz" > "2025-11-25"` is
    true and meaningless. So era questions go through the lists above, never
    through `<` / `>`. An unknown version reads as `handshake`: the older, safer
    wire.
    """
    return "modern" if version in MCP_MODERN_PROTOCOL_VERSIONS else "handshake"


def is_modern_mcp_version(version: str) -> bool:
    """Whether this client can speak `version` on the modern wire."""
    return version in MCP_MODERN_PROTOCOL_VERSIONS


def is_handshake_mcp_version(version: str) -> bool:
    """Whether `version` is reachable through the `initialize` handshake."""
    return version in MCP_HANDSHAKE_PROTOCOL_VERSIONS


def newest_mutual_modern_version(theirs: Sequence[str]) -> str | None:
    """The newest modern revision both sides speak, or None if they share none."""
    mutual = [v for v in MCP_MODERN_PROTOCOL_VERSIONS if v in theirs]
    return mutual[-1] if mutual else None


def supported_versions_from(data: object) -> tuple[str, ...] | None:
    """The `data.supported` list off a `-32022`, or None when unusable.

    A malformed payload must read as "no information", never as an empty and
    therefore disjoint list -- the difference decides whether negotiation falls
    back or gives up.
    """
    if not isinstance(data, dict):
        return None
    raw = data.get("supported")
    if not isinstance(raw, (list, tuple)):
        return None
    versions = tuple(v for v in raw if isinstance(v, str))
    return versions or None


# -- `_meta` identity keys (modern era) --------------------------------------
# Verified against mcp-py 2.0.0 `mcp_types/_types.py`.

#: Required on every modern request: the revision this request is written at.
MCP_PROTOCOL_VERSION_META_KEY: Final = "io.modelcontextprotocol/protocolVersion"
#: Required on every modern request: what the client can do.
MCP_CLIENT_CAPABILITIES_META_KEY: Final = "io.modelcontextprotocol/clientCapabilities"
#: Optional client identity, display-only.
MCP_CLIENT_INFO_META_KEY: Final = "io.modelcontextprotocol/clientInfo"
#: Server identity stamped on modern results, display-only.
MCP_SERVER_INFO_META_KEY: Final = "io.modelcontextprotocol/serverInfo"

__all__ = [
    "MCP_CLIENT_CAPABILITIES_META_KEY",
    "MCP_CLIENT_INFO_META_KEY",
    "MCP_HANDSHAKE_PROTOCOL_VERSIONS",
    "MCP_KNOWN_PROTOCOL_VERSIONS",
    "MCP_LATEST_HANDSHAKE_VERSION",
    "MCP_LATEST_MODERN_VERSION",
    "MCP_MODERN_PROTOCOL_VERSIONS",
    "MCP_PROTOCOL_VERSION_META_KEY",
    "MCP_SERVER_INFO_META_KEY",
    "McpEra",
    "is_handshake_mcp_version",
    "is_modern_mcp_version",
    "mcp_era_of",
    "newest_mutual_modern_version",
    "supported_versions_from",
]
