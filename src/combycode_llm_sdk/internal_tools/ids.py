"""A tool's name, its namespace and its version, in one string.

`orxa:sentiment@1.0.0`. The version is IN the id rather than beside it because
the prompt is the implementation: a reworded instruction is a different tool,
and two callers pinned to different wordings must be able to name what they
each got. An id that carried only `orxa:sentiment` would make that unsayable.

Transposed from `unified-library-ts/src/plugins/internal-tools/id.ts`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ID = re.compile(r"^([a-z0-9_-]+):([a-z0-9_-]+)@([0-9]+\.[0-9]+\.[0-9]+)$")
_PART = re.compile(r"^[a-z0-9_-]+$")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


@dataclass(frozen=True)
class ParsedToolId:
    """The three pieces of a tool id."""

    namespace: str
    name: str
    version: str


def parse_tool_id(tool_id: str) -> ParsedToolId:
    """Split an id, or refuse one that is not shaped like an id."""
    match = _ID.match(tool_id)
    if not match:
        raise ValueError(
            f"invalid tool id {tool_id!r}: expected 'namespace:name@major.minor.patch'"
        )
    return ParsedToolId(namespace=match.group(1), name=match.group(2), version=match.group(3))


def try_parse_tool_id(tool_id: str) -> ParsedToolId | None:
    """The parse, or `None` -- for callers that have a fallback in mind."""
    try:
        return parse_tool_id(tool_id)
    except ValueError:
        return None


def format_tool_id(namespace: str, name: str, version: str) -> str:
    """The three pieces back into an id, each checked on the way in."""
    if not _PART.match(namespace):
        raise ValueError(f"invalid namespace {namespace!r}")
    if not _PART.match(name):
        raise ValueError(f"invalid tool name {name!r}")
    if not _VERSION.match(version):
        raise ValueError(f"invalid version {version!r}: expected semver X.Y.Z")
    return f"{namespace}:{name}@{version}"


def matches_version(requested: str, actual: str) -> bool:
    """Exact equality -- deliberately NOT semver range matching.

    A version here identifies a wording, not a compatibility promise: the prompt
    IS the implementation, so `1.0.1` can answer differently from `1.0.0` in a
    way no range syntax could describe. A caller pinned to a version gets that
    version or nothing.
    """
    return requested == actual


def id_without_version(tool_id: str) -> str:
    """`orxa:sentiment@1.0.0` -> `orxa:sentiment`. An unparseable id is returned whole."""
    parsed = try_parse_tool_id(tool_id)
    return tool_id if parsed is None else f"{parsed.namespace}:{parsed.name}"


__all__ = [
    "ParsedToolId",
    "format_tool_id",
    "id_without_version",
    "matches_version",
    "parse_tool_id",
    "try_parse_tool_id",
]
