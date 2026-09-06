"""The utility spec family: token counting, model listing, file content, provenance.

Transposed from `unified-library-ts/src/wire/utility-specs.ts`.

Four small surfaces that share nothing except being requests. They are loaded
together because each is a handful of specs and no consumer reaches one without
reaching the layer that owns it.

The TypeScript imports its fourteen JSON files by name because its bundler needs
the import to exist at build time. Python resolves against the shared registry
instead -- `registry.py` already reads the whole `specs/` tree -- so what this
module adds is the ID SET and the cache. The set is written out rather than
derived from the directory: it is the statement of which specs this family owns,
and a spec that moves into `specs/files/` without being listed here should be a
lookup failure, not a silent extension of the family.
"""

from __future__ import annotations

from typing import Any

from .inherit import resolve_spec
from .registry import WIRE_SPECS

#: The fourteen ids, grouped as the TypeScript groups its imports.
UTILITY_SPEC_IDS: tuple[str, ...] = (
    # exact token counting -- each provider names its own endpoint, so the ids
    # are not `<provider>/count` however much the FILENAMES suggest it.
    "anthropic/count.messages",
    "google/count.tokens",
    "xai/count.tokenize",
    # live model listing
    "openai/models.list",
    "anthropic/models.list",
    "google/models.list",
    "xai/models.list",
    "openrouter/models.list",
    # file content retrieval
    "openai/files.content",
    "openai/files.content.container",
    "anthropic/files.content",
    "google/files.content",
    "files/download.byUrl",
    # provenance
    "openai/provenance.check",
)

_resolved: dict[str, dict[str, Any]] = {}


def utility_spec(spec_id: str) -> dict[str, Any]:
    """Resolve a utility spec by id, flattening its `extends` chain.

    Raises on an id outside the family rather than resolving whatever else the
    registry happens to hold under that name -- the families are separate so a
    file-content request cannot accidentally be built from a chat spec.
    """
    hit = _resolved.get(spec_id)
    if hit is not None:
        return hit
    if spec_id not in UTILITY_SPEC_IDS:
        raise ValueError(f"{spec_id} is not a utility spec")
    spec = resolve_spec(spec_id, WIRE_SPECS)
    _resolved[spec_id] = spec
    return spec


__all__ = ["UTILITY_SPEC_IDS", "utility_spec"]
