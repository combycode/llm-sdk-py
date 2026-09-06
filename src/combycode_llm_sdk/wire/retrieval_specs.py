"""Retrieval specs: corpora, uploads, indexing and search.

Transposed from `unified-library-ts/src/wire/retrieval-specs.ts`.
"""

from __future__ import annotations

from typing import Any

from .inherit import resolve_spec
from .registry import WIRE_SPECS

#: Bases exist to be extended, not called. Building one addresses nothing.
ABSTRACT = frozenset(
    {
        "openai/retrieval.base",
        "openai/retrieval.json",
        "google/retrieval.base",
        "google/retrieval.json",
        "xai/retrieval.base",
        "xai/retrieval.management",
        "xai/retrieval.standard",
        "xai/retrieval.standardJson",
    }
)

_resolved: dict[str, dict[str, Any]] = {}


def retrieval_spec(spec_id: str) -> dict[str, Any]:
    """The retrieval spec for `id`, with its inheritance chain applied."""
    hit = _resolved.get(spec_id)
    if hit is not None:
        return hit
    if spec_id in ABSTRACT:
        raise ValueError(f"{spec_id} is a base spec and cannot build a request on its own")
    spec = resolve_spec(spec_id, WIRE_SPECS)
    _resolved[spec_id] = spec
    return spec


__all__ = ["ABSTRACT", "retrieval_spec"]
