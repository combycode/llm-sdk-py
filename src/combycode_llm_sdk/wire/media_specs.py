"""Media specs: image, audio and video generation.

Transposed from `unified-library-ts/src/wire/media-specs.ts`.
"""

from __future__ import annotations

from typing import Any

from .inherit import resolve_spec
from .registry import WIRE_SPECS

#: Bases exist to be extended, not called.
ABSTRACT = frozenset(
    {
        "openai/media.base",
        "openai/media.images",
        "google/media.generateContent",
        "google/media.predict",
    }
)

_resolved: dict[str, dict[str, Any]] = {}


def media_spec(spec_id: str) -> dict[str, Any]:
    """The media spec for `id`, with its inheritance chain applied."""
    hit = _resolved.get(spec_id)
    if hit is not None:
        return hit
    if spec_id in ABSTRACT:
        raise ValueError(f"{spec_id} is a base spec and cannot build a request on its own")
    spec = resolve_spec(spec_id, WIRE_SPECS)
    _resolved[spec_id] = spec
    return spec


__all__ = ["ABSTRACT", "media_spec"]
