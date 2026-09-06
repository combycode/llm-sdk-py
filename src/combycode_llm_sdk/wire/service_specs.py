"""Service specs: batch, files, count, embeddings, moderation, transcription.

Transposed from `unified-library-ts/src/wire/service-specs.ts`. Resolution walks
`extends` through the shared registry; only the abstract list and the caching are
this module's own.
"""

from __future__ import annotations

from typing import Any

from .inherit import resolve_spec
from .registry import WIRE_SPECS

#: Bases exist to be extended, not called. Building one would produce a request
#: addressed at nothing, so the id is refused rather than resolved.
ABSTRACT = frozenset(
    {
        "xai/media.base",
        "xai/images.base",
        "openrouter/media.base",
        "anthropic/batch.base",
        "openai/batch.base",
        "xai/batch.base",
        "anthropic/files.base",
        "openai/files.base",
    }
)

_resolved: dict[str, dict[str, Any]] = {}


def service_spec(spec_id: str) -> dict[str, Any]:
    """The service spec for `id`, with its inheritance chain applied.

    Throws on an unknown or abstract id rather than substituting something
    plausible.
    """
    hit = _resolved.get(spec_id)
    if hit is not None:
        return hit
    if spec_id in ABSTRACT:
        raise ValueError(f"{spec_id} is a base spec and cannot build a request on its own")
    spec = resolve_spec(spec_id, WIRE_SPECS)
    _resolved[spec_id] = spec
    return spec


#: Buildable ids, asserted by the tests so this list cannot drift from the files.
SERVICE_SPEC_IDS: tuple[str, ...] = tuple(k for k in WIRE_SPECS if k not in ABSTRACT)

__all__ = ["ABSTRACT", "SERVICE_SPEC_IDS", "service_spec"]
