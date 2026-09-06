"""The wire specs the RUNTIME loads: chat only, resolved and memoised.

Transposed from `unified-library-ts/src/wire/chat-specs.ts`.

Deliberately not `registry.py`. In TypeScript that index imports all 71 specs --
every media, realtime, files and batch spec included -- so an adapter importing
it would pull the whole set into every bundle whether or not anything reads them.
Python has no bundle to grow, but the SET still matters for a different reason:
`registry.py` is the complete index the tests and ports check against, while this
is the subset the runtime can actually build, and it grows a family at a time as
each adapter is migrated. Keeping them distinct is what makes "a catalog pin
names a spec we cannot build yet" answerable instead of a crash.

Chains are resolved once per id and cached: resolution walks `extends` and merges
deltas, which is pure setup work and has no business happening per request.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .inherit import resolve_spec

_SPECS_DIR = Path(__file__).resolve().parent / "specs"

#: The nine chat specs the runtime carries, by path. The TypeScript imports each
#: by name; the paths are listed here for the same reason -- so a spec file that
#: disappears is an error at import, not a silent gap at request time.
_CHAT_SPEC_FILES = (
    "anthropic-chain/messages@4.0.json",
    "anthropic-chain/messages@4.1.json",
    "anthropic-chain/messages@4.6.json",
    "anthropic-chain/messages@4.7.json",
    "google-chain/generate@2.5.json",
    "google-chain/generate@3.json",
    "google-interactions.json",
    "openai-completions.json",
    "openai-responses.json",
)


def _load_deltas() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rel in _CHAT_SPEC_FILES:
        path = _SPECS_DIR / rel
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec_id = spec.get("id")
        if not spec_id:
            raise ValueError(f"{path} has no id")
        out[spec_id] = spec
    return out


DELTAS: dict[str, dict[str, Any]] = _load_deltas()

_resolved: dict[str, dict[str, Any]] = {}


def chat_spec(spec_id: str) -> dict[str, Any]:
    """The spec for `spec_id`, with its inheritance chain already applied.

    Raises on an unknown id rather than falling back to something plausible: a
    silently-substituted spec is a wrong request sent confidently, which is the
    exact failure the specs exist to end. Callers pick the fallback themselves --
    see each adapter's default spec.
    """
    hit = _resolved.get(spec_id)
    if hit is not None:
        return hit
    spec = resolve_spec(spec_id, DELTAS)
    _resolved[spec_id] = spec
    return spec


def is_chat_spec(spec_id: str | None) -> bool:
    """Whether a spec id is one the runtime can build.

    Lets an adapter fall back to its default instead of raising when a catalog
    pin names a spec from a family that is not migrated yet.
    """
    return spec_id is not None and spec_id in DELTAS


#: Ids the runtime carries -- asserted by the tests so this list and the shipped
#: spec files cannot drift apart unnoticed.
CHAT_SPEC_IDS: tuple[str, ...] = tuple(sorted(DELTAS.keys()))

__all__ = ["CHAT_SPEC_IDS", "DELTAS", "chat_spec", "is_chat_spec"]
