"""Every shipped stream spec, keyed by its id.

Transposed from `unified-library-ts/src/wire/stream-specs.ts`.

The id follows the corpus target key -- `anthropic/messages` becomes
`anthropic/messages.stream` -- so the differential finds a target's spec without
a second table to keep in step.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

STREAM_SPECS_DIR = Path(__file__).resolve().parent / "specs" / "stream"


def _load_all() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(STREAM_SPECS_DIR.glob("*.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec_id = spec.get("id")
        if not spec_id:
            raise ValueError(f"{path} has no id")
        if spec_id in out:
            raise ValueError(f"two stream specs claim the id {spec_id!r}")
        out[spec_id] = spec
    return out


#: id -> the spec as written. `extends` is walked by `get_stream_spec`.
STREAM_SPECS: Mapping[str, dict[str, Any]] = _load_all()


def stream_spec_id(target: str) -> str:
    """The stream spec id for a corpus/runtime target key."""
    return f"{target}.stream"


def get_stream_spec(spec_id: str, seen: set[str] | None = None) -> dict[str, Any]:
    """Flattened: `extends` is walked here, so no caller sees a delta and
    mistakes it for the whole thing.

    `state` merges by key; `on` is APPENDED parent-first, because rule order is
    rule meaning -- a child rule runs after everything the parent declared.
    """
    seen = set() if seen is None else seen
    if spec_id in seen:
        raise ValueError(f"cycle in stream spec inheritance at {spec_id}")
    seen.add(spec_id)
    spec = STREAM_SPECS.get(spec_id)
    if spec is None:
        raise ValueError(f"unknown stream spec: {spec_id}")
    if not spec.get("extends"):
        return spec
    base = get_stream_spec(spec["extends"], seen)
    return {
        **base,
        **spec,
        "state": {**(base.get("state") or {}), **(spec.get("state") or {})},
        "on": [*(base.get("on") or []), *(spec.get("on") or [])],
        "tables": {**(base.get("tables") or {}), **(spec.get("tables") or {})},
    }


__all__ = ["STREAM_SPECS", "get_stream_spec", "stream_spec_id"]
