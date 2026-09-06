"""Every shipped response spec, keyed by its id.

Transposed from `unified-library-ts/src/wire/response-specs.ts`. TypeScript
imports each spec file by name because its bundler needs the import to exist at
build time; Python reads the directory. The SET is what matters and a test
asserts the count, so a dropped file is a failure rather than a silent gap.

The id follows the corpus target key -- `anthropic/messages` becomes
`anthropic/messages.response` -- so the differential can find a target's spec
without a second mapping table to keep in step.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RESPONSE_SPECS_DIR = Path(__file__).resolve().parent / "specs" / "responses"


def _load_all() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(RESPONSE_SPECS_DIR.glob("*.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec_id = spec.get("id")
        if not spec_id:
            raise ValueError(f"{path} has no id")
        if spec_id in out:
            raise ValueError(f"two response specs claim the id {spec_id!r}")
        out[spec_id] = spec
    return out


#: id -> the spec as written. `extends` is walked by `get_response_spec`.
RESPONSE_SPECS: Mapping[str, dict[str, Any]] = _load_all()


def response_spec_id(target: str) -> str:
    """The spec id for a corpus/runtime target key, e.g. `openai/completions`."""
    return f"{target}.response"


def get_response_spec(spec_id: str, seen: set[str] | None = None) -> dict[str, Any]:
    """The spec FLATTENED: `extends` is walked here, so no caller sees a delta
    and mistakes it for the whole thing.

    Merge rules mirror the TypeScript side, one line each:
      accumulators / derive / tables  merge by key, child wins
      fields                          merge by `to`, child replaces, new append
      seed / collect / finalize       APPEND, parent first

    Append rather than merge for the ordered ones, because their order is their
    meaning: a child adding to `content` adds AFTER what the parent put there.
    """
    seen = set() if seen is None else seen
    if spec_id in seen:
        raise ValueError(f"cycle in response spec inheritance at {spec_id}")
    seen.add(spec_id)
    spec = RESPONSE_SPECS.get(spec_id)
    if spec is None:
        raise ValueError(f"unknown response spec: {spec_id}")
    if not spec.get("extends"):
        return spec

    base = get_response_spec(spec["extends"], seen)
    fields = list(base.get("fields") or [])
    for f in spec.get("fields") or []:
        at = next((i for i, x in enumerate(fields) if x.get("to") == f.get("to")), -1)
        if at >= 0:
            fields[at] = f
        else:
            fields.append(f)
    return {
        **base,
        **spec,
        "accumulators": {**(base.get("accumulators") or {}), **(spec.get("accumulators") or {})},
        "fields": fields,
        "seed": [*(base.get("seed") or []), *(spec.get("seed") or [])],
        "collect": [*(base.get("collect") or []), *(spec.get("collect") or [])],
        "finalize": [*(base.get("finalize") or []), *(spec.get("finalize") or [])],
        "derive": {**(base.get("derive") or {}), **(spec.get("derive") or {})},
        "tables": {**(base.get("tables") or {}), **(spec.get("tables") or {})},
    }


__all__ = ["RESPONSE_SPECS", "get_response_spec", "response_spec_id"]
