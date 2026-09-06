"""Every shipped wire spec, keyed by its id.

Transposed from `unified-library-ts/src/wire/registry.ts`. The TypeScript file
imports each JSON individually because its bundler needs the import to exist at
build time; Python reads the same directory instead. The SET is what matters and
it is identical -- `specs/` is vendored byte-for-byte from the TypeScript tree,
and a test asserts the count so a dropped file is a failure rather than a silent
gap.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SPECS_DIR = Path(__file__).resolve().parent / "specs"


def _load_all() -> dict[str, dict[str, Any]]:
    """Keyed by each spec's own `id`, which is what the TypeScript map keys on.

    A file whose `id` disagrees with its path would be invisible under the wrong
    key, so the mismatch is refused rather than tolerated.
    """
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(SPECS_DIR.rglob("*.json")):
        # REQUEST specs only, which is what `registry.ts` holds: it imports each
        # request spec by name, so the response and stream specs living in the
        # same tree were never in that map. Python reads the directory instead,
        # so the boundary has to be stated. They have their own loaders, and
        # their `$call` names resolve against per-provider registries -- mixing
        # them in here would let a request spec call a response helper.
        if path.parent.name in ("responses", "stream"):
            continue
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec_id = spec.get("id")
        if not spec_id:
            raise ValueError(f"{path} has no id")
        if spec_id in out:
            raise ValueError(f"two specs claim the id {spec_id!r}")
        out[spec_id] = spec
    return out


#: id -> the spec delta as written. Deltas, not resolved specs: `extends` is
#: walked by `resolve_spec` at lookup time.
WIRE_SPECS: Mapping[str, dict[str, Any]] = _load_all()


def get_wire_spec(spec_id: str) -> dict[str, Any]:
    spec = WIRE_SPECS.get(spec_id)
    if spec is None:
        raise ValueError(f"unknown wire spec: {spec_id}")
    return spec


__all__ = ["SPECS_DIR", "WIRE_SPECS", "get_wire_spec"]
