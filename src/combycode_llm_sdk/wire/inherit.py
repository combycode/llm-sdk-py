"""Spec inheritance: a spec version extends the previous one and overrides only
what changed.

Transposed from `unified-library-ts/src/wire/inherit.ts`.

This replaces the `variants` section. Instead of asking "which shape does this
model id take?" at request time -- which needed version arithmetic, and which
was WRONG for `claude-opus-4-20250514` (the date suffix parses as the minor
version, so a 4.0 model resolved to the 4.6+ shape) -- the catalog pins a model
to a spec id, and the spec chain carries the differences as deltas.

Merge rules, in one sentence each:
  - `fields`  are keyed by `to`   -- child replaces, new ones append
  - `blocks`  are keyed by `name` -- same, with `before`/`after` for placement
  - `tables`  merge per table, per key
  - `overlays`/`envelope.headers` are keyed and replaced
  - everything scalar: child wins

Block ORDER is load-bearing (proved by the mutation suite), so appended blocks
land at the end unless the delta says otherwise.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from .interpreter import js_json

Spec = dict[str, Any]

#: Keys with their own merge rule below. The list is of what is SPECIAL, not of
#: what is carried: enumerating the carried keys by hand dropped `url`, `method`,
#: `bodyKind` and then `multipart` in turn -- four instances of one mistake.
_SPECIAL = frozenset(
    {
        "id", "extends", "tables", "fields", "blocks", "unsupported", "envelope", "overlays",
        "removeFields", "removeBlocks", "placeBlocks",
        "provides", "order", "requires", "note", "_note",
    }
)


_T = TypeVar("_T")


def _clone(v: _T) -> _T:
    return copy.deepcopy(v)


def _merge_keyed(
    base: Sequence[Mapping[str, Any]],
    child: Sequence[Mapping[str, Any]] | None,
    key_of: Callable[[Mapping[str, Any]], str],
) -> list[Any]:
    out = list(base)
    for c in child or []:
        key = key_of(c)
        for i, b in enumerate(out):
            if key_of(b) == key:
                out[i] = c
                break
        else:
            out.append(c)
    return out


def _place(
    items: list[Any],
    key_of: Callable[[Mapping[str, Any]], str],
    placements: Mapping[str, Mapping[str, str]] | None,
) -> list[Any]:
    if not placements:
        return items
    out = list(items)
    for name, where in placements.items():
        idx = next((i for i, b in enumerate(out) if key_of(b) == name), -1)
        if idx < 0:
            raise ValueError(f"placeBlocks names an unknown block: {name}")
        item = out.pop(idx)
        anchor = where.get("before") or where.get("after")
        j = next((i for i, b in enumerate(out) if key_of(b) == anchor), -1)
        if j < 0:
            raise ValueError(f"placeBlocks anchor not found: {anchor}")
        out.insert(j if where.get("before") else j + 1, item)
    return out


def _require_present(op: str, names: Sequence[str], present: Sequence[str], spec_id: str) -> None:
    """A removal that matches nothing is ALWAYS a mistake -- a typo, or a delta
    left behind after the base renamed the thing it meant to drop.

    It used to filter silently, and silence is the worst behaviour here: whether
    a missed removal shows up on the wire depends on rule ORDER. A chain whose
    removals can quietly no-op is not a chain anyone can reason about.
    """
    absent = [n for n in names if n not in present]
    if absent:
        listed = ", ".join(f'"{a}"' for a in absent)
        raise ValueError(
            f"{spec_id or 'spec'}: {op} names {listed}, which the base does not define. "
            f"Present: {', '.join(present) or '(none)'}"
        )


def apply_delta(base: Mapping[str, Any], delta: Mapping[str, Any]) -> Spec:
    """Apply one delta to a resolved spec. This is the whole of composition."""
    out: Spec = _clone(dict(base))
    if delta.get("id"):
        out["id"] = delta["id"]

    for k, v in delta.items():
        if k in _SPECIAL or v is None:
            continue
        out[k] = _clone(v)

    if delta.get("tables"):
        tables = dict(out.get("tables") or {})
        for name, table in delta["tables"].items():
            tables[name] = {**(tables.get(name) or {}), **table}
        out["tables"] = tables

    out["fields"] = _merge_keyed(out.get("fields") or [], delta.get("fields"), lambda f: f["to"])
    if delta.get("removeFields"):
        _require_present(
            "removeFields", delta["removeFields"], [f["to"] for f in out["fields"]],
            delta.get("id", ""),
        )
        out["fields"] = [f for f in out["fields"] if f["to"] not in delta["removeFields"]]

    out["blocks"] = _merge_keyed(out.get("blocks") or [], delta.get("blocks"), lambda b: b["name"])
    if delta.get("removeBlocks"):
        _require_present(
            "removeBlocks", delta["removeBlocks"], [b["name"] for b in out["blocks"]],
            delta.get("id", ""),
        )
        out["blocks"] = [b for b in out["blocks"] if b["name"] not in delta["removeBlocks"]]
    out["blocks"] = _place(out["blocks"], lambda b: b["name"], delta.get("placeBlocks"))

    if delta.get("unsupported"):
        out["unsupported"] = [*(out.get("unsupported") or []), *delta["unsupported"]]

    if delta.get("envelope"):
        # Copy EVERY envelope key generically. Enumerating them by hand dropped
        # `url` and `method` (found by google/veo) and then `bodyKind` (found by
        # the file uploads) -- three instances of the same mistake, so the
        # enumeration itself was the bug. Headers still merge by name.
        delta_env = dict(delta["envelope"])
        delta_headers = delta_env.pop("headers", None)
        out["envelope"] = {**(out.get("envelope") or {}), **_clone(delta_env)}
        if delta_headers:
            # Headers merge by NAME so a child can override one the base set. A
            # spread entry has no name, so it is keyed by what it spreads: two
            # different spreads coexist, and re-declaring the same one replaces
            # it in place rather than duplicating it -- which would silently
            # re-apply a caller's header map after the overrides meant to beat it.
            def key_of(h: Mapping[str, Any]) -> str:

                return h.get("name") or f"spread:{js_json(h.get('spread'))}"

            out["envelope"]["headers"] = _merge_keyed(
                out["envelope"].get("headers") or [], delta_headers, key_of
            )

    if delta.get("overlays"):
        out["overlays"] = {**(out.get("overlays") or {}), **_clone(delta["overlays"])}

    # A resolved spec must not carry variants: the whole point is that the pin
    # already decided which version applies.
    out.pop("variants", None)
    return out


def resolve_spec(
    spec_id: str,
    by_id: Mapping[str, Mapping[str, Any]],
    seen: set[str] | None = None,
) -> Spec:
    """Resolve a spec id to its fully flattened form by walking `extends`."""
    seen = set() if seen is None else seen
    if spec_id in seen:
        raise ValueError(f"cycle in spec inheritance at {spec_id}")
    seen.add(spec_id)
    delta = by_id.get(spec_id)
    if delta is None:
        raise ValueError(f"unknown spec: {spec_id}")
    if not delta.get("extends"):
        return _clone(dict(delta))
    return apply_delta(resolve_spec(delta["extends"], by_id, seen), delta)


def spec_for_model(model: str, pins: Mapping[str, Any]) -> str:
    """Catalog pin: model id -> spec id. `default` is what an unknown model gets."""
    return str((pins.get("models") or {}).get(model, pins["default"]))


__all__ = ["apply_delta", "resolve_spec", "spec_for_model"]
