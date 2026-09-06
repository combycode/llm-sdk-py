"""Build a unified response from a provider body, driven by a spec.

Transposed from `unified-library-ts/src/wire/response-interpreter.ts`.

The request side has been spec-driven since 3.0.0; the parse side was seven
hand-written parsers doing the same four things in four spellings. This is the
other half, and the reason the port is cheap: the evaluator underneath is the
one `interpreter.py` already carries, because it never cared what the root
object was. It resolves paths against `ctx.req`, and nothing in it is
request-shaped.

The one genuinely new thing is classification. Requests map a tree onto another
tree; responses must first decide WHAT each element of a heterogeneous array is
(Anthropic `content[]`, OpenAI `output[]`, Google `parts[]`) and fan it out.
`$map` cannot express that, because one element may need to land in TWO places
at once -- and as the same object, not a copy: the adapters push one reference
into both `content` and `toolCalls`, and a consumer that mutates
`response.toolCalls[0]` sees it in `response.content` too.

Paths resolve against ``{"raw": ..., "out": ...}`` in every phase::

    {"$": "raw.stop_reason"}     the provider body
    {"$map": "out.content"}      what has been collected so far
    {"$": "@text"}               the block currently being classified

`out` is readable during collection on purpose: Anthropic attaches a tool result
to the builtin call it belongs to by matching `tool_use_id` against calls
already collected, and that is a lookup into `out`, not into `raw`.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .interpreter import (
    MISSING,
    OMIT,
    Ctx,
    Item,
    Registry,
    eval_cond,
    eval_template,
    get_path,
    is_obj,
)

Json = Any


def init_accumulators(decls: Mapping[str, Any] | None) -> dict[str, Any]:
    """The accumulators, from their declarations.

    The default is COPIED, never handed out by reference. A spec's
    ``"default": {}`` is one object living in the loaded spec, so sharing it
    would let every build mutate the same map -- and on the streaming side, the
    second conversation would open holding the first one's pending tool calls.
    """
    out: dict[str, Any] = {}
    for name, decl in (decls or {}).items():
        out[name] = [] if decl.get("kind") == "array" else copy.deepcopy(decl.get("default"))
    return out


def emit_into(out: dict[str, Any], rule: Mapping[str, Any], value: Any, spec_id: str) -> None:
    names = rule.get("emit")
    names = names if isinstance(names, list) else [names]
    for name in names:
        if name not in out:
            # A typo here would silently discard everything it emitted, which is
            # the failure mode this whole exercise exists to end.
            raise ValueError(f'{spec_id}: emit names undeclared accumulator "{name}"')
        mode = rule.get("mode")
        if mode == "scalar":
            out[name] = value
            continue
        target = out[name]
        if not isinstance(target, list):
            raise TypeError(f'{spec_id}: cannot push into scalar accumulator "{name}"')
        if mode == "concat":
            if not isinstance(value, list):
                raise TypeError(
                    f'{spec_id}: concat into "{name}" needs an array, got {type(value).__name__}'
                )
            target.extend(value)
            continue
        target.append(value)


def build_response(
    spec: Mapping[str, Any],
    raw: Any,
    reg: Registry,
    *,
    extra: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec_id = spec.get("id", "spec")
    out = init_accumulators(spec.get("accumulators"))
    root = {"raw": raw, "out": out}

    # The spec is handed to the shared evaluator as the "spec" because `$table`
    # reads `ctx.spec["tables"]`. Nothing else on it is touched here.
    ctx = Ctx(
        req=root,
        spec=spec,
        flavor="",
        config=dict(config or {}),
        variants=set(),
        body={},
    )

    def apply(rule: Mapping[str, Any], c: Ctx) -> None:
        """One emit rule, in whatever scope it was given."""
        if not eval_cond(rule.get("when"), c, reg):
            return
        effect = rule.get("effect")
        if effect:
            fn = reg.effects.get(effect)
            if fn is None:
                raise ValueError(f'{spec_id}: unknown effect "{effect}"')
            fn(c)
            return
        value = eval_template(rule.get("as"), c, reg)
        if value is OMIT:
            return
        emit_into(out, rule, value, spec_id)

    # -- 1. scalars straight across -------------------------------------------
    result: dict[str, Any] = dict(extra or {})
    for f in spec.get("fields") or []:
        if not eval_cond(f.get("when"), ctx, reg):
            continue
        if "from" in f:
            v = get_path(root, f["from"])
            if v is MISSING:
                v = f.get("default", MISSING)
        else:
            t = eval_template(f.get("value"), ctx, reg)
            v = f.get("default", MISSING) if t is OMIT else t
        if v is not MISSING:
            result[f["to"]] = v

    # -- 2. seed the accumulators with what is not in any array ----------------
    for rule in spec.get("seed") or []:
        apply(rule, ctx)

    # -- 3. classify -----------------------------------------------------------
    for rule in spec.get("collect") or []:
        arr = get_path(root, rule["from"])
        if not isinstance(arr, list):
            continue
        for i, block in enumerate(arr):
            key = (
                str(get_path(block, rule["match"]))
                if rule.get("match") is not None and is_obj(block)
                else None
            )
            picked = (rule.get("cases") or {}).get(key) if key is not None else None
            if picked is None:
                picked = rule.get("default")
            if not picked:
                continue
            item_ctx = Ctx(
                req=ctx.req,
                spec=ctx.spec,
                flavor=ctx.flavor,
                config=ctx.config,
                variants=ctx.variants,
                body=ctx.body,
                item=Item(value=block, index=i, is_last=i == len(arr) - 1),
            )
            for r in picked if isinstance(picked, list) else [picked]:
                apply(r, item_ctx)

    # -- 4. what only makes sense once the walk is over ------------------------
    for rule in spec.get("finalize") or []:
        apply(rule, ctx)

    # -- 5. derive, now that everything is collected ---------------------------
    derived: dict[str, Any] = {}
    for name, tpl in (spec.get("derive") or {}).items():
        v = eval_template(tpl, ctx, reg)
        if v is not OMIT:
            derived[name] = v

    # -- 6. assemble -----------------------------------------------------------
    for name, decl in (spec.get("accumulators") or {}).items():
        if decl.get("internal"):
            continue
        v = out[name]
        if decl.get("kind") == "array" and decl.get("omitEmpty") and isinstance(v, list) and not v:
            continue
        result[name] = v
    result.update(derived)
    return result


__all__ = ["build_response", "emit_into", "init_accumulators"]
