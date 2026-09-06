"""Turn a provider's SSE events into unified stream events, driven by a spec.

Transposed from `unified-library-ts/src/wire/stream-interpreter.ts`.

The buffered interpreter gets the whole body at once and builds one object. A
stream arrives in fragments, and the parse is a small state machine: an
Anthropic `server_tool_use` accumulates its input JSON across several deltas
before it can be paired with the `*_tool_result` that completes it; OpenAI
correlates tool-call fragments by index because only the first carries an id;
Google emits its hosted-tool events once per stream and has to remember that.

How little of this is new: measured across the five hand-written TypeScript
parsers, 518 lines of code of which 38 touched state. The other 93% is dispatch
and mapping -- the same thing the buffered specs express -- so this driver is
the buffered one with two differences:

1. ``out`` is created ONCE for the stream, not per call, so an accumulator is
   how the state machine remembers.
2. One reserved accumulator, ``events``, is drained and returned after each SSE
   event. Emitting a unified event means emitting into it.

Everything else -- the emit rules, the whole evaluator -- is shared verbatim.

Paths resolve against ``{"raw": ..., "out": ..., "event": ...}``::

    {"$": "raw.delta.text"}          the parsed event payload
    {"$": "out.current.tool"}        state carried across events
    {"eq": ["event.name", "ping"]}   the SSE envelope, for keep-alives

``event.data`` is the payload as it arrived, so a spec can match a sentinel like
``[DONE]`` that is not JSON at all.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
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
from .response_interpreter import emit_into, init_accumulators

Json = Any

#: The accumulator every stream spec emits into. Declared by the driver, not by
#: the spec, because a stream that cannot emit is not a stream.
EVENTS = "events"


def _payload_of(data: str) -> dict[str, Any]:
    """A payload that is not JSON is not an error: `[DONE]` and keep-alives are
    normal, and a spec matches them on `event.data` instead."""
    try:
        v = json.loads(data)
    except (ValueError, TypeError):
        return {}
    return v if is_obj(v) else {}


def create_stream_builder(
    spec: Mapping[str, Any],
    reg: Registry,
    *,
    config: Mapping[str, Any] | None = None,
) -> Callable[[Mapping[str, Any]], list[Any]]:
    """Build a stream parser: call the returned function per SSE event, and it
    returns the unified events that event produced (often none)."""
    spec_id = spec.get("id", "spec")
    state_decls = spec.get("state") or {}
    if EVENTS in state_decls:
        raise ValueError(f'{spec_id}: "{EVENTS}" is reserved and declared by the driver')

    # Created ONCE. This is the whole difference from the buffered interpreter.
    out = init_accumulators(state_decls)
    out[EVENTS] = []

    def parse(event: Mapping[str, Any]) -> list[Any]:
        raw = _payload_of(str(event.get("data", "")))
        envelope = {"name": event.get("event"), "data": event.get("data")}
        root = {"raw": raw, "out": out, "event": envelope}

        ctx = Ctx(
            req=root,
            # `$table` reads `ctx.spec["tables"]`; nothing else is touched.
            spec=spec,
            flavor="",
            config=dict(config or {}),
            variants=set(),
            body={},
            # The payload is also the current item, so `@`-paths address it
            # exactly as they address a block inside `collect`.
            item=Item(value=raw, index=0, is_last=True),
        )

        def apply(rule: Mapping[str, Any], c: Ctx) -> None:
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

        # Drained per event: what this call returns is what this event produced.
        # The rest of `out` survives, which is how the machine remembers.
        out[EVENTS] = []

        for rule in spec.get("on") or []:
            when = rule.get("when")
            if when is not None and not eval_cond(when, ctx, reg):
                continue

            scopes: list[Ctx] = []
            each = rule.get("each")
            if each is not None:
                arr = get_path(root, each)
                # A missing or non-list source is not an error: a chunk with no
                # parts is a normal chunk.
                if isinstance(arr, list):
                    for i, value in enumerate(arr):
                        scopes.append(
                            Ctx(
                                req=ctx.req,
                                spec=ctx.spec,
                                flavor=ctx.flavor,
                                config=ctx.config,
                                variants=ctx.variants,
                                body=ctx.body,
                                item=Item(value=value, index=i, is_last=i == len(arr) - 1),
                            )
                        )
            else:
                scopes.append(ctx)

            for scope in scopes:
                subject = scope.item.value if each is not None and scope.item else raw
                key: str | None = None
                match = rule.get("match")
                if match is not None:
                    # A LIST means "whichever of these is present", first
                    # defined wins: Google Interactions discriminates on
                    # `event_type ?? type`, and two rules would fire both.
                    for path in match if isinstance(match, list) else [match]:
                        v = get_path(subject, path)
                        if v is not MISSING:
                            key = str(v)
                            break
                    if key is None:
                        # Every candidate path was absent: no case can match,
                        # but `default` still applies.
                        key = str(None)

                picked = (rule.get("cases") or {}).get(key) if key is not None else None
                if picked is None:
                    picked = rule.get("default")
                if not picked:
                    continue
                for r in picked if isinstance(picked, list) else [picked]:
                    apply(r, scope)

            # AFTER the rule body, so `stop` is "handle this and go no further".
            # Anthropic's ping needs the plain form (a rule with nothing to do,
            # which just stops); OpenAI's moderation and usage-only chunks need
            # to emit and then stop, which is the early return the hand-written
            # parser does.
            if rule.get("stop"):
                return list(out[EVENTS])

        return list(out[EVENTS])

    return parse


__all__ = ["EVENTS", "create_stream_builder"]
