"""The named escape hatches `google.interactions.stream.json` calls.

Transposed from
`unified-library-ts/src/llm/providers/google/interactions-stream-registry.ts`.

Two things span events: the id of the currently-open function call (its argument
fragments and its close carry only an index), and whether any tool was called at
all, which decides the finish reason at the end.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ....wire.interpreter import Ctx, Registry, js_json
from .._shared.response_utils import extract_finish_reason
from .parse_helpers import google_interactions_usage


def _raw(ctx: Ctx) -> Mapping[str, Any]:
    v = ctx.req.get("raw")
    return v if isinstance(v, Mapping) else {}


def _out(ctx: Ctx) -> dict[str, Any]:
    out: dict[str, Any] = ctx.req["out"]
    return out


def _delta(ctx: Ctx) -> Mapping[str, Any]:
    d = _raw(ctx).get("delta")
    return d if isinstance(d, Mapping) else {}


def _close_open_call(out: dict[str, Any]) -> None:
    """Shared by `step.stop` and the terminal frames, which flush defensively in
    case `step.stop` was omitted."""
    if not out["callId"]:
        return
    out["events"].append({"type": "tool_call_end", "id": out["callId"]})
    out["callId"] = None


def _args_delta(_arg: Any, ctx: Ctx) -> dict[str, Any]:
    """Argument fragments carry no id; they belong to the open call."""
    return {
        "type": "tool_call_delta",
        "id": _out(ctx)["callId"] or "",
        "arguments": _delta(ctx).get("arguments") or "",
    }


def _step_start(ctx: Ctx) -> None:
    out = _out(ctx)
    step = _raw(ctx).get("step")
    step = step if isinstance(step, Mapping) else {}
    # Remembered for the signature delta: `step.start` names the type, and the
    # delta that carries the signature names only itself.
    out["openStep"] = step.get("type") if isinstance(step.get("type"), str) else None
    if step.get("type") != "function_call":
        return
    call_id = step.get("id") or ""
    out["callId"] = call_id
    out["sawToolCall"] = True
    out["events"].append(
        {"type": "tool_call_start", "id": call_id, "name": step.get("name") or ""}
    )
    # Args normally stream via arguments_delta; forward an inline object too.
    args = step.get("arguments")
    if isinstance(args, Mapping) and len(args) > 0:
        out["events"].append(
            {"type": "tool_call_delta", "id": call_id, "arguments": js_json(args)}
        )


def _signature(ctx: Ctx) -> None:
    """A signature reaches a streamed turn ONLY here.

    Measured 2026-09-29: `step.start` announces `{"type": "thought"}` with no
    signature, the signature arrives as its own `step.delta`
    (`delta.type == "thought_signature"`), and the terminal
    `interaction.completed` carries the envelope WITHOUT steps. So a stream that
    ignores this delta loses the signature outright -- which is what the spec's
    old note, calling the delta "internal", assumed was harmless.

    Rebuilt into the step shape the buffered path returns, because that is the
    shape the API accepts back.
    """
    out = _out(ctx)
    signature = _delta(ctx).get("signature")
    if not isinstance(signature, str) or not signature:
        return
    out.setdefault("signatures", []).append(
        {"type": out.get("openStep") or "thought", "signature": signature}
    )


def _step_stop(ctx: Ctx) -> None:
    """step.stop carries only an index, so the open call is what it closes."""
    out = _out(ctx)
    _close_open_call(out)
    out["openStep"] = None


def _completed(ctx: Ctx) -> None:
    out = _out(ctx)
    raw = _raw(ctx)
    _close_open_call(out)
    interaction = raw.get("interaction")
    interaction = interaction if isinstance(interaction, Mapping) else {}
    # `interaction.usage`, and nothing else. There used to be a fallback to
    # `metadata.total_usage`; google 2.25 deleted `StreamMetadata{total_usage}` from
    # every event type, and the wire agrees -- measured 2026-10-01 by streaming a
    # real interaction with each read removed in turn: without the fallback usage
    # still arrives, and with ONLY the fallback no usage event fires at all. A
    # branch that provably never executes is a line that misdescribes the wire to
    # whoever reads it next.
    usage = interaction.get("usage")
    if isinstance(usage, Mapping):
        out["events"].append({"type": "usage", "usage": google_interactions_usage(usage)})
    # `queued` is NOT terminal (google 2.13): the interaction is still to run, so
    # it must never close the stream with a `done`.
    status = interaction.get("status")
    if status != "queued":
        done: dict[str, Any] = {
            "type": "done",
            "finishReason": extract_finish_reason(
                bool(out["sawToolCall"]),
                status if isinstance(status, str) else None,
                {"failed": "error"},
            ),
        }
        # Once, at the end, rather than as its own event: an opaque blob the
        # caller never reads is noise in a stream they do.
        if out.get("signatures"):
            done["signatures"] = out["signatures"]
        out["events"].append(done)


GOOGLE_INTERACTIONS_STREAM_REGISTRY = Registry(
    transforms={"gaStreamArgsDelta": _args_delta},
    effects={
        "gaStreamStepStart": _step_start,
        "gaStreamSignature": _signature,
        "gaStreamStepStop": _step_stop,
        "gaStreamCompleted": _completed,
    },
)

__all__ = ["GOOGLE_INTERACTIONS_STREAM_REGISTRY"]
