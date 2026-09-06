"""Turning a provider's token stream back into one step of an agent run.

Transposed from `unified-library-ts/src/agent/loop-internals.ts` and
`loop-step-state.ts`.

A streamed step arrives as deltas and has to end up the same shape a buffered
one produces, because everything downstream -- history, the step report, the run
report, the final answer -- reads that shape and cannot tell how the step was
fetched. The accumulator is where the two paths meet.

Kept out of `loop.py` for the reason the TypeScript gives: the step loop is long
enough already, and this is the half that is pure. Nothing here touches history,
hooks, or the client, so it can be tested by feeding it events.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..results import Citation, Completion, Part, Usage

#: The narrow set an agent consumer sees. Deliberately not every provider event:
#: a file or a builtin-tool boundary reaches the caller on the final response
#: instead, so a UI wiring these straight through cannot be surprised by a kind
#: it has no branch for.
AgentStreamEvent = dict[str, Any]


@dataclass
class ToolCallAccumEntry:
    """One tool call being spelled out across deltas."""

    id: str
    name: str
    args: str = ""
    meta: Mapping[str, Any] | None = None


@dataclass
class StepState:
    """Everything one streaming step accumulates."""

    text: str = ""
    #: Narration, kept apart from `text` so the step's ANSWER excludes it. The
    #: buffered path applies the same rule through `final_answer_text`.
    commentary: str = ""
    thinking: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    accum: dict[str, ToolCallAccumEntry] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"
    #: Keyed by url, because Google repeats its grounding chunks across late
    #: chunks and one page cited twice is one source.
    citations: dict[str, Citation] = field(default_factory=dict)


def _parse_accum(entry: ToolCallAccumEntry) -> dict[str, Any]:
    """One accumulated call as a tool_call part.

    Invalid JSON becomes an empty argument object rather than an exception: the
    model has already been paid for, and a tool that receives `{}` fails with a
    message the model can read and retry, where a raised parse error ends the
    run.
    """
    try:
        parsed = json.loads(entry.args or "{}")
    except ValueError:
        parsed = {}
    call: dict[str, Any] = {
        "type": "tool_call",
        "id": entry.id,
        "name": entry.name,
        "arguments": parsed if isinstance(parsed, dict) else {},
    }
    if entry.meta:
        call["_meta"] = dict(entry.meta)
    return call


def accumulate_stream_event(event: Mapping[str, Any], state: StepState) -> AgentStreamEvent | None:
    """Fold one provider event into the step. Returns what to yield, or None."""
    kind = event.get("type")

    if kind == "text":
        phase = event.get("phase")
        text = str(event.get("text") or "")
        if phase == "commentary":
            state.commentary += text
        else:
            state.text += text
        # The phase is FORWARDED, not merely used here. Dropping it leaves the
        # consumer unable to tell narration from the answer while it streams,
        # and `final_answer_text` cannot help -- it reads a finished message.
        out: AgentStreamEvent = {"type": "text", "text": text}
        if phase:
            out["phase"] = phase
        return out

    if kind == "thinking":
        state.thinking += str(event.get("text") or "")
        return {"type": "thinking", "text": str(event.get("text") or "")}

    if kind == "tool_call_start":
        call_id = str(event.get("id") or "")
        state.accum[call_id] = ToolCallAccumEntry(
            id=call_id, name=str(event.get("name") or ""), meta=event.get("_meta")
        )
        return None

    if kind == "tool_call_delta":
        # Falling back to the first open entry is not tidiness: several providers
        # send argument deltas with no id at all, and dropping those silently
        # produces a tool call with empty arguments and no error anywhere.
        entry = state.accum.get(str(event.get("id") or ""))
        if entry is None:
            entry = next(iter(state.accum.values()), None)
        if entry is not None:
            entry.args += str(event.get("arguments") or "")
        return None

    if kind == "tool_call_end":
        entry = state.accum.get(str(event.get("id") or "")) if event.get("id") else None
        if entry is None:
            done = {c["id"] for c in state.tool_calls}
            entry = next((a for a in state.accum.values() if a.id not in done), None)
        if entry is not None:
            state.tool_calls.append(_parse_accum(entry))
        return None

    if kind == "citation":
        citation = event.get("citation")
        url = getattr(citation, "url", None) or (
            citation.get("url") if isinstance(citation, Mapping) else None
        )
        if citation is not None and url:
            state.citations[str(url)] = citation
        return None

    if kind == "usage":
        usage = event.get("usage")
        if usage is not None:
            state.usage = usage
        return None

    if kind == "done":
        state.finish_reason = str(event.get("finishReason") or state.finish_reason)
        return None

    return None


def finalize_unended_tool_calls(state: StepState) -> None:
    """Close any call that never got its `tool_call_end`.

    Not defensive padding: Anthropic and OpenAI both stream calls that end only
    when the message does, so without this the last tool call of a step is
    silently dropped and the model is answered as though it had not asked.
    """
    done = {call["id"] for call in state.tool_calls}
    for call_id, entry in state.accum.items():
        if call_id not in done:
            state.tool_calls.append(_parse_accum(entry))


def build_step_completion(state: StepState, model: str, latency_ms: float) -> Completion:
    """The step, in the shape the buffered path would have produced."""
    parts: list[Part] = []
    # Commentary stays as its own phase-tagged part rather than being discarded,
    # so a streamed step carries the same shape as a buffered one. A phase is
    # stamped only where the provider reported one -- inferring `final_answer`
    # for a model that reports nothing would be a guess.
    if state.commentary:
        parts.append(Part(type="text", text=state.commentary, phase="commentary"))
    if state.text:
        # `final_answer` only when there IS commentary to distinguish it from.
        parts.append(
            Part(
                type="text",
                text=state.text,
                phase="final_answer" if state.commentary else None,
            )
        )
    calls = [
        Part(
            type="tool_call",
            id=call["id"],
            name=call["name"],
            arguments=call.get("arguments") or {},
            raw={"_meta": call["_meta"]} if "_meta" in call else {},
        )
        for call in state.tool_calls
    ]
    parts.extend(calls)

    # A step that asked for tools finished for that reason whatever the provider
    # called it: several report `stop` alongside a tool call, and a loop reading
    # the raw value would end the run with the call unanswered.
    finish = "tool_use" if calls else state.finish_reason

    return Completion(
        text=state.text,
        model=model,
        finish_reason=finish,
        usage=state.usage,
        parts=tuple(parts),
        tool_calls=tuple(calls),
        thinking=state.thinking or None,
        citations=tuple(state.citations.values()),
        latency_ms=latency_ms,
        raw=None,
    )


def tool_call_start_events(step: int, calls: Iterable[Mapping[str, Any]]) -> list[AgentStreamEvent]:
    """What a consumer is told before the tools run."""
    return [
        {
            "type": "tool_call_start",
            "step": step,
            "callId": call.get("id"),
            "toolName": call.get("name"),
            "arguments": dict(call.get("arguments") or {}),
        }
        for call in calls
    ]


def tool_call_end_events(step: int, reports: Iterable[Any]) -> list[AgentStreamEvent]:
    """What a consumer is told after them, with what each one cost."""
    return [
        {
            "type": "tool_call_end",
            "step": step,
            "callId": getattr(report, "call_id", None) or getattr(report, "id", None),
            "latencyMs": getattr(report, "latency_ms", 0.0),
        }
        for report in reports
    ]


__all__ = [
    "AgentStreamEvent",
    "StepState",
    "ToolCallAccumEntry",
    "accumulate_stream_event",
    "build_step_completion",
    "finalize_unended_tool_calls",
    "tool_call_end_events",
    "tool_call_start_events",
]
