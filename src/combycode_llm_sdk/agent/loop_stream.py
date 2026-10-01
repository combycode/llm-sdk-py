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

    #: Which step this state belongs to. Needed so an event forwarded from inside
    #: the accumulator can be stamped like the ones the loop yields itself -- a
    #: stream event without a step number cannot be correlated with the step that
    #: produced it.
    step: int = 0
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


def _accum_for(state: Any, call_id: str) -> Any:
    """Which accumulating tool call a delta or an end belongs to.

    By ID whenever there is one -- that is what the id is for, and with parallel
    calls in flight it is the only thing that can be right.

    When the event carries no id the answer is the MOST RECENTLY STARTED call,
    not the first. A stream delivers a call's deltas after its start, so "most
    recent" is the only reading that holds for more than one call. This used to
    take the first entry, which with two parallel Google function calls appended
    the second call's arguments to the first: `read_file` ended up with
    `{"path":"/a"}{"path":"/b"}` (unparseable, so refused as malformed) and
    `delete_file` ended up with NOTHING -- an empty args string, which is
    deliberately read as a genuine no-argument call, so it EXECUTED with `{}`.
    Exactly the failure `_parse_accum` exists to prevent, through another door.

    An id we have never seen is treated the same way: either the provider does
    not echo ids on deltas, or a start was missed, and the most recent call is
    the best available answer in both cases.
    """
    if call_id:
        found = state.accum.get(call_id)
        if found is not None:
            return found
    last = None
    for entry in state.accum.values():
        last = entry
    return last


def _parse_accum(entry: ToolCallAccumEntry) -> dict[str, Any]:
    """One accumulated call as a tool_call part.

    Unparseable arguments are MARKED, not quietly emptied. The old fallback was
    an empty argument object, reasoning that the model had already been paid for
    and a tool receiving `{}` would fail readably. It does not: `{}` is a VALID
    call, so a stream cut at `{"path": "/et` ran the tool with no arguments at
    all, and nothing downstream could tell that from a deliberate no-argument
    call. The call is kept -- dropping it would erase the model's intent -- but
    the loop refuses to execute it.

    An absent or blank `args` is NOT malformed: that is how a genuine
    no-argument call arrives.
    """
    raw = entry.args or ""
    parsed: Any = {}
    malformed = False
    if raw.strip():
        try:
            parsed = json.loads(raw)
        except ValueError:
            malformed = True
        else:
            # A bare scalar or list parses but is not an argument object.
            if not isinstance(parsed, dict):
                malformed = True
    call: dict[str, Any] = {
        "type": "tool_call",
        "id": entry.id,
        "name": entry.name,
        "arguments": parsed if isinstance(parsed, dict) and not malformed else {},
    }
    if malformed:
        call["malformed"] = True
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
        entry = _accum_for(state, str(event.get("id") or ""))
        if entry is not None:
            entry.args += str(event.get("arguments") or "")
        # Forwarded as well as accumulated. The loop still needs the whole string
        # to parse at `tool_call_end`, so this is not a handover -- it is a second
        # reader. A UI that wants to show the arguments forming had no way to see
        # them: the fragments arrived here and died, and `tool_call_start` only
        # fires once they are complete.
        return {
            "type": "tool_call_delta",
            "step": state.step,
            "callId": entry.id if entry is not None else str(event.get("id") or ""),
            "arguments": str(event.get("arguments") or ""),
        }

    if kind == "tool_call_end":
        entry = _accum_for(state, str(event.get("id") or ""))
        # De-duped on the ENTRY's own id, not the event's: an end with no id
        # resolves to an entry that may already have been pushed, and pushing it
        # twice runs the tool twice.
        if entry is not None and not any(c["id"] == entry.id for c in state.tool_calls):
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
            malformed=bool(call.get("malformed")),
            raw={"_meta": call["_meta"]} if "_meta" in call else {},
        )
        for call in state.tool_calls
    ]
    parts.extend(calls)

    # A step that asked for tools finished for that reason whatever the provider
    # called it: several report `stop` alongside a tool call, and a loop reading
    # the raw value would end the run with the call unanswered.
    #
    # Unless one of them cannot be run. A step holding a call we will refuse did
    # not finish in `tool_use`, and saying so is what lets reflect-and-retry fire
    # on EVERY provider: only Google's API reports this reason itself, so the
    # same truncation elsewhere was indistinguishable from a successful turn.
    if any(c.malformed for c in calls):
        finish = "malformed_tool_call"
    else:
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
