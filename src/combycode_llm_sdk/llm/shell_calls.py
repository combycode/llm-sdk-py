"""The shell builtin tool: joining its two halves, and saying when it is waiting.

Transposed from `unified-library-ts/src/llm/shell-calls.ts`.

The shell tool behaves unlike every other builtin, in two ways that a caller
cannot be expected to work out from an empty answer.

1. A container-run call arrives as TWO output items -- the `shell_call` with the
   commands, then a `shell_call_output` with stdout/stderr -- linked by `call_id`.
   Reported as-is, one tool call looks like two. The streamed path joins them with
   its own state, in event order; the buffered path has the whole list at once and
   joins it here, so both report the same single call.

2. A LOCAL call is not a finished tool call at all: the model only asks, and
   whoever called has to run the commands and feed the output back. The turn then
   ends with `finishReason: "stop"` and empty text -- measured 2026-10-02,
   `text: ""`, `toolCalls: []`, no warning -- which is indistinguishable from a
   model that simply had nothing to say. That silence is what `shell_awaiting_note`
   exists to break.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def merge_shell_calls(calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One call per shell invocation, with its output folded in.

    Keyed on `callId` rather than position: the two items are adjacent in every
    response measured so far, but `call_id` is what the provider actually uses to
    link them, and a pairing that relies on order breaks silently the first time a
    turn interleaves two shell calls. A half with no partner is left alone -- an
    output whose call never arrived is still worth reporting.
    """
    rows = [dict(c) for c in calls]
    if not any(c.get("tool") == "shell" and c.get("callId") for c in rows):
        return rows
    out: list[dict[str, Any]] = []
    by_call_id: dict[str, dict[str, Any]] = {}
    for call in rows:
        key = call.get("callId") if call.get("tool") == "shell" else None
        open_call = by_call_id.get(key) if isinstance(key, str) else None
        if open_call is None:
            if isinstance(key, str):
                by_call_id[key] = call
            out.append(call)
            continue
        # Merge into the half already reported. Only fill what is missing: the call
        # half owns the commands and the environment, the output half owns the
        # output, and neither should overwrite a value the other established.
        for field in ("output", "code", "environment", "id"):
            if call.get(field) and not open_call.get(field):
                open_call[field] = call[field]
    return out


def shell_awaiting_note(calls: Sequence[Mapping[str, Any]] | None) -> str | None:
    """The warning owed when the model asked the caller to run something.

    Deliberately a pure function returning the note, with the client emitting it --
    the same shape as the OpenAI service-tier decision, for the same reason: an
    adapter has no business owning a hook.
    """
    waiting = [
        c
        for c in (calls or [])
        if c.get("tool") == "shell" and c.get("environment") == "local"
    ]
    if not waiting:
        return None
    commands = "; ".join(str(c["code"]) for c in waiting if c.get("code"))
    asked = (
        "a shell command" if len(waiting) == 1 else f"{len(waiting)} shell commands"
    )
    return (
        f"The model asked to run {asked} and is waiting on you: the `shell` tool was "
        "enabled with a local environment, so nothing ran and this turn ends with no "
        "answer. Run the commands in `builtinToolCalls[].code`"
        + (f" ({commands})" if commands else "")
        + ", then send the result back addressed to `callId`; or set "
        '`params.environment = {"type": "container_auto"}` to have OpenAI run them '
        "for you."
    )


__all__ = ["merge_shell_calls", "shell_awaiting_note"]
