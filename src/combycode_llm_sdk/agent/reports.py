"""What a run did, after it did it.

Transposed from the report half of `unified-library-ts/src/agent/types.ts`.

A run is several LLM calls and several tool calls, and by the time it returns,
the only evidence left of the middle is what was recorded on the way through.
These are that record: one report per tool call, one per step, one per run.

Frozen, like everything else the caller receives. A report handed out and then
mutated by the next run is a debugging session nobody enjoys.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..results import Usage


@dataclass(frozen=True)
class ToolCallReport:
    """One tool call: what it was asked, what it cost, whether it ran."""

    call_id: str
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    result_size_bytes: int = 0
    latency_ms: float = 0.0
    skipped: bool = False
    #: The failure, as the model was told it. None when the call succeeded --
    #: distinguishable from `""`, which would be a tool that failed silently.
    error: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    #: `"search"` when the model reached this tool through `call_tool` after
    #: finding it with `tool_search`. `tool_name` is the tool that actually ran
    #: either way -- without that unwrapping, every lazy call would report
    #: `call_tool` and per-tool attribution would be worthless.
    discovered_via: str | None = None


@dataclass(frozen=True)
class StepReport:
    """One turn: the LLM call and the tool calls it asked for."""

    index: int
    #: `"initial"` for the first, `"tool_followup"` for every turn after.
    type: str
    llm_latency_ms: float
    usage: Usage
    finish_reason: str
    tool_calls: Sequence[ToolCallReport] = ()
    tool_total_ms: float = 0.0


@dataclass(frozen=True)
class AgentRunReport:
    """One `complete()` call, end to end.

    `reason` is why the loop stopped, and it is the field worth reading first:
    a run that hit `max_steps` produced an answer the same shape as one that
    finished, and only this says which happened.
    """

    id: str
    model: str
    started_at: float
    completed_at: float
    total_ms: float
    #: `done` | `stopped` | `error` | `guardrail` | `max_steps`
    reason: str
    user_message: Any
    final_text: str
    steps: Sequence[StepReport] = ()
    step_count: int = 0
    tool_call_count: int = 0
    total_usage: Usage = field(default_factory=Usage)
    total_llm_time_ms: float = 0.0
    total_tool_time_ms: float = 0.0
    error: str | None = None


@dataclass
class ToolExecutionContext:
    """What a tool body is told about the call it is serving.

    Mutable, unlike the reports: `metrics` is an out-parameter a tool fills in,
    and freezing it would mean handing back a copy nobody remembered to collect.
    """

    step: int
    call_id: str
    #: Whatever the tool wants recorded against this call. Reaches
    #: `ToolCallReport.metrics`; the model never sees it.
    metrics: dict[str, Any] = field(default_factory=dict)
    #: `{sessionId, requestId, callId}` -- the run's identity, for telemetry
    #: that has to join a tool call back to the run that made it.
    trace: Mapping[str, Any] = field(default_factory=dict)


def add_usage(total: Usage, step: Usage) -> Usage:
    """Sum two usages, field by field.

    `Usage` is frozen, so this returns a new one rather than accumulating in
    place. The audio fields stay None unless one side reported them: zero is a
    claim that silence was billed, and None is the absence of a claim.
    """

    def _opt(a: int | None, b: int | None) -> int | None:
        return None if a is None and b is None else (a or 0) + (b or 0)

    return Usage(
        input_tokens=total.input_tokens + step.input_tokens,
        output_tokens=total.output_tokens + step.output_tokens,
        total_tokens=total.total_tokens + step.total_tokens,
        cached_tokens=total.cached_tokens + step.cached_tokens,
        cache_write_tokens=total.cache_write_tokens + step.cache_write_tokens,
        reasoning_tokens=total.reasoning_tokens + step.reasoning_tokens,
        audio_input_tokens=_opt(total.audio_input_tokens, step.audio_input_tokens),
        audio_output_tokens=_opt(total.audio_output_tokens, step.audio_output_tokens),
        # The tier is a property of ONE call, not of a sum. Keeping the first
        # one seen would report a whole run as whatever tier its first step
        # happened to use.
        service_tier=None,
        pricing_tier=None,
    )


__all__ = [
    "AgentRunReport",
    "StepReport",
    "ToolCallReport",
    "ToolExecutionContext",
    "add_usage",
]
