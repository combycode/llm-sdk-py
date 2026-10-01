"""Two things a trace could not tell you about an agent run.

Transposed from
`unified-library-ts/tests/unit/plugins/telemetry-agent-attrs.test.ts`.

**Which agent ran a tool.** `execute_tool` spans carried an agent id and no
name, while the `invoke_agent` spans beside them were named -- so a backend
grouped tool calls under an opaque id and the agent's own spans under a label,
and joining them was the reader's problem. The name is read off the open agent
span rather than threaded through the tool hook's context: widening a public
shape for telemetry's benefit alone is the worse trade.

**What the run cost.** Token usage went onto each llm span and into process
totals. Neither answers "what did THIS run spend", which is the question
someone reading one trace is asking.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")


from combycode_llm_sdk.telemetry import TelemetryAdapter


class _Event:
    """What the bus hands a subscriber."""

    def __init__(self, name: str, **ctx: Any) -> None:
        self.type = name
        self.ctx = ctx


class _Adapter:
    """The adapter plus a span sink, driven the way the bus would drive it."""

    def __init__(self) -> None:
        self.seen: list[Any] = []
        # Tracing is opt-in; without it `_span_for` is never reached and the
        # adapter emits no spans at all.
        self.adapter = TelemetryAdapter(traces=True, redact_free_text=False)
        self.adapter.on_span_end(self.seen.append)

    def record(self, name: str, ctx: dict[str, Any]) -> None:
        self.adapter._record(_Event(name, **ctx))


def rig() -> tuple[_Adapter, list[Any]]:
    a = _Adapter()
    return a, a.seen


def completion(tokens_in: int, tokens_out: int) -> dict[str, Any]:
    return {
        "provider": "openai",
        "model": "gpt-5.4-nano",
        "response": {"model": "gpt-5.4-nano", "usage": {"inputTokens": tokens_in, "outputTokens": tokens_out}},
        "gen_ai.operation.name": "chat",
    }


class TestTheAgentNameOnAToolSpan:
    def test_it_is_the_label_the_run_was_opened_with(self) -> None:
        adapter, seen = rig()
        adapter.record("onRunStart", {"runId": "r1", "agentId": "a1", "label": "triage"})
        adapter.record("onToolCallStart", {"runId": "r1", "callId": "c1", "toolName": "search"})
        adapter.record("onToolCallComplete", {"runId": "r1", "callId": "c1", "toolName": "search"})

        tool = next(s for s in seen if s.kind == "tool")
        assert tool.attributes["gen_ai.agent.name"] == "triage"

    def test_it_is_absent_rather_than_invented_without_a_label(self) -> None:
        adapter, seen = rig()
        adapter.record("onRunStart", {"runId": "r1", "agentId": "a1"})
        adapter.record("onToolCallStart", {"runId": "r1", "callId": "c1", "toolName": "search"})
        adapter.record("onToolCallComplete", {"runId": "r1", "callId": "c1", "toolName": "search"})

        tool = next(s for s in seen if s.kind == "tool")
        assert "gen_ai.agent.name" not in tool.attributes

    def test_a_tool_call_outside_any_run_still_works(self) -> None:
        # A bare client can call a tool with no agent span open at all.
        adapter, seen = rig()
        adapter.record("onToolCallStart", {"callId": "c1", "toolName": "search"})
        adapter.record("onToolCallComplete", {"callId": "c1", "toolName": "search"})
        tool = next(s for s in seen if s.kind == "tool")
        assert "gen_ai.agent.name" not in tool.attributes


class TestPerRunTokenUsage:
    def test_it_sums_every_call_the_run_made(self) -> None:
        adapter, seen = rig()
        adapter.record("onRunStart", {"runId": "r1", "agentId": "a1"})
        adapter.record("onCompletion", completion(10, 3))
        adapter.record("onCompletion", completion(20, 7))
        adapter.record("onRunComplete", {"runId": "r1", "agentId": "a1", "reason": "stop"})

        agent = [s for s in seen if s.kind == "agent"][-1]
        assert agent.attributes["gen_ai.usage.input_tokens"] == 30
        assert agent.attributes["gen_ai.usage.output_tokens"] == 10

    def test_it_forgets_the_run_afterwards(self) -> None:
        # A long-lived process must not carry one run's tokens into the next.
        adapter, seen = rig()
        for run in ("r1", "r2"):
            adapter.record("onRunStart", {"runId": run, "agentId": "a1"})
            adapter.record("onCompletion", completion(5, 1))
            adapter.record("onRunComplete", {"runId": run, "agentId": "a1", "reason": "stop"})

        agents = [s for s in seen if s.kind == "agent"]
        # The second run reports ITS five tokens, not ten.
        assert agents[-1].attributes["gen_ai.usage.input_tokens"] == 5

    def test_it_says_nothing_when_the_run_made_no_calls(self) -> None:
        adapter, seen = rig()
        adapter.record("onRunStart", {"runId": "r1", "agentId": "a1"})
        adapter.record("onRunComplete", {"runId": "r1", "agentId": "a1", "reason": "stop"})
        agent = [s for s in seen if s.kind == "agent"][-1]
        assert "gen_ai.usage.input_tokens" not in agent.attributes
