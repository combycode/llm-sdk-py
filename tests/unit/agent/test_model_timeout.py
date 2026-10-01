"""Bounding the MODEL half of a step.

Transposed from `unified-library-ts/tests/unit/agent/model-timeout.test.ts`.

`tool_timeout` already bounded the tool half. The model half was bounded only by
whatever the client was configured with -- so on a long run one slow step could
hold the whole run open past any deadline the caller thought they had set.

Three decisions worth pinning, because each is a thing someone will assume the
other way: it is per STEP not per run (a run-wide budget is a cancellation the
caller already controls); a per-call `timeout` still wins, because this is the
run's DEFAULT and not a cap; and there is no new error type, since the network
layer already reports a timeout and a second class would make every consumer learn
both.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")

from combycode_llm_sdk.agent.loop import AgentLoop
from combycode_llm_sdk.results import Completion, Part, Usage


class Client:
    """Records the options each step was given."""

    id = "client_1"
    provider = "anthropic"
    model = "claude-haiku-4.5"

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    def complete(self, _messages: Any, **options: Any) -> Completion:
        self.seen.append(dict(options))
        return Completion(
            text="ok",
            model="m",
            finish_reason="stop",
            usage=Usage(),
            parts=[Part(type="text", text="ok")],
        )

    def destroy(self) -> None:
        pass


class TestModelTimeout:
    def test_it_reaches_the_model_call_as_the_step_default(self) -> None:
        client = Client()
        AgentLoop(client, system="s", model_timeout=30.0).complete("go")
        assert client.seen[0]["timeout"] == 30.0

    def test_it_is_absent_when_nobody_set_it(self) -> None:
        # Not a default smuggled in: a run with no timeout configured must send
        # none, or every existing caller silently acquires one.
        client = Client()
        AgentLoop(client, system="s").complete("go")
        assert client.seen[0].get("timeout") is None

    def test_it_yields_to_a_per_call_timeout(self) -> None:
        # `model_timeout` is the run's default, not a cap on what one call may ask.
        client = Client()
        AgentLoop(client, system="s", model_timeout=30.0).complete("go", timeout=90.0)
        assert client.seen[0]["timeout"] == 90.0

    def test_it_applies_to_every_step_not_once_per_run(self) -> None:
        # The per-step reading is the whole point.
        client = Client()
        loop = AgentLoop(client, system="s", model_timeout=5.0)
        loop.complete("one")
        loop.complete("two")
        assert [o["timeout"] for o in client.seen] == [5.0, 5.0]

    def test_it_leaves_the_tool_timeout_alone(self) -> None:
        # Two budgets for two halves of a step; setting one must not set the other.
        client = Client()
        loop = AgentLoop(client, system="s", model_timeout=5.0, tool_timeout=60.0)
        loop.complete("go")
        assert client.seen[0]["timeout"] == 5.0
