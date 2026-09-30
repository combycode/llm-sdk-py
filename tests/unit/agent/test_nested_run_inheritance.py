"""A nested agent run belongs to the run above it.

Transposed from
`unified-library-ts/tests/unit/agent/nested-run-inheritance.test.ts`, with one
deliberate divergence carried over from `AgentLoop.stop()`: cancellation here is
COOPERATIVE. Python cannot kill a running worker, and pretending otherwise would
leak a half-applied side effect while reporting a clean stop -- so a tool is told
a stop was asked for and decides for itself, rather than having a signal fire at
it. A sub-agent given that stops at its next step boundary, which is the same
guarantee the parent gives itself.

The trace half has no divergence. A nested run used to be started with nothing,
so its spans rooted a trace of their own and the two halves of one request could
not be joined. Worse in one specific way: `_step_options` already let a caller's
ctx reach the LLM calls, so those joined the caller's trace while the RUN that
made them -- onRunStart, every tool call -- sat in a second, unrelated one.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")


from combycode_llm_sdk import current_tool_run, delegate, handoff, nested_run_options, tool
from combycode_llm_sdk.agent.loop import AgentLoop
from combycode_llm_sdk.agent.tool_run import ToolRunContext
from combycode_llm_sdk.bus.hook_bus import HookBus
from combycode_llm_sdk.results import Completion, Part, Usage


def completion(**over: Any) -> Completion:
    fields: dict[str, Any] = {
        "text": "",
        "model": "m",
        "finish_reason": "stop",
        "usage": Usage(),
        "parts": [],
    }
    fields.update(over)
    return Completion(**fields)


def one_call(name: str) -> Completion:
    part = Part(
        type="tool_call",
        id="c1",
        name=name,
        arguments={"task": "summarise"},
        raw={"type": "tool_call"},
    )
    return completion(parts=[part], tool_calls=[part], finish_reason="tool_use")


class Client:
    """Replays a queue and records the options each call was given."""

    id = "client_1"

    def __init__(self, queue: list[Completion]) -> None:
        self.queue = list(queue)
        self.seen: list[dict[str, Any]] = []

    def complete(self, messages: Any, **options: Any) -> Completion:
        self.seen.append(options)
        return self.queue.pop(0)


class TestWhatAToolIsToldAboutItsCall:
    def test_it_sees_the_call_it_is_serving(self) -> None:
        seen: list[ToolRunContext | None] = []

        @tool
        def look(task: str) -> str:
            """Look something up."""
            seen.append(current_tool_run())
            return "ok"

        client = Client([one_call("look"), completion(text="done")])
        AgentLoop(client, tools=[look]).run("go")

        ctx = seen[0]
        assert ctx is not None
        assert ctx.tool_name == "look"
        assert ctx.call_id == "c1"
        assert ctx.trace.get("requestId")

    def test_it_crosses_the_worker_thread(self) -> None:
        # A pool worker starts with an EMPTY context, so without copying it the
        # tool would see nothing -- and a delegating tool would silently inherit
        # nothing, which is the bug this exists to fix.
        seen: list[ToolRunContext | None] = []

        @tool
        def look(task: str) -> str:
            """Look something up, on whatever thread the loop chose."""
            seen.append(current_tool_run())
            return "ok"

        client = Client([one_call("look"), completion(text="done")])
        AgentLoop(client, tools=[look], tool_timeout=5).run("go")
        assert seen[0] is not None

    def test_outside_a_tool_there_is_nothing_to_report(self) -> None:
        # None is a normal answer: a delegate tool stays directly callable, and
        # testing one without a loop must not require a fake context.
        assert current_tool_run() is None
        assert nested_run_options() == {}

    def test_a_long_tool_can_ask_whether_a_stop_was_requested(self) -> None:
        # Cooperative by design -- see the module docstring. The flag can flip
        # WHILE the tool runs, which is why it is read through a callable.
        answers: list[bool] = []
        loop_ref: list[AgentLoop] = []

        @tool
        def look(task: str) -> str:
            """Check whether the caller has given up."""
            ctx = current_tool_run()
            assert ctx is not None
            answers.append(ctx.is_stopping())
            loop_ref[0].stop()
            answers.append(ctx.is_stopping())
            return "ok"

        client = Client([one_call("look"), completion(text="done")])
        loop = AgentLoop(client, tools=[look])
        loop_ref.append(loop)
        loop.run("go")

        assert answers == [False, True]


class TestNestedRunOptions:
    ctx = ToolRunContext(
        tool_name="helper",
        call_id="call_9",
        step=2,
        trace={"sessionId": "sess_1", "requestId": "run_1", "traceparent": "00-abc-def-01"},
    )

    def test_it_passes_the_whole_trace(self) -> None:
        # Not a field or two of it: sessionId and requestId are resolved together
        # when a run begins, and half of them is how one run becomes two traces.
        out = nested_run_options(self.ctx)["ctx"]
        assert out["sessionId"] == "sess_1"
        assert out["requestId"] == "run_1"
        assert out["traceparent"] == "00-abc-def-01"

    def test_it_names_the_call_that_delegated(self) -> None:
        assert nested_run_options(self.ctx)["ctx"]["callId"] == "call_9"


class TestDelegateAndHandoffPassItDown:
    def nested(self, make_tool: Any) -> tuple[Client, Client]:
        child_client = Client([completion(text="summary")])
        child = AgentLoop(child_client)
        parent_client = Client([one_call("helper"), completion(text="relayed")])
        parent = AgentLoop(parent_client, tools=[make_tool(child)])
        parent.run("delegate it")
        return child_client, parent_client

    def test_delegate_joins_the_callers_trace(self) -> None:
        child, parent = self.nested(lambda c: delegate("helper", "Summarise", c))
        child_ctx = child.seen[0]["ctx"]
        parent_ctx = parent.seen[0]["ctx"]
        assert child_ctx["sessionId"] == parent_ctx["sessionId"]
        assert child_ctx["requestId"] == parent_ctx["requestId"]
        # And is attributed to the call that delegated, not the parent at large.
        assert child_ctx["callId"] == "c1"

    def test_handoff_does_the_same(self) -> None:
        child, parent = self.nested(lambda c: handoff("helper", "Summarise", c))
        child_ctx = child.seen[0]["ctx"]
        assert child_ctx["requestId"] == parent.seen[0]["ctx"]["requestId"]
        assert child_ctx["callId"] == "c1"

    def test_the_child_keeps_its_own_conversation_id(self) -> None:
        # It is a different agent with a different history; joining the trace is
        # not the same as pretending to be the same conversation.
        child, parent = self.nested(lambda c: delegate("helper", "Summarise", c))
        assert child.seen[0]["ctx"]["conversationId"] != parent.seen[0]["ctx"]["conversationId"]


class TestTheRunsOwnSpansJoinTooNotJustItsLlmCalls:
    def test_on_run_start_carries_the_callers_trace(self) -> None:
        # The specific shape of the old bug: a caller's ctx reached the LLM calls
        # through `_step_options` while the run that made them sat elsewhere.
        hooks = HookBus()
        starts: list[dict[str, Any]] = []
        hooks.on("onRunStart", lambda c: starts.append(dict(c)))

        client = Client([completion(text="hi")])
        loop = AgentLoop(client, hooks=hooks)
        loop.run("go", ctx={"sessionId": "outer", "requestId": "outer_run"})

        trace = starts[0]["trace"]
        assert trace["sessionId"] == "outer"
        assert trace["requestId"] == "outer_run"

    def test_a_traceparent_is_carried_when_given(self) -> None:
        client = Client([completion(text="hi")])
        loop = AgentLoop(client)
        loop.run("go", ctx={"traceparent": "00-abc-def-01"})
        assert client.seen[0]["ctx"]["traceparent"] == "00-abc-def-01"

    def test_without_a_caller_ctx_the_run_mints_its_own(self) -> None:
        client = Client([completion(text="hi")])
        loop = AgentLoop(client)
        loop.run("go")
        ctx = client.seen[0]["ctx"]
        assert ctx["sessionId"] == loop.id
        assert str(ctx["requestId"]).startswith("run_")
        assert "traceparent" not in ctx


class TestAStoppedParentStopsTheChild:
    """Cooperatively: the child is handed the question, and asks it each step.

    The parent cannot reach into the child's flag -- they are separate loops --
    so what travels is `stop_when`, and the child consults it exactly where it
    consults its own stop.
    """

    def test_the_child_stops_at_its_next_step_boundary(self) -> None:
        # A child that would otherwise take two steps: one tool call, then an
        # answer. The parent gives up while the child's tool is running, so the
        # child must not start a second step.
        parent_ref: list[AgentLoop] = []

        @tool
        def slow(task: str) -> str:
            """Take a while, and notice the caller giving up."""
            parent_ref[0].stop()
            return "partial"

        child_client = Client(
            [one_call("slow"), completion(text="would have continued")]
        )
        child = AgentLoop(child_client, tools=[slow])

        parent_client = Client([one_call("helper"), completion(text="relayed")])
        parent = AgentLoop(parent_client, tools=[delegate("helper", "Summarise", child)])
        parent_ref.append(parent)
        parent.run("delegate it")

        # One step: the child asked for `slow`, and stopped rather than stepping
        # again. Without the inherited stop it would have consumed both replies.
        assert len(child_client.seen) == 1
        assert len(child_client.queue) == 1

    def test_without_a_parent_stop_the_child_runs_to_its_own_end(self) -> None:
        @tool
        def quick(task: str) -> str:
            """Answer at once."""
            return "done"

        child_client = Client([one_call("quick"), completion(text="finished")])
        child = AgentLoop(child_client, tools=[quick])
        parent_client = Client([one_call("helper"), completion(text="relayed")])
        parent = AgentLoop(parent_client, tools=[delegate("helper", "Summarise", child)])
        parent.run("delegate it")

        assert len(child_client.seen) == 2
        assert child_client.queue == []

    def test_stop_when_never_reaches_the_provider_as_an_option(self) -> None:
        # It is ours, not a request field. Forwarding it would put a callable in
        # the body of an HTTP request.
        client = Client([completion(text="hi")])
        AgentLoop(client).run("go", stop_when=lambda: False)
        assert "stop_when" not in client.seen[0]
