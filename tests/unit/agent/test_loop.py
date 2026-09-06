"""`AgentLoop` -- the durable loop, driven through a scripted client.

The loop's job is the sequence: what it sends, what it keeps, what it reports,
and when it stops. A scripted client is the only way to assert on all four
without a provider deciding half of them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from combycode_llm_sdk import tool
from combycode_llm_sdk.agent.loop import AgentLoop, ToolNameCollision
from combycode_llm_sdk.agent.reflect_retry import ReflectAndRetry
from combycode_llm_sdk.llm.output_errors import AgentRunError
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


def answer(text: str, **over: Any) -> Completion:
    return completion(text=text, parts=[Part(type="text", text=text)], **over)


def calls(*specs: tuple[str, str, dict[str, Any]]) -> Completion:
    parts = [
        Part(type="tool_call", id=i, name=n, arguments=a, raw={"type": "tool_call"})
        for i, n, a in specs
    ]
    return completion(parts=list(parts), tool_calls=list(parts), finish_reason="tool_use")


class Client:
    """A scripted LLM client, recording every call it was given."""

    id = "client_1"
    provider = "anthropic"
    model = "claude-haiku-4.5"

    def __init__(self, *turns: Completion) -> None:
        self.turns = list(turns)
        self.seen: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
        self.destroyed = False

    def complete(self, messages: Any, **options: Any) -> Completion:
        self.seen.append(([dict(m) for m in messages], dict(options)))
        index = min(len(self.seen) - 1, len(self.turns) - 1)
        return self.turns[index]

    def destroy(self) -> None:
        self.destroyed = True


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"sunny in {city}"


@tool
def get_population(city: str) -> str:
    """Get the population of a city."""
    return "2.1 million"


class TestTheConversationIsKept:
    def test_a_second_call_sees_the_first(self) -> None:
        # The whole difference between an Agent and a `complete()` call.
        client = Client(answer("one"), answer("two"))
        loop = AgentLoop(client)
        loop.complete("first question")
        loop.complete("second question")

        second_request = client.seen[1][0]
        assert [m["role"] for m in second_request] == ["user", "assistant", "user"]
        assert len(loop.history) == 4

    def test_the_identity_survives_a_clear(self) -> None:
        # Hooks and observers are bound to it; a clear that re-identified the
        # agent would silently detach every one of them.
        loop = AgentLoop(Client(answer("x")))
        before = loop.id
        loop.complete("hi")
        loop.history.clear()
        assert len(loop.history) == 0
        # The HISTORY's id, not only the loop's copy of it: the history is what
        # stamps `conversationId` on every later call, so an id that changed
        # here would detach every observer while `loop.id` still looked right.
        assert loop.history.id == before
        assert loop.id == before

    def test_the_system_prompt_is_sent_every_step(self) -> None:
        client = Client(calls(("c1", "get_weather", {"city": "P"})), answer("done"))
        AgentLoop(client, system="You are terse.", tools=[get_weather]).complete("?")
        assert all(options["system"] == "You are terse." for _, options in client.seen)

    def test_context_is_appended_after_the_persona(self) -> None:
        client = Client(answer("x"))
        AgentLoop(client, system="You are terse.", context="The user is offline.").complete("?")
        assert client.seen[0][1]["system"] == "You are terse.\n\nThe user is offline."

    def test_a_callable_system_is_re_read_every_run(self) -> None:
        # Live-reload prompts: a config file that changed between runs.
        prompts = iter(["first", "second"])
        client = Client(answer("a"), answer("b"))
        loop = AgentLoop(client, system=lambda: next(prompts))
        loop.complete("one")
        loop.complete("two")
        assert [options["system"] for _, options in client.seen] == ["first", "second"]


class TestTheStepMachine:
    def test_a_tool_round_trip_produces_two_steps(self) -> None:
        client = Client(calls(("c1", "get_weather", {"city": "Paris"})), answer("Sunny."))
        loop = AgentLoop(client, tools=[get_weather])
        assert loop.complete("weather?").text == "Sunny."

        report = loop.last_report
        assert report is not None
        assert report.step_count == 2
        assert report.tool_call_count == 1
        assert report.reason == "done"

    def test_the_tool_result_reaches_the_next_request(self) -> None:
        client = Client(calls(("c1", "get_weather", {"city": "Paris"})), answer("Sunny."))
        AgentLoop(client, tools=[get_weather]).complete("?")
        roles = [m["role"] for m in client.seen[1][0]]
        assert roles == ["user", "assistant", "tool"]
        assert client.seen[1][0][2]["content"][0]["content"] == "sunny in Paris"

    def test_max_steps_stops_a_model_that_never_finishes(self) -> None:
        # An agent that never stops is a bill that never stops.
        looping = calls(("c1", "get_weather", {"city": "P"}))
        loop = AgentLoop(Client(looping), tools=[get_weather], max_steps=3)
        result = loop.complete("?")
        assert loop.last_report is not None
        assert loop.last_report.reason == "max_steps"
        assert loop.last_report.step_count == 3
        assert result.finish_reason == "max_steps"

    def test_a_non_positive_max_steps_means_the_default_not_unlimited(self) -> None:
        loop = AgentLoop(Client(answer("x")), max_steps=0)
        assert loop._max_steps == 16

    def test_stop_ends_the_run_after_the_current_step(self) -> None:
        client = Client(calls(("c1", "get_weather", {"city": "P"})), answer("done"))
        loop = AgentLoop(client, tools=[get_weather])
        loop.hooks.on("onStepComplete", lambda ctx: loop.stop())
        loop.complete("?")
        assert loop.last_report is not None
        assert loop.last_report.reason == "stopped"

    def test_usage_is_the_run_total_not_the_last_step(self) -> None:
        # A caller billing on `result.usage` after a three-step run would
        # otherwise be told about a third of it.
        step = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
        client = Client(
            calls(("c1", "get_weather", {"city": "P"})).__class__(
                **{
                    **calls(("c1", "get_weather", {"city": "P"})).__dict__,
                    "usage": step,
                }
            ),
            answer("done", usage=step),
        )
        loop = AgentLoop(client, tools=[get_weather])
        result = loop.complete("?")
        assert result.usage.output_tokens == 10
        assert result.usage.input_tokens == 20


class TestTools:
    def test_a_failing_tool_reports_back_to_the_model(self) -> None:
        @tool
        def explode(city: str) -> str:
            """Always fails."""
            raise RuntimeError("no such city")

        client = Client(calls(("c1", "explode", {"city": "X"})), answer("I could not."))
        loop = AgentLoop(client, tools=[explode])
        assert loop.complete("?").text == "I could not."

        report = loop.last_report
        assert report is not None
        assert report.steps[0].tool_calls[0].error is not None
        assert "no such city" in report.steps[0].tool_calls[0].error

    def test_parallel_calls_are_all_answered(self) -> None:
        client = Client(
            calls(
                ("c1", "get_weather", {"city": "P"}),
                ("c2", "get_population", {"city": "P"}),
            ),
            answer("both"),
        )
        loop = AgentLoop(client, tools=[get_weather, get_population])
        loop.complete("?")
        results = client.seen[1][0][2]["content"]
        assert [r["id"] for r in results] == ["c1", "c2"]

    def test_a_duplicate_name_warns_by_default(self) -> None:
        seen: list[Any] = []
        loop = AgentLoop(Client(answer("x")), tools=[get_weather])
        loop.hooks.on("onWarning", seen.append)

        @tool
        def collides(city: str) -> str:
            """A different tool that will be renamed onto an existing one."""
            return "other"

        collides.definition["name"] = "get_weather"
        loop.register_tool(collides)
        assert [w["code"] for w in seen] == ["tool_name_collision"]

    def test_a_duplicate_name_can_be_made_fatal(self) -> None:
        # Before the model is ever called, rather than as "it called the wrong
        # tool" much later. Two DISTINCT tools sharing a name: handing the same
        # object in twice shadows nothing and is not a collision.
        @tool
        def shadow(city: str) -> str:
            """A second tool that will be renamed onto an existing one."""
            return "other"

        shadow.definition["name"] = "get_weather"
        with pytest.raises(ToolNameCollision):
            AgentLoop(
                Client(answer("x")),
                tools=[get_weather, shadow],
                tool_name_collision="error",
            )

    def test_the_same_tool_handed_in_twice_is_not_fatal(self) -> None:
        # Idempotent re-registration replaces a tool with itself. Refusing it
        # would break any caller that re-adds a tool defensively.
        loop = AgentLoop(
            Client(answer("x")),
            tools=[get_weather, get_weather],
            tool_name_collision="error",
        )
        assert loop.tool_names() == ["get_weather"]


class TestLazyTools:
    def test_a_lazy_tool_is_registered_but_not_declared(self) -> None:
        @tool(lazy=True)
        def rebuild_index(namespace: str) -> str:
            """Rebuild the search index for a namespace."""
            return "queued"

        loop = AgentLoop(Client(answer("x")), tools=[get_weather, rebuild_index])
        declared = {t["name"] for t in loop.declared_tools()}
        assert "get_weather" in declared
        assert "rebuild_index" not in declared
        # ...but registered, or lazy would just mean missing.
        assert loop.has_tool("rebuild_index")

    def test_the_built_ins_appear_only_when_something_is_lazy(self) -> None:
        eager = AgentLoop(Client(answer("x")), tools=[get_weather])
        assert {t["name"] for t in eager.declared_tools()} == {"get_weather"}

    def test_call_tool_reports_the_tool_that_actually_ran(self) -> None:
        # A trace that attributes every lazy call to `call_tool` is useless.
        @tool(lazy=True)
        def rebuild_index(namespace: str) -> str:
            """Rebuild the search index for a namespace."""
            return "queued"

        client = Client(
            calls(("c1", "call_tool", {"name": "rebuild_index", "input": {"namespace": "a"}})),
            answer("done"),
        )
        loop = AgentLoop(client, tools=[rebuild_index])
        loop.complete("?")

        report = loop.last_report
        assert report is not None
        call = report.steps[0].tool_calls[0]
        assert call.tool_name == "rebuild_index"
        assert call.discovered_via == "search"
        assert client.seen[1][0][2]["content"][0]["content"] == "queued"


class TestReflectAndRetry:
    def test_a_recoverable_failure_is_retried_with_guidance(self) -> None:
        client = Client(
            completion(finish_reason="malformed_tool_call"),
            answer("recovered"),
        )
        loop = AgentLoop(
            client, reflect_and_retry=ReflectAndRetry(max_attempts=2, on=["malformed_tool_call"])
        )
        assert loop.complete("?").text == "recovered"

        # The guidance went back as a user turn, and the FAILED turn did not:
        # the model must not learn from its own broken output.
        second = client.seen[1][0]
        assert second[-1]["role"] == "user"
        assert "retry attempt 1 of 2" in second[-1]["content"]
        assert not any(m["role"] == "assistant" for m in second)

    def test_the_budget_is_bounded_and_then_raises(self) -> None:
        # Unbounded reflection is an infinite loop with a bill attached.
        client = Client(completion(finish_reason="malformed_tool_call"))
        loop = AgentLoop(client, reflect_and_retry=ReflectAndRetry(max_attempts=1))
        with pytest.raises(AgentRunError, match="2 times in a row") as caught:
            loop.complete("?")
        assert caught.value.reason == "model_failure_retry_exhausted"

    def test_it_can_hand_back_the_last_response_instead(self) -> None:
        client = Client(completion(text="unusable", finish_reason="malformed_tool_call"))
        loop = AgentLoop(
            client,
            reflect_and_retry=ReflectAndRetry(max_attempts=0, raise_if_exceeded=False),
        )
        assert loop.complete("?").text == "unusable"

    def test_a_success_returns_the_budget_within_one_run(self) -> None:
        """The streak is CONSECUTIVE failures, and a good turn breaks it.

        Deliberately inside ONE run: the loop also resets the policy BETWEEN
        runs, so a fail/succeed/fail spread over two runs passes whether or not
        `record_success` does anything -- which is exactly what the first
        version of this test did, and a mutation caught it. Here the second
        failure is only survivable because the tool turn returned the budget.
        """
        client = Client(
            completion(finish_reason="malformed_tool_call"),
            calls(("c1", "get_weather", {"city": "P"})),
            completion(finish_reason="malformed_tool_call"),
            answer("recovered twice"),
        )
        loop = AgentLoop(
            client, tools=[get_weather], reflect_and_retry=ReflectAndRetry(max_attempts=1)
        )
        assert loop.complete("?").text == "recovered twice"

    def test_failures_are_not_carried_between_runs(self) -> None:
        client = Client(
            completion(finish_reason="malformed_tool_call"),
            answer("recovered"),
            completion(finish_reason="malformed_tool_call"),
            answer("recovered again"),
        )
        loop = AgentLoop(client, reflect_and_retry=ReflectAndRetry(max_attempts=1))
        assert loop.complete("one").text == "recovered"
        assert loop.complete("two").text == "recovered again"

    def test_a_finish_reason_it_does_not_handle_is_left_alone(self) -> None:
        client = Client(answer("fine"))
        loop = AgentLoop(client, reflect_and_retry=ReflectAndRetry(on=["malformed_tool_call"]))
        assert loop.complete("?").text == "fine"
        assert len(client.seen) == 1


class TestGuardrails:
    def test_a_raising_guard_stops_the_run(self) -> None:
        class BudgetExceeded(RuntimeError):
            pass

        def budget(ctx: Any) -> None:
            raise BudgetExceeded("too much")

        loop = AgentLoop(Client(answer("x")), after=[budget])
        with pytest.raises(BudgetExceeded):
            loop.complete("?")
        assert loop.last_report is not None
        assert loop.last_report.reason == "error"

    def test_an_input_guard_runs_before_the_call(self) -> None:
        client = Client(answer("x"))

        def refuse(ctx: Any) -> None:
            raise RuntimeError("not allowed")

        with pytest.raises(RuntimeError):
            AgentLoop(client, before=[refuse]).complete("?")
        assert client.seen == []

    def test_a_failing_decision_halts_without_raising_out(self) -> None:
        # The TypeScript shape: a decision object rather than an exception.
        loop = AgentLoop(Client(answer("x")), after=[lambda ctx: {"pass": False, "reason": "no"}])
        result = loop.complete("?")
        assert loop.last_report is not None
        assert loop.last_report.reason == "guardrail"
        assert result.text == "no"

    def test_a_healthy_run_is_undisturbed(self) -> None:
        loop = AgentLoop(Client(answer("done")), before=[lambda ctx: None], after=[lambda c: None])
        assert loop.complete("hi").text == "done"

    def test_the_output_guard_sees_the_response(self) -> None:
        seen: list[Any] = []
        AgentLoop(Client(answer("hello")), after=[seen.append]).complete("?")
        assert seen[0].kind == "output"
        assert seen[0].response.text == "hello"


class TestHooksAndIdentity:
    def test_every_call_carries_the_conversation_id(self) -> None:
        # How an Observer tells one agent's traffic from another's when both
        # share an engine.
        client = Client(calls(("c1", "get_weather", {"city": "P"})), answer("x"))
        loop = AgentLoop(client, tools=[get_weather])
        loop.complete("?")
        assert all(o["ctx"]["conversationId"] == loop.id for _, o in client.seen)

    def test_every_step_of_one_run_shares_a_request_id(self) -> None:
        # A conversation that arrives as several unrelated traces is a trace id
        # that failed at the one thing it is for.
        client = Client(calls(("c1", "get_weather", {"city": "P"})), answer("x"))
        AgentLoop(client, tools=[get_weather]).complete("?")
        ids = {o["ctx"]["requestId"] for _, o in client.seen}
        assert len(ids) == 1

    def test_two_runs_do_not_share_one(self) -> None:
        client = Client(answer("a"), answer("b"))
        loop = AgentLoop(client)
        loop.complete("one")
        loop.complete("two")
        assert client.seen[0][1]["ctx"]["requestId"] != client.seen[1][1]["ctx"]["requestId"]

    def test_the_run_hooks_fire_in_order(self) -> None:
        seen: list[str] = []
        loop = AgentLoop(Client(answer("x")))
        def record(name: str) -> Callable[[Any], None]:
            # A named closure rather than a lambda with a default argument:
            # the default is what binds `name` per iteration, and it is also
            # what makes the lambda's type impossible to infer.
            def handler(ctx: Any) -> None:
                seen.append(name)

            return handler

        for name in ("onRunStart", "onStepStart", "onStepComplete", "onRunComplete"):
            loop.hooks.on(name, record(name))
        loop.complete("?")
        assert seen == ["onRunStart", "onStepStart", "onStepComplete", "onRunComplete"]

    def test_a_run_that_raises_reports_the_error_and_re_raises(self) -> None:
        class Boom(Client):
            def complete(self, messages: Any, **options: Any) -> Completion:
                raise RuntimeError("provider down")

        loop = AgentLoop(Boom())
        with pytest.raises(RuntimeError, match="provider down"):
            loop.complete("?")
        # Recorded before re-raising, so a caller that catches can still read it.
        assert loop.last_report is not None
        assert loop.last_report.reason == "error"
        assert loop.last_report.error is not None

    def test_one_agent_refuses_two_concurrent_runs(self) -> None:
        loop = AgentLoop(Client(answer("x")))
        loop._running = True
        with pytest.raises(RuntimeError, match="already running"):
            loop.complete("?")


class TestSnapshot:
    def test_a_dump_restores_the_conversation(self) -> None:
        loop = AgentLoop(Client(answer("hello")), system="be terse")
        loop.complete("hi")
        snapshot = loop.dump()

        restored = AgentLoop(Client(answer("x")), history=snapshot["history"])
        assert restored.id == loop.id
        assert len(restored.history) == 2

    def test_tools_are_recorded_by_name_only(self) -> None:
        # A snapshot claiming to carry them would be one that cannot be
        # restored: the bodies are functions and the transport is a socket.
        loop = AgentLoop(Client(answer("x")), tools=[get_weather])
        assert loop.dump()["toolNames"] == ["get_weather"]
