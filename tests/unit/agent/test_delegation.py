"""`delegate`, `handoff`, `Observer` -- and whose work is whose.

What has to be right here is attribution. A specialist called as a tool runs
INSIDE its caller's run, on the same engine and usually the same model, so
anything that scopes by run -- or guesses by model -- files the specialist's
tokens under the caller. Nothing downstream can tell: every number stays
plausible, the per-agent cost is simply wrong, and the specialist looks free.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from combycode_llm_sdk import Observer, delegate, handoff
from combycode_llm_sdk.agent.lazy_tools import rank_tools, tokenize, unwrap_lazy_call
from combycode_llm_sdk.helpers.tool import Tool, tool
from combycode_llm_sdk.results import Completion, Usage


class FakeAgent:
    """An agent-shaped stand-in: an id, a bus, and a scripted answer."""

    def __init__(self, text: str, usage: Usage | None = None, id: str = "agent_1") -> None:
        from combycode_llm_sdk.bus.hook_bus import HookBus

        self.id = id
        self.hooks = HookBus()
        self.text = text
        self.usage = usage or Usage()
        self.asked: list[Any] = []

    def complete(self, task: Any = None, **options: Any) -> Completion:
        self.asked.append(task)
        return Completion(
            text=self.text, model="m", finish_reason="stop", usage=self.usage, parts=()
        )


class TestDelegate:
    def test_it_returns_the_sub_agents_own_answer(self) -> None:
        expert = FakeAgent("42")
        ask = delegate("ask_expert", "Ask the expert.", expert)
        assert ask(task="what is 6*7") == "42"

    def test_only_the_task_crosses(self) -> None:
        # The specialist starts from its OWN system prompt and an empty history.
        # That is the point of a specialist, and also the thing that surprises a
        # caller who expected the conversation to travel with the task.
        expert = FakeAgent("42")
        delegate("ask_expert", "Ask.", expert)(task="what is 6*7")
        assert expert.asked == ["what is 6*7"]

    def test_the_tool_is_named_by_the_caller_not_the_function(self) -> None:
        # A parent may hold one specialist under two names; deriving the name
        # from `__name__` would give both the same one.
        expert = FakeAgent("42")
        first = delegate("ask_maths", "Maths.", expert)
        second = delegate("ask_physics", "Physics.", expert)
        assert first.name == "ask_maths"
        assert second.name == "ask_physics"

    def test_it_declares_one_string_parameter(self) -> None:
        schema = delegate("ask", "Ask.", FakeAgent("x")).schema
        assert schema["parameters"]["properties"]["task"]["type"] == "string"
        assert schema["parameters"]["required"] == ["task"]


class TestHandoff:
    def test_it_says_which_specialist_answered(self) -> None:
        # `delegate` returns bare text, so a parent holding three specialists
        # cannot tell which one replied.
        expert = FakeAgent("42", Usage(output_tokens=11))
        answered = json.loads(handoff("ask_expert", "Ask.", expert)(task="q"))
        assert answered["text"] == "42"
        assert answered["agent_name"] == "ask_expert"

    def test_it_reports_what_the_sub_agents_run_cost(self) -> None:
        expert = FakeAgent("42", Usage(input_tokens=3, output_tokens=11))
        answered = json.loads(handoff("ask_expert", "Ask.", expert)(task="q"))
        assert answered["usage"]["output_tokens"] == 11

    def test_usage_is_none_when_the_provider_reported_none(self) -> None:
        # A handoff that substituted zeros would look exactly like a provider
        # that never sent them, and the specialist would appear free.
        answered = json.loads(handoff("ask", "Ask.", FakeAgent("42"))(task="q"))
        assert answered["usage"] is None

    def test_an_input_filter_rewrites_the_task(self) -> None:
        expert = FakeAgent("ok")
        handoff("ask", "Ask.", expert, input_filter=str.upper)(task="quiet")
        assert expert.asked == ["QUIET"]


class TestObserver:
    def test_it_sees_only_its_own_agents_events(self) -> None:
        # The assertion the whole design exists for.
        boss, expert = FakeAgent("b", id="boss"), FakeAgent("e", id="expert")
        for_boss: list[Any] = []
        with Observer(boss, "on_run_complete", for_boss.append):
            boss.hooks.emit_sync("onRunComplete", {"agentId": "boss"})
            boss.hooks.emit_sync("onRunComplete", {"agentId": "expert"})
        assert len(for_boss) == 1
        assert expert.id != boss.id

    def test_a_completion_is_matched_by_conversation_id(self) -> None:
        # `onCompletion` carries no agentId -- the LLM client emits it, and the
        # only thing tying it to an agent is the conversation id the loop stamps.
        agent = FakeAgent("x", id="agent_7")
        seen: list[Any] = []
        Observer(agent, "on_completion", seen.append)
        agent.hooks.emit_sync(
            "onCompletion",
            {
                "provider": "anthropic",
                "model": "m",
                "response": {"text": "hi", "usage": {"outputTokens": 4}},
                "ctx": {"conversationId": "agent_7"},
            },
        )
        agent.hooks.emit_sync(
            "onCompletion",
            {
                "provider": "anthropic",
                "model": "m",
                "response": {"text": "other"},
                "ctx": {"conversationId": "someone_else"},
            },
        )
        assert len(seen) == 1
        assert seen[0].usage.output_tokens == 4

    def test_a_completion_arrives_as_a_completion(self) -> None:
        # The same view `complete()` returns, so nobody learns a second shape.
        agent = FakeAgent("x", id="a1")
        seen: list[Any] = []
        Observer(agent, "on_completion", seen.append)
        agent.hooks.emit_sync(
            "onCompletion",
            {
                "provider": "p",
                "model": "m",
                # The wire shape a real client emits: text lives in `content`
                # parts, because that is where commentary and the answer are
                # told apart.
                "response": {"content": [{"type": "text", "text": "hi"}]},
                "ctx": {"conversationId": "a1"},
            },
        )
        assert isinstance(seen[0], Completion)
        assert seen[0].text == "hi"

    def test_an_event_with_no_identity_is_refused(self) -> None:
        # A watcher that reports another agent's work as this one's is worse
        # than one that reports nothing, because only the first is believed.
        agent = FakeAgent("x", id="a1")
        seen: list[Any] = []
        Observer(agent, "on_run_complete", seen.append)
        agent.hooks.emit_sync("onRunComplete", {})
        assert seen == []

    def test_leaving_the_block_unsubscribes(self) -> None:
        agent = FakeAgent("x", id="a1")
        seen: list[Any] = []
        with Observer(agent, "on_run_complete", seen.append):
            agent.hooks.emit_sync("onRunComplete", {"agentId": "a1"})
        agent.hooks.emit_sync("onRunComplete", {"agentId": "a1"})
        assert len(seen) == 1

    def test_entering_returns_the_agent(self) -> None:
        # So `with Observer(boss, ...) as watching: watching.run(...)` reads.
        agent = FakeAgent("x")
        with Observer(agent, "on_run_complete", lambda c: None) as bound:
            assert bound is agent

    def test_an_unknown_event_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match="not an observable event"):
            Observer(FakeAgent("x"), "on_banana", lambda c: None)


class TestLazyRanking:
    def test_stop_words_do_not_score(self) -> None:
        assert tokenize("what is the weather for a city") == ["weather", "city"]

    def test_the_name_bonus_breaks_a_tie(self) -> None:
        # Both score the same on raw overlap -- the query words appear once in
        # each. Only the double weight on the NAME separates them, so this is
        # the assertion that the weighting exists at all.
        @tool
        def weather(city: str) -> str:
            """Return a value."""
            return ""

        @tool
        def unrelated(city: str) -> str:
            """Look up the weather elsewhere."""
            return ""

        ranked = rank_tools("weather", [unrelated, weather], 5)
        assert [t.name for t in ranked] == ["weather", "unrelated"]

    def test_a_name_match_beats_a_description_match(self) -> None:
        @tool
        def rebuild_index(namespace: str) -> str:
            """Rebuild something."""
            return ""

        @tool
        def unrelated(x: str) -> str:
            """A tool whose description mentions the index rebuild process."""
            return ""

        ranked = rank_tools("rebuild index", [unrelated, rebuild_index], 5)
        assert ranked[0].name == "rebuild_index"

    def test_a_query_matching_nothing_returns_nothing(self) -> None:
        @tool
        def get_weather(city: str) -> str:
            """Get the weather."""
            return ""

        assert rank_tools("quantum chromodynamics", [get_weather], 5) == []

    def test_the_limit_is_honoured(self) -> None:
        tools = []
        for i in range(10):

            def body(query: str, _i: int = i) -> str:
                return ""

            body.__name__ = f"search_thing_{i}"
            body.__doc__ = "Search for a thing."
            tools.append(tool(body))
        assert len(rank_tools("search thing", tools, 3)) == 3

    def test_unwrap_names_the_inner_tool(self) -> None:
        assert unwrap_lazy_call("call_tool", {"name": "rebuild_index"}) == "rebuild_index"

    def test_unwrap_ignores_an_ordinary_call(self) -> None:
        assert unwrap_lazy_call("get_weather", {"city": "P"}) is None


class TestTheToolDecoratorLazyForm:
    def test_the_bare_form_is_not_lazy(self) -> None:
        @tool
        def f(x: str) -> str:
            """Do a thing."""
            return x

        assert f.lazy is False
        assert isinstance(f, Tool)

    def test_the_called_form_marks_it_lazy(self) -> None:
        @tool(lazy=True)
        def f(x: str) -> str:
            """Do a thing."""
            return x

        assert f.lazy is True
        assert f(x="ok") == "ok"

    def test_the_called_form_without_lazy_still_builds(self) -> None:
        @tool()
        def f(x: str) -> str:
            """Do a thing."""
            return x

        assert f.lazy is False
