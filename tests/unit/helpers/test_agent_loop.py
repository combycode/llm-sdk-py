"""The loop `complete(tools=...)` owns -- scenarios 06 through 09.

Driven through a fake `call_model` rather than a stubbed transport: the loop's
job is the TURN STRUCTURE it builds, and asserting on messages is the only way to
see it. What reaches the wire from those messages is the adapters' job, and they
have their own recorded tests.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from combycode_llm_sdk import tool
from combycode_llm_sdk.helpers.agent_loop import ToolLoopLimit, arun_tools, run_tools
from combycode_llm_sdk.results import Completion, Part, Usage


def completion(**over: Any) -> Completion:
    """A Completion with the fields these tests do not care about filled in."""
    fields: dict[str, Any] = {
        "text": "",
        "model": "m",
        "finish_reason": "stop",
        "usage": Usage(),
        "parts": [],
    }
    fields.update(over)
    return Completion(**fields)


def answer(text: str) -> Completion:
    return completion(text=text, parts=[Part(type="text", text=text)])


def calls(*specs: tuple[str, str, dict[str, Any]]) -> Completion:
    """A turn that asks for tools. `(id, name, arguments)` each."""
    parts = [
        Part(type="tool_call", id=i, name=n, arguments=a, raw={"type": "tool_call"})
        for i, n, a in specs
    ]
    return completion(parts=list(parts), tool_calls=list(parts))


class Model:
    """Answers with a scripted sequence, recording what it was sent."""

    def __init__(self, *turns: Completion) -> None:
        self.turns = list(turns)
        self.seen: list[list[dict[str, Any]]] = []
        self.definitions: list[list[dict[str, Any]]] = []

    def __call__(self, messages: list[dict[str, Any]], definitions: Any) -> Completion:
        self.seen.append([dict(m) for m in messages])
        self.definitions.append(definitions)
        return self.turns[len(self.seen) - 1]


class AsyncModel(Model):
    async def __call__(  # type: ignore[override]
        self, messages: list[dict[str, Any]], definitions: Any
    ) -> Completion:
        return Model.__call__(self, messages, definitions)


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"sunny in {city}"


@tool
def get_population(city: str) -> str:
    """Get the population of a city."""
    return f"2.1 million in {city}"


@tool
def get_user_city() -> str:
    """Get the user's current city."""
    return "Paris"


class TestOneCall:
    def test_the_result_goes_back_and_the_second_answer_is_returned(self) -> None:
        model = Model(calls(("c1", "get_weather", {"city": "Paris"})), answer("It is sunny."))
        got = run_tools(model, "What is the weather in Paris?", [get_weather])
        assert got.text == "It is sunny."

        # Second request: the original turn, the assistant's call, the result.
        second = model.seen[1]
        assert [m["role"] for m in second] == ["user", "assistant", "tool"]
        assert second[1]["content"][0]["name"] == "get_weather"
        assert second[2]["content"][0] == {
            "type": "tool_result",
            "id": "c1",
            "content": "sunny in Paris",
        }

    def test_the_definitions_are_sent_every_turn(self) -> None:
        # Not only the first: a provider builds each request from scratch, and a
        # turn without the declarations is a turn where the model cannot call.
        model = Model(calls(("c1", "get_weather", {"city": "Paris"})), answer("done"))
        run_tools(model, "?", [get_weather])
        assert all(d[0]["name"] == "get_weather" for d in model.definitions)

    def test_a_turn_with_no_calls_returns_immediately(self) -> None:
        model = Model(answer("Paris is the capital."))
        assert run_tools(model, "capital of France?", [get_weather]).text == (
            "Paris is the capital."
        )
        assert len(model.seen) == 1


class TestParallelCalls:
    def test_every_call_in_the_round_is_answered(self) -> None:
        # `07`. Answering only the first left the rest unanswered, and OpenAI
        # rejects the whole next request with "No tool output found".
        model = Model(
            calls(
                ("c1", "get_weather", {"city": "Paris"}),
                ("c2", "get_population", {"city": "Paris"}),
            ),
            answer("Sunny, 2.1 million."),
        )
        run_tools(model, "weather and population?", [get_weather, get_population])
        results = model.seen[1][2]["content"]
        assert [r["id"] for r in results] == ["c1", "c2"]
        assert results[1]["content"] == "2.1 million in Paris"

    def test_all_results_ride_in_one_tool_message(self) -> None:
        model = Model(
            calls(("c1", "get_weather", {"city": "P"}), ("c2", "get_population", {"city": "P"})),
            answer("ok"),
        )
        run_tools(model, "?", [get_weather, get_population])
        assert [m["role"] for m in model.seen[1]] == ["user", "assistant", "tool"]

    def test_results_keep_the_order_the_model_asked_in(self) -> None:
        # They are matched by id, so order is not correctness -- but a transcript
        # that reorders them is a transcript nobody can read against the calls.
        model = Model(
            calls(("c1", "get_population", {"city": "P"}), ("c2", "get_weather", {"city": "P"})),
            answer("ok"),
        )
        run_tools(model, "?", [get_weather, get_population])
        assert [r["id"] for r in model.seen[1][2]["content"]] == ["c1", "c2"]


class TestDependentCalls:
    def test_a_second_round_sees_the_first_round_s_result(self) -> None:
        # `08`. The model learns the city, then asks for its weather.
        model = Model(
            calls(("c1", "get_user_city", {})),
            calls(("c2", "get_weather", {"city": "Paris"})),
            answer("Sunny in Paris."),
        )
        got = run_tools(model, "weather where I am?", [get_user_city, get_weather])
        assert got.text == "Sunny in Paris."
        assert [m["role"] for m in model.seen[2]] == [
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
        ]


class TestAsyncTools:
    def test_a_sync_loop_runs_an_async_tool(self) -> None:
        # `09`. A sync `complete()` may take async tools; the loop it owns is
        # what makes that true.
        @tool
        async def fetch_price(symbol: str) -> str:
            """Look up a price."""
            await asyncio.sleep(0)
            return f"{symbol} is 100 USD"

        model = Model(calls(("c1", "fetch_price", {"symbol": "ACME"})), answer("100 USD."))
        run_tools(model, "price of ACME?", [fetch_price])
        assert model.seen[1][2]["content"][0]["content"] == "ACME is 100 USD"

    def test_the_async_loop_runs_an_async_tool(self) -> None:
        # The combination `acomplete(tools=...)` actually hits, and the only path
        # where `Tool.is_async` decides anything: the sync loop asks
        # `inspect.isawaitable` of the RETURN value instead, so a mis-detected
        # tool still works there and this is where it would not.
        @tool
        async def fetch_price(symbol: str) -> str:
            """Look up a price."""
            await asyncio.sleep(0)
            return f"{symbol} is 100 USD"

        model = AsyncModel(calls(("c1", "fetch_price", {"symbol": "ACME"})), answer("100 USD."))
        got = asyncio.run(arun_tools(model, "?", [fetch_price]))
        assert got.text == "100 USD."
        assert model.seen[1][2]["content"][0]["content"] == "ACME is 100 USD"

    def test_an_async_tool_does_not_go_to_a_worker_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What `Tool.is_async` actually decides -- and all it decides.

        Both branches produce the right ANSWER and even run the body on the same
        thread: `to_thread` would only CREATE the coroutine on a worker, and a
        coroutine body always runs on the event loop that awaits it. So the flag
        is unobservable except here, in whether a worker is spawned at all --
        which is the reason the branch exists, and without this assertion
        deleting the flag changes no test.
        """
        spawned: list[str] = []
        real = asyncio.to_thread

        async def counting(fn: Any, *args: Any, **kwargs: Any) -> Any:
            spawned.append(getattr(fn, "__name__", "?"))
            return await real(fn, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", counting)

        @tool
        async def fetch_price(symbol: str) -> str:
            """Look up a price."""
            return f"{symbol} is 100 USD"

        model = AsyncModel(calls(("c1", "fetch_price", {"symbol": "ACME"})), answer("ok"))
        asyncio.run(arun_tools(model, "?", [fetch_price]))
        assert spawned == []

        # ...while a SYNC tool does go to one, or it would block the loop and,
        # with it, every other tool in the same round.
        spawned.clear()
        sync_model = AsyncModel(calls(("c2", "get_weather", {"city": "P"})), answer("ok"))
        asyncio.run(arun_tools(sync_model, "?", [get_weather]))
        assert len(spawned) == 1

    def test_the_async_loop_runs_a_sync_tool(self) -> None:
        model = AsyncModel(calls(("c1", "get_weather", {"city": "Paris"})), answer("Sunny."))
        got = asyncio.run(arun_tools(model, "?", [get_weather]))
        assert got.text == "Sunny."
        assert model.seen[1][2]["content"][0]["content"] == "sunny in Paris"

    def test_both_loops_build_the_same_messages(self) -> None:
        script = (calls(("c1", "get_weather", {"city": "Paris"})), answer("Sunny."))
        sync_model, async_model = Model(*script), AsyncModel(*script)
        run_tools(sync_model, "?", [get_weather])
        asyncio.run(arun_tools(async_model, "?", [get_weather]))
        assert sync_model.seen == async_model.seen


class TestWhenThingsGoWrong:
    def test_a_raising_tool_reports_back_rather_than_out(self) -> None:
        # A tool that fails is a fact the model can act on. Raising would throw
        # away a conversation that was one turn from an answer.
        @tool
        def explode(city: str) -> str:
            """Always fails."""
            raise RuntimeError("no such city")

        model = Model(calls(("c1", "explode", {"city": "Atlantis"})), answer("I could not."))
        got = run_tools(model, "?", [explode])
        result = model.seen[1][2]["content"][0]
        assert result["isError"] is True
        assert "no such city" in result["content"]
        assert got.text == "I could not."

    def test_an_unknown_tool_name_is_reported_with_the_real_ones(self) -> None:
        model = Model(calls(("c1", "get_traffic", {})), answer("I cannot."))
        run_tools(model, "?", [get_weather])
        result = model.seen[1][2]["content"][0]
        assert result["isError"] is True
        assert "get_weather" in result["content"]

    def test_a_call_with_no_arguments_is_dispatched(self) -> None:
        # `08`'s `get_user_city()`. The absent `arguments` arrives as None, and
        # `**None` is a TypeError that would read as a library bug.
        model = Model(
            completion(
                tool_calls=[Part(type="tool_call", id="c1", name="get_user_city")]
            ),
            answer("Paris."),
        )
        run_tools(model, "?", [get_user_city])
        assert model.seen[1][2]["content"][0]["content"] == "Paris"

    def test_two_tools_with_one_name_is_refused(self) -> None:
        @tool
        def get_weather(city: str) -> str:
            """A second tool with the same name."""
            return "different"

        with pytest.raises(ValueError, match="two tools are named"):
            run_tools(Model(answer("x")), "?", [get_weather, get_weather.__wrapped__])

    def test_the_step_ceiling_ends_a_model_that_never_stops(self) -> None:
        # An unbounded spend becomes an error a caller can read.
        looping = [calls(("c1", "get_weather", {"city": "P"})) for _ in range(10)]
        with pytest.raises(ToolLoopLimit, match="after 3 turns"):
            run_tools(Model(*looping), "?", [get_weather], max_steps=3)

    def test_a_non_string_return_becomes_json_not_a_python_repr(self) -> None:
        # `str({'a': 1})` is `{'a': 1}` -- single quotes, not JSON, and models
        # do misread it.
        @tool
        def lookup(city: str) -> dict[str, int]:
            """Return a record."""
            return {"temp": 20}

        model = Model(calls(("c1", "lookup", {"city": "P"})), answer("ok"))
        run_tools(model, "?", [lookup])
        assert model.seen[1][2]["content"][0]["content"] == '{"temp": 20}'


class TestTheAssistantTurn:
    def test_narration_travels_with_the_calls(self) -> None:
        # A model that explains its plan before calling loses the explanation if
        # only the calls are echoed back.
        turn = calls(("c1", "get_weather", {"city": "Paris"}))
        narrated = completion(
            text="Let me check.",
            parts=[Part(type="text", text="Let me check."), *turn.parts],
            tool_calls=list(turn.tool_calls),
        )
        model = Model(narrated, answer("Sunny."))
        run_tools(model, "?", [get_weather])
        content = model.seen[1][1]["content"]
        assert content[0] == {"type": "text", "text": "Let me check."}
        assert content[1]["type"] == "tool_call"

    def test_provider_metadata_on_a_call_is_carried_back(self) -> None:
        # Google's thought signatures live in `_meta`, and a turn that drops them
        # is rejected on the next request.
        part = Part(
            type="tool_call",
            id="c1",
            name="get_weather",
            arguments={"city": "P"},
            raw={"type": "tool_call", "_meta": {"thoughtSignature": "sig"}},
        )
        model = Model(completion(parts=[part], tool_calls=[part]), answer("ok"))
        run_tools(model, "?", [get_weather])
        assert model.seen[1][1]["content"][0]["_meta"] == {"thoughtSignature": "sig"}
