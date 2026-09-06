"""`LLM` / `AsyncLLM`: the vocabulary translation, and that both faces agree.

The cores are already tested (`test_client.py`, `test_client_parity.py`). What is
new here is everything the facade adds: snake_case keywords in, frozen
dataclasses out, and the handful of renames the API contract asks for.

The wire bodies below are the shapes the provider ACTUALLY sends, taken from
`tests/fixtures/response-golden.json` rather than invented -- notably `phase`,
which sits on the message ITEM, so a two-phase turn is two message items.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from combycode_llm_sdk.events import (
    CitationEvent,
    DoneEvent,
    TextEvent,
    UnknownEvent,
    UsageEvent,
    to_event,
)
from combycode_llm_sdk.helpers.llm import LLM, AsyncLLM, _options
from combycode_llm_sdk.results import Completion
from combycode_llm_sdk.transport import TransportRequest, TransportResponse

SIMPLE = {
    "id": "resp_1",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "OK"}],
        }
    ],
    "usage": {"input_tokens": 10, "output_tokens": 5},
}

TWO_PHASE = {
    "id": "resp_2",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "Let me think. "}],
        },
        {
            "type": "message",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": "42"}],
        },
    ],
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


class Stub:
    """A transport that records what it was asked and answers with a fixed body."""

    def __init__(self, body: Any = None, frames: list[bytes] | None = None) -> None:
        self.body = body if body is not None else SIMPLE
        self.frames = frames
        self.seen: list[TransportRequest] = []

    def __call__(self, request: TransportRequest) -> TransportResponse:
        self.seen.append(request)
        if request.response_type == "stream":
            return TransportResponse(status=200, body=iter(self.frames or []))
        return TransportResponse(status=200, body=self.body)


class AsyncStub(Stub):
    async def __call__(self, request: TransportRequest) -> TransportResponse:  # type: ignore[override]
        self.seen.append(request)
        if request.response_type == "stream":
            return TransportResponse(status=200, body=_aiter(self.frames or []))
        return TransportResponse(status=200, body=self.body)


async def _aiter(chunks: list[bytes]) -> Any:
    for chunk in chunks:
        yield chunk


def sse(*payloads: str) -> list[bytes]:
    return [f"data: {p}\n\n".encode() for p in payloads]


class TestOptionTranslation:
    def test_snake_case_keywords_become_wire_options(self) -> None:
        assert _options({"max_tokens": 16, "top_p": 0.5, "service_tier": "flex"}) == {
            "maxTokens": 16,
            "topP": 0.5,
            "serviceTier": "flex",
        }

    def test_reasoning_is_the_public_name_for_thinking(self) -> None:
        # The wire calls it `thinking`; the examples settled on `reasoning`, and
        # it is what the provider docs call the feature.
        assert _options({"reasoning": {"effort": "low"}}) == {"thinking": {"effort": "low"}}

    def test_state_is_the_public_name_for_previous_response_id(self) -> None:
        # A caller passes back what `Completion.state` gave them and never sees
        # the wire name.
        assert _options({"state": "resp_1"}) == {"previousResponseId": "resp_1"}

    @pytest.mark.parametrize(
        ("given", "expected"), [(True, "auto"), (False, "off"), ("auto", "auto")]
    )
    def test_cache_takes_a_bool_or_the_wire_form(self, given: Any, expected: str) -> None:
        assert _options({"cache": given})["cache"] == expected

    def test_builtin_tools_is_sugar_that_merges_with_function_tools(self) -> None:
        fn = {"name": "f", "description": "d", "parameters": {}}
        out = _options({"tools": [fn], "builtin_tools": ["web_search", "code_interpreter"]})
        assert out["tools"] == [fn, {"type": "web_search"}, {"type": "code_interpreter"}]

    def test_a_none_option_is_dropped_rather_than_sent(self) -> None:
        # Otherwise every unset keyword would reach the wire as an explicit null.
        assert _options({"max_tokens": None, "temperature": 0.1}) == {"temperature": 0.1}

    def test_an_unknown_keyword_is_refused_and_names_the_escape_hatch(self) -> None:
        # The one untyped hole in the TypeScript request was the one place a typo
        # produced silence rather than a failure.
        with pytest.raises(TypeError, match="max_token"):
            _options({"max_token": 16})
        with pytest.raises(TypeError, match="provider_options"):
            _options({"nonsense": 1})


class TestCompleteShape:
    def make(self, body: Any = None, **over: Any) -> tuple[LLM, Stub]:
        stub = Stub(body)
        return LLM(model="openai/gpt-5.4-nano", api_key="k", transport=stub, **over), stub

    def test_returns_a_frozen_completion_in_snake_case(self) -> None:
        llm, _ = self.make()
        result = llm.complete("hi")
        assert isinstance(result, Completion)
        assert result.text == "OK"
        assert result.finish_reason == "stop"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        with pytest.raises(AttributeError):
            result.text = "mutated"  # type: ignore[misc]

    def test_text_is_the_answer_alone_and_commentary_stays_reachable(self) -> None:
        llm, _ = self.make(TWO_PHASE)
        result = llm.complete("hi")
        assert result.text == "42"
        assert [(p.phase, p.text) for p in result.parts] == [
            ("commentary", "Let me think. "),
            ("final_answer", "42"),
        ]

    def test_state_carries_a_stateful_conversation_forward(self) -> None:
        llm, stub = self.make()
        first = llm.complete("hi")
        assert first.state == "resp_1"
        llm.complete("again", state=first.state)
        assert stub.seen[-1].body["previous_response_id"] == "resp_1"

    def test_state_is_none_on_a_stateless_api(self) -> None:
        # Anthropic's Messages API keeps no server-side conversation, and sending
        # it an id is a 400 -- so `state` must be None there however plainly the
        # response carries an `id` of its own.
        anthropic_body = {
            "id": "msg_1",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "OK"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        llm = LLM(
            model="anthropic/claude-haiku-4.5",
            api_key="k",
            transport=Stub(anthropic_body),
        )
        result = llm.complete("hi")
        assert result.text == "OK"
        assert result.state is None
        assert "serverStateId" not in result.assistant_message()["origin"]

    def test_assistant_message_is_ready_to_append(self) -> None:
        llm, _ = self.make()
        message = llm.complete("hi").assistant_message()
        assert message["role"] == "assistant"
        assert message["origin"]["provider"] == "openai"
        # A stateful api stamps the id so the next turn can continue server-side.
        assert message["origin"]["serverStateId"] == "resp_1"

    def test_options_reach_the_wire(self) -> None:
        llm, stub = self.make()
        llm.complete("hi", max_tokens=16, temperature=0.25)
        body = stub.seen[0].body
        assert body["max_output_tokens"] == 16
        assert body["temperature"] == 0.25

    def test_the_api_key_reaches_the_auth_header(self) -> None:
        llm, stub = self.make()
        llm.complete("hi")
        assert stub.seen[0].headers["authorization"] == "Bearer k"


class TestCost:
    def test_a_priced_model_reports_a_cost(self) -> None:
        llm = LLM(model="openai/gpt-5.4-nano", api_key="k", transport=Stub())
        cost = llm.complete("hi").cost
        assert cost is not None
        assert cost.total > 0
        assert cost.source == "calculated"

    def test_an_unpriced_model_reports_none_and_never_zero(self) -> None:
        # A silent zero is the bug the API contract names by hand: a client
        # reported 72k tokens billed at $0.00.
        llm = LLM(
            model="openai/a-model-nobody-has-priced", api_key="k", transport=Stub()
        )
        assert llm.complete("hi").cost is None


class TestStream:
    FRAMES = sse(
        '{"type":"response.output_text.delta","delta":"Hel","item_id":"i1"}',
        '{"type":"response.output_text.delta","delta":"lo","item_id":"i1"}',
        '{"type":"response.completed","response":{"status":"completed",'
        '"usage":{"input_tokens":1,"output_tokens":2}}}',
    )

    def test_events_are_dataclasses_with_a_type_discriminator(self) -> None:
        llm = LLM(
            model="openai/gpt-5.4-nano", api_key="k", transport=Stub(frames=self.FRAMES)
        )
        events = list(llm.stream("hi"))
        assert [e.type for e in events] == ["text", "text", "usage", "done"]
        assert "".join(e.text for e in events if isinstance(e, TextEvent)) == "Hello"
        assert isinstance(events[-1], DoneEvent)
        assert events[-1].finish_reason == "stop"

    def test_events_support_structural_pattern_matching(self) -> None:
        llm = LLM(
            model="openai/gpt-5.4-nano", api_key="k", transport=Stub(frames=self.FRAMES)
        )
        text = ""
        tokens = 0
        for event in llm.stream("hi"):
            match event:
                case TextEvent(text=chunk):
                    text += chunk
                case UsageEvent(usage=usage):
                    tokens = usage.output_tokens
        assert text == "Hello"
        assert tokens == 2

    def test_an_unmodelled_event_kind_still_reaches_the_caller(self) -> None:
        # The unified event set is open; a provider adding a kind must not make
        # the reply silently lose a piece.
        event = to_event({"type": "quantum_entanglement_delta", "payload": 1})
        assert isinstance(event, UnknownEvent)
        assert event.type == "quantum_entanglement_delta"
        assert event.payload["payload"] == 1

    def test_a_citation_event_carries_the_dataclass(self) -> None:
        event = to_event({"type": "citation", "citation": {"url": "https://a", "title": "A"}})
        assert isinstance(event, CitationEvent)
        assert event.citation.url == "https://a"
        assert event.citation.title == "A"


class TestConstruction:
    def test_a_namespaced_model_supplies_the_provider(self) -> None:
        llm = LLM(model="anthropic/claude-haiku-4.5", api_key="k", transport=Stub())
        assert llm.provider == "anthropic"

    def test_an_explicit_provider_wins_over_the_prefix(self) -> None:
        # Every OpenRouter id is `vendor/model`, so the prefix losing here is how
        # an OpenRouter key once went to api.openai.com.
        llm = LLM(
            model="openai/gpt-5.4-nano",
            provider="openrouter",
            api_key="k",
            transport=Stub(),
        )
        assert llm.provider == "openrouter"

    def test_a_bare_model_without_a_provider_is_refused(self) -> None:
        with pytest.raises(ValueError, match="requires a provider"):
            LLM(model="gpt-5.4-nano", api_key="k", transport=Stub())

    def test_a_missing_key_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match="no API key"):
            LLM(model="openai/gpt-5.4-nano", transport=Stub())

    def test_an_engine_supplies_the_key_hooks_and_catalog(self) -> None:
        from combycode_llm_sdk.helpers.engine import Engine

        stub = Stub()
        engine = Engine(
            transport=stub,
            api_keys={"openai": "from-engine"},
            register_as_default=False,
        )
        seen: list[Any] = []
        engine.on_completion(lambda ctx: seen.append(ctx["provider"]))

        # No api_key= here: the engine's is used, and so is its hook bus.
        llm = LLM(model="openai/gpt-5.4-nano", engine=engine)
        llm.complete("hi")

        assert stub.seen[0].headers["authorization"] == "Bearer from-engine"
        assert seen == ["openai"]
        assert llm.client.catalog is engine.catalog

    def test_an_explicit_key_still_beats_the_engines(self) -> None:
        from combycode_llm_sdk.helpers.engine import Engine

        stub = Stub()
        engine = Engine(
            transport=stub, api_keys={"openai": "engine"}, register_as_default=False
        )
        LLM(model="openai/gpt-5.4-nano", api_key="explicit", engine=engine).complete("hi")
        assert stub.seen[0].headers["authorization"] == "Bearer explicit"

    def test_an_explicit_transport_beats_an_ambient_default_engine(self) -> None:
        # The default is a FALLBACK. A caller who named a transport has said
        # where this client's requests go, and a global registered elsewhere
        # winning over that is how a stub silently starts making real calls.
        from combycode_llm_sdk.helpers.engine import Engine

        Engine(transport=Stub(), api_keys={"openai": "engine"})  # registers itself
        mine = Stub()
        LLM(model="openai/gpt-5.4-nano", api_key="k", transport=mine).complete("hi")
        assert len(mine.seen) == 1

    def test_a_missing_key_names_the_engine_as_a_place_to_put_one(self) -> None:
        from combycode_llm_sdk.helpers.engine import Engine

        engine = Engine(transport=Stub(), register_as_default=False)
        with pytest.raises(ValueError, match="engine.api_keys"):
            LLM(model="openai/gpt-5.4-nano", engine=engine)

    def test_the_api_choice_selects_the_adapter(self) -> None:
        stub = Stub(body={"candidates": [], "usageMetadata": {}})
        llm = LLM(
            model="google/gemini-2.5-flash", api_key="k", api="interactions", transport=stub
        )
        llm.complete("hi")
        assert "/v1beta/interactions" in stub.seen[0].url


class TestTheTwoFacesAgree:
    """`LLM` and `AsyncLLM` are faces on two cores; they must still look alike."""

    FRAMES = TestStream.FRAMES

    def test_complete_returns_the_same_completion(self) -> None:
        sync = LLM(model="openai/gpt-5.4-nano", api_key="k", transport=Stub(TWO_PHASE))
        a = AsyncLLM(model="openai/gpt-5.4-nano", api_key="k", transport=AsyncStub(TWO_PHASE))
        got_sync = sync.complete("hi")
        got_async = asyncio.run(a.complete("hi"))
        assert _comparable(got_sync) == _comparable(got_async)

    def test_stream_yields_the_same_events(self) -> None:
        sync = LLM(
            model="openai/gpt-5.4-nano", api_key="k", transport=Stub(frames=self.FRAMES)
        )
        a = AsyncLLM(
            model="openai/gpt-5.4-nano", api_key="k", transport=AsyncStub(frames=self.FRAMES)
        )

        async def collect() -> list[Any]:
            return [e async for e in a.stream("hi")]

        assert list(sync.stream("hi")) == asyncio.run(collect())

    def test_both_send_the_same_request(self) -> None:
        sync_stub, async_stub = Stub(), AsyncStub()
        LLM(model="openai/gpt-5.4-nano", api_key="k", transport=sync_stub).complete(
            "hi", max_tokens=8
        )
        asyncio.run(
            AsyncLLM(
                model="openai/gpt-5.4-nano", api_key="k", transport=async_stub
            ).complete("hi", max_tokens=8)
        )
        assert sync_stub.seen[0].body == async_stub.seen[0].body
        assert sync_stub.seen[0].url == async_stub.seen[0].url


def _comparable(completion: Completion) -> Any:
    """Everything but the per-run clock."""
    return (
        completion.text,
        completion.model,
        completion.finish_reason,
        completion.usage,
        completion.parts,
        completion.state,
        completion.cost,
    )
