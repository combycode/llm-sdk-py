"""`complete` / `acomplete`, and the attachment loading they do first.

The one-shot helper is the corpus's most-used entry point -- 20 of the 34
scenarios call it -- and almost all of what it does happens BEFORE the client:
resolving attachments, folding them into whatever shape the prompt is, reading a
`model:tier` suffix, and routing audio to an api that accepts it.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

import pytest

from combycode_llm_sdk import acomplete, complete
from combycode_llm_sdk.helpers.content import load_content
from combycode_llm_sdk.transport import TransportRequest, TransportResponse

OK_BODY = {
    "id": "msg_1",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "OK"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 1},
}

#: A one-pixel PNG, so the magic bytes are real rather than asserted.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class Stub:
    def __init__(self, body: Any = None) -> None:
        self.body = body if body is not None else OK_BODY
        self.seen: list[TransportRequest] = []

    def __call__(self, request: TransportRequest) -> TransportResponse:
        self.seen.append(request)
        return TransportResponse(status=200, body=self.body)


class AsyncStub(Stub):
    async def __call__(self, request: TransportRequest) -> TransportResponse:  # type: ignore[override]
        self.seen.append(request)
        return TransportResponse(status=200, body=self.body)


def one(stub: Stub, **over: Any) -> Any:
    kwargs: dict[str, Any] = {
        "model": "anthropic/claude-haiku-4.5",
        "api_key": "k",
        "transport": stub,
    }
    kwargs.update(over)
    return complete(**kwargs)


class TestTheInput:
    def test_a_string_prompt_becomes_a_user_turn(self) -> None:
        stub = Stub()
        assert one(stub, prompt="hi", max_tokens=8).text == "OK"
        assert stub.seen[0].body["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        ]

    def test_messages_is_read_as_well_as_prompt(self) -> None:
        # `03_multi_turn.py` uses `messages=`. Reading only `prompt` sent an
        # EMPTY text block, and Anthropic answers `text content blocks must be
        # non-empty` -- found by a live run.
        stub = Stub()
        one(
            stub,
            messages=[
                {"role": "user", "content": "My name is Ada."},
                {"role": "assistant", "content": "Nice to meet you."},
                {"role": "user", "content": "What is my name?"},
            ],
            max_tokens=8,
        )
        roles = [m["role"] for m in stub.seen[0].body["messages"]]
        assert roles == ["user", "assistant", "user"]

    def test_nothing_to_send_is_refused_before_the_wire(self) -> None:
        # Better than an empty text block and a 400 from the provider.
        with pytest.raises(TypeError, match="nothing to send"):
            one(Stub(), max_tokens=8)

    def test_a_system_prompt_is_lifted_out_of_the_messages(self) -> None:
        stub = Stub()
        one(stub, prompt="hi", system="be terse", max_tokens=8)
        assert stub.seen[0].body["system"] == "be terse"


class TestAttachments:
    def test_bytes_become_an_image_part_before_the_text(self) -> None:
        # Attachments go FIRST: models attend to an image better when the
        # instruction about it follows.
        stub = Stub()
        one(stub, prompt="what is this?", attachments=[PNG], max_tokens=8)
        content = stub.seen[0].body["messages"][0]["content"]
        assert [p["type"] for p in content] == ["image", "text"]
        assert content[0]["source"]["media_type"] == "image/png"

    def test_a_path_is_read_from_disk(self, tmp_path: Path) -> None:
        # `pathlib.Path` is what Python callers actually hold; requiring `str()`
        # at every call site is the reason the contract asks for this.
        target = tmp_path / "pixel.png"
        target.write_bytes(PNG)
        stub = Stub()
        one(stub, prompt="?", attachments=[target], max_tokens=8)
        content = stub.seen[0].body["messages"][0]["content"]
        assert content[0]["source"]["media_type"] == "image/png"

    def test_the_extension_decides_the_part_type(self, tmp_path: Path) -> None:
        # A pdf is a DOCUMENT part, not an image -- the part type follows the
        # MIME, which is why `load_content` exists beside `load_image_content`.
        target = tmp_path / "report.pdf"
        target.write_bytes(b"%PDF-1.4 fake")
        part = load_content(target)
        assert part["type"] == "document"
        assert part["source"]["mimeType"] == "application/pdf"

    def test_a_ready_made_part_passes_through(self) -> None:
        stub = Stub()
        given = {"type": "text", "text": "already a part"}
        one(stub, prompt="and this", attachments=[given], max_tokens=8)
        content = stub.seen[0].body["messages"][0]["content"]
        assert content[0]["text"] == "already a part"

    def test_attachments_join_the_first_user_message_of_a_transcript(self) -> None:
        stub = Stub()
        one(
            stub,
            messages=[
                {"role": "user", "content": "look"},
                {"role": "assistant", "content": "at what?"},
            ],
            attachments=[PNG],
            max_tokens=8,
        )
        first = stub.seen[0].body["messages"][0]["content"]
        assert [p["type"] for p in first] == ["image", "text"]


class TestRouting:
    def test_a_model_tier_suffix_is_read_and_stripped(self) -> None:
        # The unified tier name does NOT reach the wire: Anthropic's
        # `service_tier` is an allow/forbid-priority switch (`auto` /
        # `standard_only`), not a tier selector, and the chain spec maps it.
        # What this asserts is that the suffix was READ -- it left the model id
        # and produced a tier param that a plain call does not.
        stub = Stub()
        one(stub, model="anthropic/claude-haiku-4.5:priority", prompt="hi", max_tokens=8)
        assert ":" not in stub.seen[0].body["model"]
        assert stub.seen[0].body["service_tier"] == "auto"

        plain = Stub()
        one(plain, model="anthropic/claude-haiku-4.5", prompt="hi", max_tokens=8)
        assert "service_tier" not in plain.seen[0].body

    def test_an_explicit_service_tier_beats_the_suffix(self) -> None:
        # `standard` forbids priority, so it maps to `standard_only` -- a
        # different value from the suffix's, which is the point.
        stub = Stub()
        one(
            stub,
            model="anthropic/claude-haiku-4.5:priority",
            prompt="hi",
            service_tier="standard",
            max_tokens=8,
        )
        assert stub.seen[0].body["service_tier"] == "standard_only"

    def test_an_openrouter_suffix_that_is_not_a_tier_is_left_alone(self) -> None:
        # `:free` and `:online` are OpenRouter model VARIANTS, and stripping one
        # would ask for a model that does not exist.
        stub = Stub(
            body={
                "id": "c1",
                "model": "x",
                "choices": [
                    {"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )
        complete(
            model="openrouter/qwen/qwen3-coder:free",
            api_key="k",
            transport=stub,
            prompt="hi",
            max_tokens=8,
        )
        assert stub.seen[0].body["model"].endswith(":free")
        assert "service_tier" not in stub.seen[0].body

    def test_openai_audio_input_routes_to_chat_completions(self) -> None:
        # The Responses API -- OpenAI's default here -- rejects `input_audio`.
        stub = Stub(
            body={
                "id": "c1",
                "model": "gpt-5.4-nano",
                "choices": [
                    {"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )
        complete(
            model="openai/gpt-5.4-nano",
            api_key="k",
            transport=stub,
            prompt="what did they say?",
            attachments=[{"type": "audio", "source": {"type": "base64", "mimeType": "audio/wav", "data": "AA=="}}],
            max_tokens=8,
        )
        assert stub.seen[0].url.endswith("/v1/chat/completions")

    def test_a_pinned_api_is_not_overridden_by_the_audio_rule(self) -> None:
        stub = Stub(body={"id": "r", "status": "completed", "output": [], "usage": {}})
        complete(
            model="openai/gpt-5.4-nano",
            api_key="k",
            transport=stub,
            client={"api": "responses"},
            prompt="?",
            attachments=[{"type": "audio", "source": {"type": "base64", "mimeType": "audio/wav", "data": "AA=="}}],
            max_tokens=8,
        )
        assert stub.seen[0].url.endswith("/v1/responses")


class TestTheCostCeiling:
    """`max_cost_usd` used to be REFUSED here, because accepting an option and
    ignoring it is how a caller believes a budget is enforced when it is not.
    It is enforced now; the guard itself is covered in
    `test_facts_and_budget.py`, and what this pins is the entry point."""

    def test_an_affordable_call_goes_through(self) -> None:
        assert one(Stub(), prompt="hi", max_cost_usd=1.0).text == "OK"

    def test_an_unaffordable_call_never_reaches_the_transport(self) -> None:
        from combycode_llm_sdk.estimate import BudgetExceededError

        stub = Stub()
        with pytest.raises(BudgetExceededError):
            one(stub, prompt="hi", max_cost_usd=0.000_000_1)
        assert stub.seen == [], "the guard runs BEFORE the call, or it is a bill"


class TestStructured:
    def test_a_schema_gets_the_parsed_object(self) -> None:
        stub = Stub(
            body={
                **OK_BODY,
                "content": [{"type": "text", "text": '{"n": 7}'}],
            }
        )
        result = one(
            stub,
            prompt="give me json",
            structured={"schema": {"type": "object"}},
            max_tokens=16,
        )
        assert result.parsed == {"n": 7}
        assert result.text == '{"n": 7}'

    def test_without_a_schema_parsed_stays_none(self) -> None:
        assert one(Stub(), prompt="hi", max_tokens=8).parsed is None


class TestAsyncTwin:
    def test_acomplete_answers_the_same_as_complete(self) -> None:
        sync_stub, async_stub = Stub(), AsyncStub()
        got_sync = one(sync_stub, prompt="hi", max_tokens=8)
        got_async = asyncio.run(
            acomplete(
                model="anthropic/claude-haiku-4.5",
                api_key="k",
                transport=async_stub,
                prompt="hi",
                max_tokens=8,
            )
        )
        assert got_sync.text == got_async.text == "OK"
        assert sync_stub.seen[0].body == async_stub.seen[0].body

    def test_both_destroy_their_client_even_when_the_call_fails(self) -> None:
        # The helper exists so a caller who wanted one answer does not leak a
        # client; a failure is exactly when that is easiest to get wrong.
        from combycode_llm_sdk.bus.hook_bus import HookBus

        hooks = HookBus()
        destroyed: list[Any] = []
        hooks.on("onClientDestroy", lambda ctx: destroyed.append(ctx["provider"]))

        def failing(request: TransportRequest) -> TransportResponse:
            return TransportResponse(status=400, body={"error": {"message": "no"}})

        with pytest.raises(Exception, match="no"):
            complete(
                model="anthropic/claude-haiku-4.5",
                api_key="k",
                transport=failing,
                client={"hooks": hooks},
                prompt="hi",
                max_tokens=8,
            )
        assert destroyed == ["anthropic"]
