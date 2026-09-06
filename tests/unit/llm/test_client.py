"""Translated from `unified-library-ts/tests/unit/llm/client.test.ts`.

`AsyncLLMClient` with a mock provider adapter and a stub fetch. The sync twin
is not re-tested case for case here: `test_client_parity.py` runs both cores
over identical inputs and asserts identical output, which is a stronger claim
than two copies of these assertions would be -- a copy can be edited to agree
with a divergence, a differential cannot.

AsyncLLMClient with a mock provider adapter and a stub fetch. Validates input
normalization, hook emission, model+system fixed at construction, fetch
injection, ctx propagation, and a custom cacheKeyFn.

The stub adapter is what makes these tests about the CLIENT: a real adapter
would put spec resolution and wire building between the assertion and the thing
being asserted, and the spec differentials already cover that half.

Two translations are not one-for-one, and both are the sync/async split:

- `onBeforeSubmit` interception is `ctx.resultPromise` in TypeScript, where every
  value is awaitable. Here the key is `result` and it takes a value OR an
  awaitable, so a synchronous cache hit does not have to fabricate a coroutine.
  Both forms are tested.
- `retrieve_file` / `stream_file` return bytes and an async iterator rather than
  a `Blob` and a `ReadableStream`, which have no Python equivalent. The
  assertions on WHICH request goes out are unchanged.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import pytest

from combycode_llm_sdk.bus.hook_bus import HookBus
from combycode_llm_sdk.llm.async_client import AsyncLLMClient
from combycode_llm_sdk.llm.output_errors import InvalidFinalOutputError
from combycode_llm_sdk.network.errors import RequestAborted

SCHEMA = {"type": "object", "properties": {"n": {"type": "number"}}}


class MockAdapter:
    """A ProviderAdapter that records what it was handed and answers trivially."""

    name = "mock"

    def __init__(self) -> None:
        self.last_request: dict[str, Any] | None = None

    def build_request(self, req: dict[str, Any]) -> Any:
        self.last_request = req
        return _Built(
            {"model": req.get("model"), "messages": req["messages"], "system": req.get("system")}
        )

    def parse_response(self, raw: Any, latency_ms: float) -> dict[str, Any]:
        text = (raw or {}).get("text") or ""
        return {
            "id": "r1",
            "model": "mock-model",
            "content": [{"type": "text", "text": text}],
            "finishReason": "stop",
            "usage": {
                "inputTokens": 1,
                "outputTokens": 1,
                "totalTokens": 2,
                "cachedTokens": 0,
                "cacheWriteTokens": 0,
                "reasoningTokens": 0,
            },
            "text": text,
            "toolCalls": [],
            "thinking": None,
            "media": [],
            "latencyMs": latency_ms,
            "raw": raw,
        }

    def parse_stream_event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        return []

    def create_stream_parser(self) -> Any:
        return lambda event: []

    def auth_headers(self) -> dict[str, str]:
        return {"x-api-key": "mock-key"}

    def base_url(self) -> str:
        return "https://mock.test"

    def completion_path(self) -> str:
        return "/v1/complete"


class _Built:
    """The `BuiltRequest` fields the client reads, without the wire interpreter."""

    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body
        self.headers: dict[str, str] | None = None
        self.path: str | None = None
        self.notes: list[str] | None = None


class StubFetch:
    def __init__(self, body: Any) -> None:
        self._body = body
        self.calls: list[tuple[dict[str, Any], Any]] = []

    async def __call__(self, req: dict[str, Any], opts: Any = None) -> dict[str, Any]:
        self.calls.append((req, opts))
        return {"status": 200, "headers": {}, "body": self._body}


class QueuedFetch:
    """Returns a queue of bodies, one per call; the last one repeats."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.count = 0

    async def __call__(self, req: dict[str, Any], opts: Any = None) -> dict[str, Any]:
        text = self._texts[min(self.count, len(self._texts) - 1)]
        self.count += 1
        return {"status": 200, "headers": {}, "body": {"text": text}}


def make_client(**over: Any) -> AsyncLLMClient:
    config: dict[str, Any] = {
        "provider": "anthropic",
        "model": "claude-3",
        "apiKey": "sk-test",
        "adapter": MockAdapter(),
        "fetch": StubFetch({"text": "hi"}),
    }
    config.update(over)
    return AsyncLLMClient(config)


class TestConstruction:
    def test_raises_on_missing_required_fields(self) -> None:
        with pytest.raises(ValueError):
            AsyncLLMClient({})

    def test_emits_on_client_create(self) -> None:
        hooks = HookBus()
        seen: dict[str, Any] = {}
        hooks.on("onClientCreate", lambda ctx: seen.update(ctx))
        client = make_client(hooks=hooks)
        assert seen["clientId"] == client.id
        assert seen["provider"] == "anthropic"
        assert seen["model"] == "claude-3"

    def test_exposes_id_provider_model_system(self) -> None:
        client = make_client(system="be helpful")
        assert client.provider == "anthropic"
        assert client.model == "claude-3"
        assert client.system == "be helpful"
        assert re.fullmatch(r"[0-9a-f-]+", client.id)

    def test_emits_on_client_destroy(self) -> None:
        hooks = HookBus()
        destroyed: list[bool] = []
        hooks.on("onClientDestroy", lambda ctx: destroyed.append(True))
        make_client(hooks=hooks).destroy()
        assert destroyed == [True]


class TestInputNormalization:
    async def test_a_string_wraps_as_a_user_message(self) -> None:
        adapter = MockAdapter()
        client = make_client(adapter=adapter)
        await client.complete("hello")
        assert adapter.last_request is not None
        assert adapter.last_request["messages"] == [{"role": "user", "content": "hello"}]

    async def test_content_parts_wrap_as_a_user_message_with_parts(self) -> None:
        adapter = MockAdapter()
        client = make_client(adapter=adapter)
        await client.complete([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
        assert adapter.last_request is not None
        messages = adapter.last_request["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert isinstance(messages[0]["content"], list)

    async def test_a_message_list_is_used_directly(self) -> None:
        adapter = MockAdapter()
        client = make_client(adapter=adapter)
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        await client.complete(msgs)
        assert adapter.last_request is not None
        assert adapter.last_request["messages"] == msgs


class TestSystemAndModelFixedAtConstruction:
    async def test_system_reaches_the_adapter_on_every_call(self) -> None:
        adapter = MockAdapter()
        client = make_client(adapter=adapter, system="fixed system prompt")
        await client.complete("a")
        assert adapter.last_request is not None
        assert adapter.last_request["system"] == "fixed system prompt"
        await client.complete("b")
        assert adapter.last_request["system"] == "fixed system prompt"

    async def test_model_reaches_the_adapter_on_every_call(self) -> None:
        adapter = MockAdapter()
        client = make_client(adapter=adapter, model="fixed-model")
        await client.complete("a")
        assert adapter.last_request is not None
        assert adapter.last_request["model"] == "fixed-model"


class TestHooksPipeline:
    async def test_emits_resolve_submit_completion_in_order(self) -> None:
        hooks = HookBus()
        order: list[str] = []
        hooks.on("onMessageResolve", lambda ctx: order.append("resolve"))
        hooks.on("onBeforeSubmit", lambda ctx: order.append("submit"))
        hooks.on("onCompletion", lambda ctx: order.append("completion"))
        await make_client(hooks=hooks).complete("hello")
        assert order == ["resolve", "submit", "completion"]

    async def test_threads_session_id_and_mints_request_and_call_ids(self) -> None:
        hooks = HookBus()
        seen: list[dict[str, Any]] = []
        hooks.on("onCompletion", lambda c: seen.append(c["ctx"]))
        await make_client(hooks=hooks, sessionId="sess_test").complete("hello")
        ctx = seen[0]
        assert ctx["sessionId"] == "sess_test"
        assert ctx["requestId"].startswith("req_")
        assert ctx["callId"].startswith("call_")

    def test_a_standalone_client_mints_its_own_session_id(self) -> None:
        assert make_client(hooks=HookBus()).session_id.startswith("sess_")

    async def test_on_message_resolve_can_mutate_messages_in_place(self) -> None:
        adapter = MockAdapter()
        hooks = HookBus()
        hooks.on(
            "onMessageResolve",
            lambda ctx: ctx["messages"].append({"role": "system", "content": "INJECTED"}),
        )
        await make_client(adapter=adapter, hooks=hooks).complete("hello")
        assert adapter.last_request is not None
        # The injected message is a SYSTEM one and system extraction has already
        # run, so it reaches the adapter as a message -- which is what the
        # TypeScript asserts too. A handler that injects after extraction is
        # injecting a message, not a system prompt.
        assert any(m.get("content") == "INJECTED" for m in adapter.last_request["messages"])

    async def test_on_message_resolve_can_replace_the_list_outright(self) -> None:
        # In-place mutation is visible whether or not the client re-reads the
        # context, because it is the same list object. REPLACEMENT is not: a
        # handler that assigns a new list (a context guard handing back a
        # compacted transcript, which is the real case) is silently ignored
        # unless the normalized request is re-anchored to what the handler left.
        adapter = MockAdapter()
        hooks = HookBus()

        def replace(ctx: dict[str, Any]) -> None:
            ctx["messages"] = [{"role": "user", "content": "COMPACTED"}]
            ctx["system"] = "REWRITTEN"

        hooks.on("onMessageResolve", replace)
        await make_client(adapter=adapter, hooks=hooks).complete("hello")
        assert adapter.last_request is not None
        assert adapter.last_request["messages"] == [{"role": "user", "content": "COMPACTED"}]
        assert adapter.last_request["system"] == "REWRITTEN"

    async def test_on_message_resolve_abort_stops_the_request(self) -> None:
        hooks = HookBus()

        def abort(ctx: dict[str, Any]) -> None:
            ctx["abort"] = True
            ctx["abortReason"] = "too long"

        hooks.on("onMessageResolve", abort)
        fetch = StubFetch({"text": "hi"})
        client = make_client(hooks=hooks, fetch=fetch)
        with pytest.raises(RequestAborted, match="too long"):
            await client.complete("hello")
        assert fetch.calls == []

    async def test_on_before_submit_interception_short_circuits_http(self) -> None:
        hooks = HookBus()

        def intercept(ctx: dict[str, Any]) -> None:
            ctx["intercepted"] = True
            ctx["result"] = {"text": "cached!"}

        hooks.on("onBeforeSubmit", intercept)
        fetch = StubFetch({"text": "hi"})
        res = await make_client(hooks=hooks, fetch=fetch).complete("hello")
        assert res["text"] == "cached!"
        assert fetch.calls == []

    async def test_an_intercepted_result_may_also_be_awaitable(self) -> None:
        """The TypeScript hands back a Promise and has no other option; this port
        accepts either, so an async cache is not forced to be synchronous."""
        hooks = HookBus()

        async def cached() -> dict[str, Any]:
            return {"text": "async cached!"}

        def intercept(ctx: dict[str, Any]) -> None:
            ctx["intercepted"] = True
            ctx["result"] = cached()

        hooks.on("onBeforeSubmit", intercept)
        fetch = StubFetch({"text": "hi"})
        res = await make_client(hooks=hooks, fetch=fetch).complete("hello")
        assert res["text"] == "async cached!"
        assert fetch.calls == []


class TestBuildNotes:
    """What the request builder left out on purpose reaches the caller.

    Dropping a capability the caller asked for and saying nothing is how a
    missing feature gets mistaken for a working one -- so every note the spec
    attached becomes an `onWarning`. Untested until a parity mutation showed the
    warning could be deleted with every test still green: a differential cannot
    see a change both cores share, and this is shared code.
    """

    async def test_every_build_note_becomes_a_warning(self) -> None:
        adapter = MockAdapter()
        notes = ["hosted tool dropped beside an attachment", "top_k not supported here"]

        original = adapter.build_request

        def with_notes(req: dict[str, Any]) -> Any:
            built = original(req)
            built.notes = notes
            return built

        adapter.build_request = with_notes  # type: ignore[method-assign]

        hooks = HookBus()
        seen: list[dict[str, Any]] = []
        hooks.on("onWarning", lambda ctx: seen.append(ctx))
        await make_client(adapter=adapter, hooks=hooks).complete("hi")

        assert [w["message"] for w in seen] == notes
        assert {w["code"] for w in seen} == {"request_adjusted"}
        assert {w["source"] for w in seen} == {"llm"}
        assert seen[0]["details"]["provider"] == "anthropic"

    async def test_a_hosted_tool_the_provider_cannot_run_is_reported(self) -> None:
        """The silent loss this mechanic exists to prevent.

        Found by the 2026-09-04 sweep: `builtin_tools=["code_interpreter"]`
        against OpenRouter put `tools: []` on the wire and said nothing. The
        catalog is right -- OpenRouter runs no hosted code execution -- so
        dropping it is correct; staying quiet about it is not.
        """
        hooks = HookBus()
        seen: list[dict[str, Any]] = []
        hooks.on("onWarning", lambda ctx: seen.append(dict(ctx)))
        client = make_client(provider="openrouter", model="openai/gpt-5.4-nano", hooks=hooks)

        result = await client.complete(
            "hi", {"tools": [{"type": "code_interpreter"}, {"type": "web_search"}]}
        )

        messages = [w["message"] for w in seen if w["code"] == "request_adjusted"]
        assert len(messages) == 1, "web_search IS supported there and must not be reported"
        assert "code_interpreter" in messages[0]
        assert "openrouter" in messages[0]
        # A caller holding only the result must not have had to subscribe. This
        # is the low-level client, so the result is still the wire dict.
        assert [w["code"] for w in result["warnings"]] == ["request_adjusted"]

    async def test_a_supported_hosted_tool_is_not_reported(self) -> None:
        hooks = HookBus()
        seen: list[dict[str, Any]] = []
        hooks.on("onWarning", lambda ctx: seen.append(dict(ctx)))
        client = make_client(provider="anthropic", model="claude-haiku-4.5", hooks=hooks)

        await client.complete("hi", {"tools": [{"type": "code_interpreter"}]})
        assert seen == []

    async def test_a_function_tool_is_never_reported_as_a_hosted_one(self) -> None:
        # Function tools are the caller's own code and no provider "supports"
        # them from a list; reporting one would be noise on every tool call.
        hooks = HookBus()
        seen: list[dict[str, Any]] = []
        hooks.on("onWarning", lambda ctx: seen.append(dict(ctx)))
        client = make_client(provider="openrouter", model="openai/gpt-5.4-nano", hooks=hooks)

        await client.complete(
            "hi",
            {"tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]},
        )
        assert seen == []

    async def test_a_model_the_catalog_does_not_carry_is_left_alone(self) -> None:
        # An unknown model is not evidence that a tool is unsupported, and
        # warning on one would fire for every custom deployment.
        hooks = HookBus()
        seen: list[dict[str, Any]] = []
        hooks.on("onWarning", lambda ctx: seen.append(dict(ctx)))
        client = make_client(provider="openrouter", model="some/unlisted-model", hooks=hooks)

        await client.complete("hi", {"tools": [{"type": "code_interpreter"}]})
        assert seen == []

    async def test_a_build_with_no_notes_warns_about_nothing(self) -> None:
        hooks = HookBus()
        seen: list[Any] = []
        hooks.on("onWarning", lambda ctx: seen.append(ctx))
        await make_client(hooks=hooks).complete("hi")
        assert seen == []


class TestRequestContextAndRouting:
    async def test_queue_name_defaults_to_provider_slash_model(self) -> None:
        fetch = StubFetch({"text": "hi"})
        await make_client(fetch=fetch).complete("hello")
        assert fetch.calls[0][1]["queueName"] == "anthropic/claude-3"

    async def test_queue_name_from_config_overrides_the_default(self) -> None:
        fetch = StubFetch({"text": "hi"})
        await make_client(fetch=fetch, queueName="shared/cheap").complete("hello")
        assert fetch.calls[0][1]["queueName"] == "shared/cheap"

    async def test_cache_key_fn_computes_the_cache_key(self) -> None:
        hooks = HookBus()
        seen: list[Any] = []
        hooks.on("onBeforeSubmit", lambda ctx: seen.append(ctx["ctx"].get("cacheKey")))
        client = make_client(
            hooks=hooks, cacheKeyFn=lambda req, ctx: f"custom:{len(req['messages'])}"
        )
        await client.complete("hello")
        assert seen[0] == "custom:1"

    async def test_an_explicit_ctx_cache_key_beats_the_fn(self) -> None:
        hooks = HookBus()
        seen: list[Any] = []
        hooks.on("onBeforeSubmit", lambda ctx: seen.append(ctx["ctx"].get("cacheKey")))
        client = make_client(hooks=hooks, cacheKeyFn=lambda req, ctx: "from-fn")
        await client.complete("hello", {"ctx": {"cacheKey": "override"}})
        assert seen[0] == "override"

    async def test_call_id_is_minted_per_call(self) -> None:
        hooks = HookBus()
        seen: list[str] = []
        hooks.on("onBeforeSubmit", lambda ctx: seen.append(ctx["ctx"]["callId"]))
        client = make_client(hooks=hooks)
        await client.complete("a")
        await client.complete("b")
        assert len(seen) == 2
        assert seen[0] != seen[1]


class TestFetchOptionsPropagation:
    async def test_priority_defaults_to_interactive_in_foreground(self) -> None:
        fetch = StubFetch({"text": "hi"})
        await make_client(fetch=fetch).complete("hello")
        assert fetch.calls[0][1]["priority"] == 1

    async def test_priority_shifts_to_background_in_background_mode(self) -> None:
        fetch = StubFetch({"text": "hi"})
        await make_client(fetch=fetch, mode="background").complete("hello")
        assert fetch.calls[0][1]["priority"] == 2


class TestAdapterUrlComposition:
    async def test_full_url_is_base_url_plus_completion_path(self) -> None:
        fetch = StubFetch({"text": "hi"})
        await make_client(fetch=fetch).complete("hello")
        assert fetch.calls[0][0]["url"] == "https://mock.test/v1/complete"

    async def test_adapter_auth_headers_are_merged_in(self) -> None:
        fetch = StubFetch({"text": "hi"})
        await make_client(fetch=fetch).complete("hello")
        assert fetch.calls[0][0]["headers"]["x-api-key"] == "mock-key"


class TestStreamWithoutFetchStream:
    async def test_raises_if_stream_is_called_without_fetch_stream(self) -> None:
        client = make_client()
        with pytest.raises(RuntimeError, match="no fetchStream function configured"):
            await client.stream("hi").__anext__()


class TestStructuredComplete:
    async def test_parses_valid_json_output(self) -> None:
        client = make_client(fetch=QueuedFetch(['{"n":1}']))
        assert await client.structured_complete("go", SCHEMA) == {"n": 1}

    async def test_raises_invalid_final_output_with_raw_text(self) -> None:
        client = make_client(fetch=QueuedFetch(["not json"]))
        with pytest.raises(InvalidFinalOutputError) as info:
            await client.structured_complete("go", SCHEMA)
        assert info.value.reason == "invalid_final_output"
        assert info.value.raw_text == "not json"

    async def test_repair_attempts_retry_and_succeed(self) -> None:
        fetch = QueuedFetch(["oops", '{"n":7}'])
        client = make_client(fetch=fetch)
        out = await client.structured_complete(
            "go", SCHEMA, {"structured": {"schema": SCHEMA, "repairAttempts": 1}}
        )
        assert out == {"n": 7}
        assert fetch.count == 2  # original + 1 repair

    async def test_raises_after_repairs_are_exhausted(self) -> None:
        fetch = QueuedFetch(["bad", "still bad", "nope"])
        client = make_client(fetch=fetch)
        with pytest.raises(InvalidFinalOutputError):
            await client.structured_complete(
                "go", SCHEMA, {"structured": {"schema": SCHEMA, "repairAttempts": 1}}
            )
        assert fetch.count == 2  # original + 1 repair, then gives up


def _response(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "resp_abc",
        "model": "claude-3",
        "content": [{"type": "text", "text": "answer"}],
        "finishReason": "stop",
        "usage": {
            "inputTokens": 1,
            "outputTokens": 1,
            "totalTokens": 2,
            "cachedTokens": 0,
            "cacheWriteTokens": 0,
            "reasoningTokens": 0,
        },
        "text": "answer",
        "toolCalls": [],
        "thinking": None,
        "media": [],
        "latencyMs": 1,
        "raw": None,
    }
    base.update(over)
    return base


class TestAssistantMessageProvenance:
    def test_stamps_role_content_id_and_origin(self) -> None:
        m = make_client().assistant_message(_response())
        assert m["role"] == "assistant"
        assert m["content"] == [{"type": "text", "text": "answer"}]
        assert m["id"] == "resp_abc"
        assert m["origin"]["provider"] == "anthropic"
        assert m["origin"]["model"] == "claude-3"
        assert m["createdAt"] > 0

    def test_a_stateless_api_carries_no_server_state_id(self) -> None:
        # api defaults to `messages` for anthropic; resending an id there is a 400.
        m = make_client().assistant_message(_response())
        assert "serverStateId" not in m["origin"]

    def test_a_stateful_api_carries_the_response_id(self) -> None:
        client = make_client(provider="openai", api="responses")
        m = client.assistant_message(_response())
        assert m["origin"]["serverStateId"] == "resp_abc"

    def test_a_stateful_response_with_no_id_gets_a_generated_message_id(self) -> None:
        client = make_client(provider="openai", api="responses")
        m = client.assistant_message(_response(id=""))
        assert re.fullmatch(r"[0-9a-f-]{36}", m["id"])
        assert "serverStateId" not in m["origin"]


class TestFileRetrievalDelegation:
    async def test_retrieve_file_uses_this_client_provider_key_and_base_url(self) -> None:
        seen: list[dict[str, Any]] = []

        async def fetch(req: dict[str, Any], opts: Any = None) -> dict[str, Any]:
            seen.append(req)
            return {"status": 200, "headers": {"content-type": "image/png"}, "body": b"\x01\x02"}

        file = await make_client(fetch=fetch).retrieve_file({"id": "file_1"})
        assert seen[0]["headers"]["x-api-key"] == "sk-test"
        assert "/v1/files/file_1/content" in seen[0]["url"]
        assert file["mimeType"] == "image/png"
        assert file["size"] == 2

    async def test_stream_file_uses_the_same_context(self) -> None:
        seen: list[dict[str, Any]] = []

        async def chunks() -> AsyncIterator[bytes]:
            yield b"\x01\x02\x03"

        async def fetch(req: dict[str, Any], opts: Any = None) -> dict[str, Any]:
            seen.append(req)
            return {
                "status": 200,
                "headers": {"content-type": "text/csv", "content-length": "3"},
                "body": chunks(),
            }

        s = await make_client(fetch=fetch).stream_file({"id": "file_2"})
        assert seen[0]["headers"]["x-api-key"] == "sk-test"
        assert seen[0]["responseType"] == "stream"
        assert s["mimeType"] == "text/csv"
        assert s["size"] == 3
        assert [chunk async for chunk in s["stream"]] == [b"\x01\x02\x03"]


class TestSystemExtractionFromContentParts:
    async def test_a_system_message_as_content_parts_is_flattened(self) -> None:
        adapter = MockAdapter()
        client = make_client(adapter=adapter, fetch=StubFetch({"text": "ok"}))
        await client.complete(
            [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "line one"},
                        {"type": "text", "text": "line two"},
                    ],
                },
                {"role": "user", "content": "hi"},
            ]
        )
        assert adapter.last_request is not None
        assert adapter.last_request["system"] == "line one\nline two"
        # The system message must not ALSO reach the adapter as a message.
        assert not any(m["role"] == "system" for m in adapter.last_request["messages"])

    async def test_a_system_message_with_no_text_contributes_nothing(self) -> None:
        adapter = MockAdapter()
        client = make_client(adapter=adapter, fetch=StubFetch({"text": "ok"}))
        await client.complete(
            [
                {
                    "role": "system",
                    "content": [{"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}],
                },
                {"role": "user", "content": "hi"},
            ]
        )
        assert adapter.last_request is not None
        assert adapter.last_request["system"] is None


class TestAdapterIsMandatory:
    def test_neither_adapter_nor_fetch_is_caught_at_construction(self) -> None:
        with pytest.raises(ValueError, match=r"adapter \(or factory\) is required"):
            AsyncLLMClient({"provider": "anthropic", "model": "claude-3", "apiKey": "k"})

    def test_a_fetch_without_an_adapter_is_caught_by_the_resolver(self) -> None:
        # The constructor guard only fires when BOTH are absent, so this pair
        # reaches resolve_adapter -- which has to say a factory is acceptable
        # too, or the caller reads the message as "adapters are required".
        with pytest.raises(ValueError, match="adapter or AdapterFactory must be supplied"):
            AsyncLLMClient(
                {
                    "provider": "anthropic",
                    "model": "claude-3",
                    "apiKey": "k",
                    "fetch": StubFetch({}),
                }
            )

    def test_an_adapter_factory_is_called_with_provider_key_api_and_base_url(self) -> None:
        seen: list[tuple[Any, ...]] = []

        def factory(*args: Any) -> MockAdapter:
            seen.append(args)
            return MockAdapter()

        AsyncLLMClient(
            {
                "provider": "anthropic",
                "model": "claude-3",
                "apiKey": "k",
                "baseURL": "https://custom",
                "adapter": factory,
                "fetch": StubFetch({}),
            }
        )
        assert seen[0] == ("anthropic", "k", "messages", "https://custom")


class StreamingAdapter(MockAdapter):
    """A mock whose stream parser replays a scripted list of unified events.

    The scripting is deliberate: what is under test is what the CLIENT does with
    events, and using a real provider parser here would make a failure ambiguous
    between the two.
    """

    def __init__(self, events: list[dict[str, Any]]) -> None:
        super().__init__()
        self._events = events
        self.streaming_enabled = False

    def enable_streaming(self, provider_req: Any, req: dict[str, Any]) -> None:
        self.streaming_enabled = True
        provider_req.body["stream"] = True

    def create_stream_parser(self) -> Any:
        # One event per SSE frame, in order -- the frames carry the index.
        def parse(frame: dict[str, Any]) -> list[dict[str, Any]]:
            return [self._events[int(frame["data"])]]

        return parse


class StubFetchStream:
    """Yields one frame per scripted event, so the parser is driven the real way."""

    def __init__(self, count: int) -> None:
        self._count = count
        self.calls: list[tuple[dict[str, Any], Any]] = []

    def __call__(self, req: dict[str, Any], opts: Any = None) -> AsyncIterator[dict[str, Any]]:
        self.calls.append((req, opts))

        async def frames() -> AsyncIterator[dict[str, Any]]:
            for i in range(self._count):
                yield {"data": str(i)}

        return frames()


def make_streaming_client(events: list[dict[str, Any]], **over: Any) -> tuple[AsyncLLMClient, Any]:
    adapter = StreamingAdapter(events)
    fetch_stream = StubFetchStream(len(events))
    config: dict[str, Any] = {
        "provider": "anthropic",
        "model": "claude-3",
        "apiKey": "sk-test",
        "adapter": adapter,
        "fetch": StubFetch({}),
        "fetchStream": fetch_stream,
    }
    config.update(over)
    return AsyncLLMClient(config), (adapter, fetch_stream)


class TestStreamAccumulation:
    """`stream()` yields events AND adds them up, so the final `onCompletion`
    describes the same call `complete()` would have."""

    EVENTS: ClassVar[list[dict[str, Any]]] = [
        {"type": "text", "text": "Hel"},
        {"type": "thinking", "text": "hmm"},
        {"type": "text", "text": "lo"},
        {"type": "citation", "citation": {"url": "https://a", "title": "A"}},
        {"type": "citation", "citation": {"url": "https://a", "title": "A"}},
        {"type": "builtin_tool_start", "tool": "web_search"},
        {"type": "builtin_tool_end", "tool": "web_search", "query": "q", "id": "b1"},
        {"type": "file", "file": {"id": "f1", "source": "code_execution"}},
        {
            "type": "usage",
            "usage": {
                "inputTokens": 5,
                "outputTokens": 2,
                "totalTokens": 7,
                "cachedTokens": 0,
                "cacheWriteTokens": 0,
                "reasoningTokens": 0,
            },
        },
        {"type": "done", "finishReason": "stop"},
    ]

    async def test_every_event_reaches_the_caller_in_order(self) -> None:
        client, _ = make_streaming_client(self.EVENTS)
        got = [ev async for ev in client.stream("hi")]
        assert got == self.EVENTS

    async def test_the_final_completion_hook_describes_the_whole_turn(self) -> None:
        hooks = HookBus()
        seen: list[dict[str, Any]] = []
        hooks.on("onCompletion", lambda c: seen.append(c))
        client, _ = make_streaming_client(self.EVENTS, hooks=hooks)

        async for _ in client.stream("hi"):
            pass

        assert len(seen) == 1
        response = seen[0]["response"]
        assert response["text"] == "Hello"
        assert response["content"] == [{"type": "text", "text": "Hello"}]
        assert response["thinking"] == "hmm"
        assert response["finishReason"] == "stop"
        assert response["usage"]["totalTokens"] == 7
        # Deduped by url: a model that cites one page twice is still one source.
        assert response["citations"] == [{"url": "https://a", "title": "A"}]
        # Collected on END, which is the event carrying the payload.
        assert response["builtinToolCalls"] == [{"tool": "web_search", "id": "b1", "query": "q"}]
        assert response["files"] == [{"id": "f1", "source": "code_execution"}]
        assert response["id"].startswith("stream_")
        assert response["latencyMs"] >= 0

    async def test_the_optional_fields_stay_absent_when_nothing_produced_them(self) -> None:
        # R3: a response type grows by OPTIONAL fields only, so an empty turn
        # must not report `citations: []` where `complete()` reports nothing.
        hooks = HookBus()
        seen: list[dict[str, Any]] = []
        hooks.on("onCompletion", lambda c: seen.append(c))
        client, _ = make_streaming_client([{"type": "text", "text": "x"}], hooks=hooks)

        async for _ in client.stream("hi"):
            pass

        response = seen[0]["response"]
        for absent in ("citations", "files", "builtinToolCalls", "moderation"):
            assert absent not in response

    async def test_streaming_is_enabled_on_the_provider_request(self) -> None:
        client, (adapter, fetch_stream) = make_streaming_client(self.EVENTS)
        async for _ in client.stream("hi"):
            pass
        assert adapter.streaming_enabled is True
        assert fetch_stream.calls[0][0]["stream"] is True
        assert fetch_stream.calls[0][0]["body"]["stream"] is True

    async def test_an_abort_from_a_resolve_handler_emits_no_completion(self) -> None:
        # A cost is a COMPLETED call: a stream that never ran must not be priced.
        hooks = HookBus()
        completions: list[Any] = []
        hooks.on("onCompletion", lambda c: completions.append(c))

        def abort(ctx: dict[str, Any]) -> None:
            ctx["abort"] = True

        hooks.on("onMessageResolve", abort)
        client, (_, fetch_stream) = make_streaming_client(self.EVENTS, hooks=hooks)
        with pytest.raises(RequestAborted, match="Stream aborted"):
            async for _ in client.stream("hi"):
                pass
        assert completions == []
        assert fetch_stream.calls == []

    async def test_each_stream_gets_its_own_parser(self) -> None:
        # One parser per stream, so per-stream state (the Google code-execution
        # latch, the Anthropic pending tool input) cannot leak between concurrent
        # calls. Two streams from ONE client must not share.
        made: list[Any] = []

        class CountingAdapter(StreamingAdapter):
            def create_stream_parser(self) -> Any:
                parser = super().create_stream_parser()
                made.append(parser)
                return parser

        client = AsyncLLMClient(
            {
                "provider": "anthropic",
                "model": "claude-3",
                "apiKey": "k",
                "adapter": CountingAdapter(self.EVENTS),
                "fetch": StubFetch({}),
                "fetchStream": StubFetchStream(len(self.EVENTS)),
            }
        )
        async for _ in client.stream("a"):
            pass
        async for _ in client.stream("b"):
            pass
        assert len(made) == 2
        assert made[0] is not made[1]
