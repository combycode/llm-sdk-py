"""The two cores produce the same thing, or one of them is wrong.

`LLMClient` and `AsyncLLMClient` are two implementations of one behaviour. That
is the arrangement `PORTING.md` warns about -- "transpose, do not reimplement" --
and the first Python attempt failed because a second implementation drifted from
the first with nothing comparing them.

So they are compared. Every scenario below is defined ONCE and run through both
cores over identical inputs, asserting that the completion, the events, the hook
payloads and the outgoing HTTP requests all match. A divergence in either core
fails here, and it cannot be fixed by editing one side to agree -- the oracle is
the other implementation, not a recorded expectation someone can adjust.

What this deliberately does NOT compare is timing or concurrency: the async core
runs the two emulated-moderation calls together and the sync core runs them in
sequence. That is the one intended difference, and it changes no output.

**A differential cannot catch a bug in code both cores share.** Break
`client_base.py` and both sides break identically, so this file stays green -- a
mutation sweep confirmed exactly that. The shared half is covered by
`test_client.py`, which asserts against expectations rather than against the
other core. The two files are complements, and neither is sufficient alone.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

import pytest

from combycode_llm_sdk.bus.hook_bus import HookBus
from combycode_llm_sdk.llm.async_client import AsyncLLMClient
from combycode_llm_sdk.llm.client import LLMClient
from combycode_llm_sdk.llm.output_errors import InvalidFinalOutputError


class MockAdapter:
    """Records what it was handed; answers from the body it is given."""

    name = "mock"

    def __init__(self, notes: list[str] | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self._notes = notes

    def build_request(self, req: dict[str, Any]) -> Any:
        self.requests.append(
            {"model": req.get("model"), "system": req.get("system"), "messages": req["messages"]}
        )
        return _Built(
            {"model": req.get("model"), "messages": req["messages"], "system": req.get("system")},
            self._notes,
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
            # latencyMs is wall-clock and cannot match between two runs, so it is
            # pinned here rather than excluded from the comparison -- excluding a
            # field is how a field stops being compared for every reason.
            "latencyMs": 0,
            "raw": raw,
        }

    def parse_stream_event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        return []

    def create_stream_parser(self) -> Any:
        return lambda frame: []

    def auth_headers(self) -> dict[str, str]:
        return {"x-api-key": "mock-key"}

    def base_url(self) -> str:
        return "https://mock.test"

    def completion_path(self) -> str:
        return "/v1/complete"


class StreamingAdapter(MockAdapter):
    def __init__(self, events: list[dict[str, Any]]) -> None:
        super().__init__()
        self._events = events

    def enable_streaming(self, provider_req: Any, req: dict[str, Any]) -> None:
        provider_req.body["stream"] = True

    def create_stream_parser(self) -> Any:
        return lambda frame: [self._events[int(frame["data"])]]


class _Built:
    def __init__(self, body: dict[str, Any], notes: list[str] | None = None) -> None:
        self.body = body
        self.headers: dict[str, str] | None = None
        self.path: str | None = None
        self.notes = notes


class Recorder:
    """Collects everything observable, so parity is asserted on the whole surface."""

    def __init__(self) -> None:
        self.hooks = HookBus()
        self.events: list[tuple[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        for name in ("onMessageResolve", "onBeforeSubmit", "onCompletion", "onWarning"):
            self.hooks.on(name, self._make(name))

    def _make(self, name: str) -> Any:
        def record(ctx: dict[str, Any]) -> None:
            self.events.append((name, _scrub(ctx)))

        return record

    def snapshot(self) -> dict[str, Any]:
        return {"hooks": self.events, "requests": self.requests}


def _scrub(value: Any) -> Any:
    """Drop the fields that are per-run by nature, keep everything else.

    Only ids and clocks: a uuid and a millisecond reading differ between two runs
    of the SAME core, so comparing them would fail for a reason that is not
    divergence. Every other field is compared.
    """
    volatile = {"clientId", "callId", "requestId", "sessionId", "latencyMs", "id", "createdAt"}
    # `Mapping`, not `dict`: a hook payload arrives as a `HookContext` view, and
    # a dict-only test would skip it entirely -- leaving the per-run ids in and
    # failing for the one reason this function exists to rule out.
    if isinstance(value, Mapping):
        return {k: ("<volatile>" if k in volatile else _scrub(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


# -- the two runners ---------------------------------------------------------

BODIES = {"text": "hello world"}


def run_sync(scenario: Any, *, config: dict[str, Any] | None = None) -> Any:
    rec = Recorder()
    adapter = scenario.adapter()

    def fetch(req: dict[str, Any], opts: Any = None) -> dict[str, Any]:
        rec.requests.append(_scrub(req))
        return {"status": 200, "headers": {}, "body": scenario.body(len(rec.requests) - 1)}

    def fetch_stream(req: dict[str, Any], opts: Any = None) -> Iterator[dict[str, Any]]:
        rec.requests.append(_scrub(req))
        return iter([{"data": str(i)} for i in range(scenario.frames)])

    client = LLMClient(
        {
            "provider": "anthropic",
            "model": "claude-3",
            "apiKey": "k",
            "adapter": adapter,
            "hooks": rec.hooks,
            "fetch": fetch,
            "fetchStream": fetch_stream,
            **(config or {}),
        }
    )
    return scenario.sync(client), rec.snapshot()


def run_async(scenario: Any, *, config: dict[str, Any] | None = None) -> Any:
    rec = Recorder()
    adapter = scenario.adapter()

    async def fetch(req: dict[str, Any], opts: Any = None) -> dict[str, Any]:
        rec.requests.append(_scrub(req))
        return {"status": 200, "headers": {}, "body": scenario.body(len(rec.requests) - 1)}

    def fetch_stream(req: dict[str, Any], opts: Any = None) -> Any:
        rec.requests.append(_scrub(req))

        async def frames() -> Any:
            for i in range(scenario.frames):
                yield {"data": str(i)}

        return frames()

    client = AsyncLLMClient(
        {
            "provider": "anthropic",
            "model": "claude-3",
            "apiKey": "k",
            "adapter": adapter,
            "hooks": rec.hooks,
            "fetch": fetch,
            "fetchStream": fetch_stream,
            **(config or {}),
        }
    )
    return asyncio.run(scenario.run_async(client)), rec.snapshot()


class Scenario:
    """One behaviour, expressed once per core."""

    frames = 0

    def adapter(self) -> MockAdapter:
        return MockAdapter()

    def body(self, call: int) -> Any:
        return BODIES

    def sync(self, client: LLMClient) -> Any:
        raise NotImplementedError

    async def run_async(self, client: AsyncLLMClient) -> Any:
        raise NotImplementedError


class Complete(Scenario):
    def sync(self, client: LLMClient) -> Any:
        return client.complete("hello")

    async def run_async(self, client: AsyncLLMClient) -> Any:
        return await client.complete("hello")


class CompleteWithSystemAndOptions(Scenario):
    INPUT: ClassVar[list[dict[str, Any]]] = [
        {"role": "system", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        {"role": "user", "content": "hi"},
    ]
    OPTIONS: ClassVar[dict[str, Any]] = {"system": "per call", "maxTokens": 32, "temperature": 0.5, "tools": []}

    def sync(self, client: LLMClient) -> Any:
        return client.complete(self.INPUT, self.OPTIONS)

    async def run_async(self, client: AsyncLLMClient) -> Any:
        return await client.complete(self.INPUT, self.OPTIONS)


class CompleteWithBuildNotes(Scenario):
    def adapter(self) -> MockAdapter:
        return MockAdapter(notes=["dropped a hosted tool", "and another"])

    def sync(self, client: LLMClient) -> Any:
        return client.complete("hello")

    async def run_async(self, client: AsyncLLMClient) -> Any:
        return await client.complete("hello")


class StructuredWithRepair(Scenario):
    SCHEMA: ClassVar[dict[str, Any]] = {"type": "object", "properties": {"n": {"type": "number"}}}

    def body(self, call: int) -> Any:
        return {"text": "not json" if call == 0 else '{"n": 7}'}

    def sync(self, client: LLMClient) -> Any:
        return client.structured_complete(
            "go", self.SCHEMA, {"structured": {"schema": self.SCHEMA, "repairAttempts": 1}}
        )

    async def run_async(self, client: AsyncLLMClient) -> Any:
        return await client.structured_complete(
            "go", self.SCHEMA, {"structured": {"schema": self.SCHEMA, "repairAttempts": 1}}
        )


class StructuredThatGivesUp(Scenario):
    SCHEMA: ClassVar[dict[str, Any]] = {"type": "object"}

    def body(self, call: int) -> Any:
        return {"text": "never json"}

    def sync(self, client: LLMClient) -> Any:
        try:
            client.structured_complete("go", self.SCHEMA)
        except InvalidFinalOutputError as err:
            return ("raised", err.reason, err.raw_text)
        return ("no raise",)

    async def run_async(self, client: AsyncLLMClient) -> Any:
        try:
            await client.structured_complete("go", self.SCHEMA)
        except InvalidFinalOutputError as err:
            return ("raised", err.reason, err.raw_text)
        return ("no raise",)


class Stream(Scenario):
    EVENTS: ClassVar[list[dict[str, Any]]] = [
        {"type": "text", "text": "Hel"},
        {"type": "thinking", "text": "hmm"},
        {"type": "text", "text": "lo"},
        {"type": "citation", "citation": {"url": "https://a", "title": "A"}},
        {"type": "citation", "citation": {"url": "https://a", "title": "A"}},
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
    frames = len(EVENTS)

    def adapter(self) -> MockAdapter:
        return StreamingAdapter(self.EVENTS)

    def sync(self, client: LLMClient) -> Any:
        return list(client.stream("hi"))

    async def run_async(self, client: AsyncLLMClient) -> Any:
        return [ev async for ev in client.stream("hi")]


class StreamThatYieldsNothing(Scenario):
    """An empty stream still has to emit `onCompletion` with an empty response --
    the case where the two accumulators could most easily disagree."""

    EVENTS: ClassVar[list[dict[str, Any]]] = []
    frames = 0

    def adapter(self) -> MockAdapter:
        return StreamingAdapter(self.EVENTS)

    def sync(self, client: LLMClient) -> Any:
        return list(client.stream("hi"))

    async def run_async(self, client: AsyncLLMClient) -> Any:
        return [ev async for ev in client.stream("hi")]


class ResolveHandlerReplacesTheTranscript(Scenario):
    """A context guard handing back a compacted transcript.

    In-place mutation would be visible whether or not a core re-reads the
    resolve context, because it is the same list. REPLACEMENT is only visible if
    it does -- so this is the scenario that holds both cores to it.
    """

    def sync(self, client: LLMClient) -> Any:
        _arm(client)
        return client.complete("hello")

    async def run_async(self, client: AsyncLLMClient) -> Any:
        _arm(client)
        return await client.complete("hello")


def _arm(client: Any) -> None:
    def replace(ctx: dict[str, Any]) -> None:
        ctx["messages"] = [{"role": "user", "content": "COMPACTED"}]
        ctx["system"] = "REWRITTEN"

    client.hooks.on("onMessageResolve", replace)


class AssistantMessage(Scenario):
    def sync(self, client: LLMClient) -> Any:
        return _scrub(client.assistant_message(client.complete("hi")))

    async def run_async(self, client: AsyncLLMClient) -> Any:
        return _scrub(client.assistant_message(await client.complete("hi")))


SCENARIOS = {
    "complete": Complete(),
    "complete_system_and_options": CompleteWithSystemAndOptions(),
    "complete_build_notes": CompleteWithBuildNotes(),
    "structured_repair": StructuredWithRepair(),
    "structured_gives_up": StructuredThatGivesUp(),
    "stream": Stream(),
    "stream_empty": StreamThatYieldsNothing(),
    "assistant_message": AssistantMessage(),
    "resolve_replaces_transcript": ResolveHandlerReplacesTheTranscript(),
}


def test_it_is_actually_checking_something() -> None:
    # A scenario map that silently emptied would make every case below vacuous.
    assert len(SCENARIOS) >= 9


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_the_two_cores_agree(name: str) -> None:
    scenario = SCENARIOS[name]
    sync_result, sync_seen = run_sync(scenario)
    async_result, async_seen = run_async(scenario)

    assert _scrub(sync_result) == _scrub(async_result), f"{name}: results differ"
    assert sync_seen["requests"] == async_seen["requests"], f"{name}: outgoing requests differ"
    assert sync_seen["hooks"] == async_seen["hooks"], f"{name}: hook payloads differ"


def test_the_shared_base_never_waits() -> None:
    """The rule that keeps the fork small enough to trust.

    If a step that waits creeps into the base, one core inherits an `await` it
    cannot honour -- and the parity test above would still pass, because both
    cores would call the same broken thing.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[3]
        / "src/combycode_llm_sdk/llm/client_base.py"
    ).read_text(encoding="utf-8")
    code = [
        line
        for line in source.splitlines()
        # Skip the module docstring's own discussion of async, which is prose.
        if not line.lstrip().startswith(("#", '"', ">", "*"))
    ]
    offenders = [line for line in code if "async def" in line or "await " in line]
    assert offenders == []
