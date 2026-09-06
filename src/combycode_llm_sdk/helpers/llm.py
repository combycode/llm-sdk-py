"""`LLM` and `AsyncLLM` -- the client a caller actually holds.

Where TypeScript has `createLLM({...})`, Python has a class, and the API
contract is explicit about there being only one way::

    Where TS offers a factory function, Python gets a class.
    No `create_llm()` *and* `LLM()`.

Each is a thin face on one of the two cores: `LLM` on `LLMClient`, `AsyncLLM` on
`AsyncLLMClient`. Neither wraps the other and neither wraps a loop -- the two
cores are real, and this layer only translates vocabulary:

- snake_case keyword arguments in, camelCase `ExecuteOptions` out;
- wire dicts back, frozen `Completion` and event dataclasses out.

Both directions live in `_options` and `results.Completion.of`, so a renamed
field is one edit rather than a search.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any

from ..bus.hook_bus import HookBus
from ..catalog.catalog import ModelCatalog, resolve_catalog
from ..events import Event, to_event
from ..llm.async_client import AsyncLLMClient
from ..llm.client import LLMClient
from ..llm.client_base import BaseLLMClient
from ..llm.providers.anthropic.messages import AnthropicAdapter
from ..llm.providers.google.generate import GoogleAdapter
from ..llm.providers.google.interactions import GoogleInteractionsAdapter
from ..llm.providers.openai.completions import OpenAIAdapter
from ..llm.providers.openai.responses import OpenAIResponsesAdapter
from ..llm.providers.openrouter.completions import OpenRouterAdapter
from ..llm.providers.openrouter.responses import OpenRouterResponsesAdapter
from ..llm.providers.xai.completions import XAIAdapter
from ..llm.providers.xai.responses import XAIResponsesAdapter
from ..results import Completion, build_cost
from .client_resolver import resolve_model
from .engine import default_engine

#: snake_case keyword -> the `ExecuteOptions` key it becomes.
#:
#: Mostly mechanical, and the exceptions are the interesting part:
#:
#: - `reasoning` is the public name for what the wire calls `thinking`. The
#:   examples settled on it and it is what the provider docs call the feature.
#: - `state` is the server-side conversation id, which the wire carries as
#:   `previousResponseId`. A caller passes back what `Completion.state` gave
#:   them and never sees the wire name.
#: - `stateful` keeps its name: it is the on/off switch for the automatic
#:   version of the same thing.
_OPTION_NAMES = {
    "system": "system",
    "max_tokens": "maxTokens",
    "temperature": "temperature",
    "top_p": "topP",
    "top_k": "topK",
    "seed": "seed",
    "presence_penalty": "presencePenalty",
    "frequency_penalty": "frequencyPenalty",
    "stop": "stop",
    "tools": "tools",
    "tool_choice": "toolChoice",
    "structured": "structured",
    "reasoning": "thinking",
    "cache": "cache",
    "service_tier": "serviceTier",
    "moderation": "moderation",
    "provider_options": "providerOptions",
    "audio": "audio",
    "output_modalities": "outputModalities",
    "state": "previousResponseId",
    "stateful": "stateful",
    "timeout": "timeout",
    "history": "history",
    "cache_key": "cacheKey",
    "cache_name": "cacheName",
    "config_name": "configName",
    "ctx": "ctx",
}


def _declarations(tools: Any) -> Any:
    """`tools=` accepts what `@tool` produces, here as everywhere else.

    A `Tool` handed straight to the wire serialised to `{"name": ...,
    "parameters": {}}`: the model was told a function existed and told nothing
    about it -- no description, no arguments -- which reads as a model that
    ignores its tools rather than as a request that never described them.

    `run_tools` already normalised this, so `complete(tools=[my_tool])` worked
    and `LLM(...).complete(tools=[my_tool])` silently did not. One funnel, one
    answer.
    """
    from .tool import Tool

    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        return tools
    return [dict(t.definition) if isinstance(t, Tool) else t for t in tools]


def _options(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """snake_case keyword arguments -> the camelCase options the client takes.

    An unknown keyword is an ERROR, not a passthrough: the one untyped hole in
    the TypeScript request was the one place a typo produced silence rather than
    a failure -- `promtCacheOptions` type-checked and was simply never sent. A
    caller who needs something this SDK does not model yet has
    `provider_options=` for exactly that, and reaches it deliberately.
    """
    unknown = sorted(set(kwargs) - set(_OPTION_NAMES) - {"builtin_tools"})
    if unknown:
        known = ", ".join(sorted(_OPTION_NAMES))
        raise TypeError(
            f"unexpected option(s) {unknown}. Known options: {known}. "
            "For a provider parameter this SDK does not model, use provider_options=."
        )

    options: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key == "builtin_tools" or value is None:
            continue
        options[_OPTION_NAMES[key]] = _declarations(value) if key == "tools" else value

    # `cache=True` is the ergonomic form of the wire's `'auto'`. The wire also
    # takes `'off'` and a per-part dict, and both pass through untouched.
    if options.get("cache") is True:
        options["cache"] = "auto"
    elif options.get("cache") is False:
        options["cache"] = "off"

    # `builtin_tools=["web_search"]` is sugar for the hosted-tool entries the
    # wire wants, and it MERGES with any function tools rather than replacing
    # them -- a caller passing both means both.
    builtin = kwargs.get("builtin_tools")
    if builtin:
        options["tools"] = [*(options.get("tools") or []), *({"type": t} for t in builtin)]
    return options


def default_adapter_factory() -> Any:
    """Build the adapter for a (provider, api), as `createLLM` does.

    The api decides within a provider: OpenAI and xAI each have a Responses and
    a Chat Completions adapter, and Google a generateContent and an Interactions
    one. Which api applies was already resolved by the client, from the caller's
    `api=`, the catalog's per-model preference, or the provider default.
    """

    def factory(provider: str, api_key: str, api: str, base_url: str | None = None) -> Any:
        cfg = {"apiKey": api_key, "baseURL": base_url}
        if provider == "anthropic":
            return AnthropicAdapter(cfg)
        if provider == "openai":
            return OpenAIResponsesAdapter(cfg) if api == "responses" else OpenAIAdapter(cfg)
        if provider == "google":
            return (
                GoogleInteractionsAdapter(cfg) if api == "interactions" else GoogleAdapter(cfg)
            )
        if provider == "xai":
            return XAIResponsesAdapter(cfg) if api == "responses" else XAIAdapter(cfg)
        if provider == "openrouter":
            return (
                OpenRouterResponsesAdapter(cfg)
                if api == "responses"
                else OpenRouterAdapter(cfg)
            )
        raise ValueError(f"LLM: no default adapter for provider '{provider}'")

    return factory


class _BaseLLM:
    """Construction, shared by both faces.

    The client is built here, in one place, so the two faces cannot disagree
    about how a model string is read or which adapter a provider gets.
    """

    _client_cls: type[BaseLLMClient] = LLMClient

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        provider: str | None = None,
        transport: Any = None,
        api: str | None = None,
        engine: Any = None,
        system: str | None = None,
        hooks: HookBus | None = None,
        catalog: ModelCatalog | str | None = None,
        check_response_shapes: bool = False,
        base_url: str | None = None,
        **client_options: Any,
    ) -> None:
        resolved = resolve_model(model, provider, type(self).__name__)
        provider_name, model_name = resolved["provider"], resolved["model"]

        # An engine supplies everything shared: the hook bus every subscriber is
        # already on, the catalog, the keys, and the fetch that carries the
        # queue and the retry policy. Anything passed explicitly still wins --
        # a caller naming a key for one client should not have to reconfigure
        # the engine for it.
        #
        # The ambient default is a FALLBACK, not an override: a caller who named
        # a transport has said where this client's requests go, and letting a
        # global registered somewhere else win over that is how a stub transport
        # silently starts making real calls.
        if engine is None and transport is None:
            engine = default_engine()
        if engine is not None:
            catalog = catalog or engine.catalog
            hooks = hooks or engine.hooks
            api_key = api_key or engine.api_keys.get(provider_name)
            check_response_shapes = check_response_shapes or engine.check_response_shapes

        catalog = resolve_catalog(catalog)
        key = api_key
        if not key:
            where = " or engine.api_keys" if engine is not None else ""
            raise ValueError(
                f'{type(self).__name__}: no API key for provider "{provider_name}". '
                f"Pass api_key={where}."
            )

        self._hooks = hooks or HookBus()
        #: The engine this client sends through, or None when it was given a
        #: transport directly. Exposed so a fleet can be checked to be sharing
        #: one -- two engines against one provider means two rate limiters, and
        #: the concurrency configured for it silently doubles.
        self.engine = engine
        fetch, fetch_stream = self._resolve_fetches(engine, transport)
        self._client = self._client_cls(
            {
                "provider": provider_name,
                # The catalog's alias index turns our normalised slug into the
                # exact id the provider will accept; an already-callable id or
                # an unknown model passes through untouched.
                "model": catalog.resolve_model_id(provider_name, model_name),
                "apiKey": key,
                "system": system,
                "baseURL": base_url,
                "api": api,
                "adapter": default_adapter_factory(),
                "fetch": fetch,
                "fetchStream": fetch_stream,
                "hooks": self._hooks,
                "catalog": catalog,
                "checkResponseShapes": check_response_shapes,
                **client_options,
            }
        )
        self._catalog = catalog
        self.provider = provider_name
        self.model = self._client.model

    def _hooks_for_executor(self) -> HookBus:
        """The bus the client will emit on, so network and llm hooks agree."""
        return self._hooks

    def _fetches(self, transport: Any) -> tuple[Any, Any]:
        """The fetch pair for a bare transport, when there is no engine."""
        raise NotImplementedError

    def _resolve_fetches(self, engine: Any, transport: Any) -> tuple[Any, Any]:
        """Who queues and who sends.

        Both given is a real combination, not a conflict: a fleet shares one
        engine so there is one rate limiter, while each client may still bring
        its own transport (a stub in a test, a proxy for one tenant). The engine
        keeps its policy and the transport does the sending.
        """
        if engine is not None and transport is not None:
            return self._engine_fetches_with(engine, transport)
        if engine is not None:
            return self._engine_fetches(engine)
        return self._fetches(transport)

    def _engine_fetches(self, engine: Any) -> tuple[Any, Any]:
        """The fetch pair from an engine -- queued, retried, and hooked."""
        raise NotImplementedError

    def _engine_fetches_with(self, engine: Any, transport: Any) -> tuple[Any, Any]:
        """The engine's policy, the caller's transport."""
        raise NotImplementedError

    @property
    def hooks(self) -> HookBus:
        """The hook bus this client emits on."""
        return self._client.hooks

    @property
    def client(self) -> Any:
        """The core underneath, for callers who need what this face does not expose."""
        return self._client

    def destroy(self) -> None:
        self._client.destroy()

    def _completion(self, wire: Mapping[str, Any], parsed: Any = None) -> Completion:
        answer = Completion.of(
            wire,
            provider=self.provider,
            model=self.model,
            api=self._client.api,
            cost=build_cost(self._catalog, self.provider, self.model, wire),
            parsed=parsed,
        )
        return self._bind_files(answer)

    def _bind_files(self, answer: Completion) -> Completion:
        """Give each file descriptor a way to fetch itself.

        Bound here rather than inside `Completion.of`, which is a pure view over
        a wire dict and has no client to fetch with. Without it `descriptor.read()`
        would be a method that exists and cannot work -- which is worse than not
        having it, because it type-checks.
        """
        if not answer.files:
            return answer
        from dataclasses import replace

        return replace(
            answer,
            files=tuple(replace(f, fetch=self._fetch_file) for f in answer.files),
        )

    def _fetch_file(self, descriptor: Any) -> bytes:
        """The bytes of one descriptor, through this client's auth and engine."""
        raise NotImplementedError


class LLM(_BaseLLM):
    """A synchronous LLM client.

        llm = LLM(model="anthropic/claude-haiku-4.5", api_key=key)
        print(llm.complete("Say hi").text)

        for event in llm.stream("Count to five"):
            if event.type == "text":
                print(event.text, end="")
    """

    _client_cls = LLMClient
    _client: LLMClient

    def _fetches(self, transport: Any) -> tuple[Any, Any]:
        """Even without an engine, a response still gets classified.

        Handing the raw transport straight to the client meant nothing checked
        the status: a 400 parsed into an empty completion with
        `finish_reason: "stop"` and no exception -- the caller saw a model that
        had nothing to say rather than a request that was rejected. So the bare
        path gets a `RequestExecutor` too, which is where the taxonomy and the
        retry policy live.
        """
        from ..network.executor import RequestExecutor
        from ..network.retry import DEFAULT_RETRY
        from ..transport import as_fetch, as_fetch_stream, http_transport

        chosen = transport or http_transport()
        send = as_fetch(chosen)
        executor = RequestExecutor(self._hooks_for_executor())

        def fetch(req: Any, options: Any = None) -> Any:
            return executor.execute(req, lambda r: send(r), DEFAULT_RETRY)

        return fetch, as_fetch_stream(chosen)

    def _engine_fetches(self, engine: Any) -> tuple[Any, Any]:
        return engine.fetch, engine.fetch_stream

    def _fetch_file(self, descriptor: Any) -> bytes:
        retrieved = self.retrieve_file(descriptor)
        # `retrieve_file` answers `{bytes, name, mimeType, size}` -- the bytes
        # live under `bytes`, not `data`; `data` is the INLINE base64 a
        # descriptor may already carry, which is a different thing.
        raw = retrieved.get("bytes") if isinstance(retrieved, Mapping) else retrieved
        return bytes(raw or b"")

    def _engine_fetches_with(self, engine: Any, transport: Any) -> tuple[Any, Any]:
        pair: tuple[Any, Any] = engine.fetches_for(transport)
        return pair

    def complete(self, input_: Any, **options: Any) -> Completion:
        """One completion."""
        return self._completion(self._client.complete(input_, _options(options)))

    def structured_complete(self, input_: Any, schema: Mapping[str, Any], **options: Any) -> Any:
        """A completion parsed against `schema`, returning the parsed object."""
        return self._client.structured_complete(input_, schema, _options(options))

    def stream(self, input_: Any, **options: Any) -> Iterator[Event]:
        """Stream a completion, yielding events as they arrive."""
        for raw in self._client.stream(input_, _options(options)):
            yield to_event(raw)

    def retrieve_file(self, file: Any) -> Any:
        """The bytes of a hosted-tool output file, plus its name and type."""
        return self._client.retrieve_file(_file_ref(file))

    def stream_file(self, file: Any) -> Any:
        """A hosted-tool output file as an iterator of bytes."""
        return self._client.stream_file(_file_ref(file))


class AsyncLLM(_BaseLLM):
    """An asynchronous LLM client. The twin of `LLM`; neither wraps the other.

        llm = AsyncLLM(model="anthropic/claude-haiku-4.5", api_key=key)
        print((await llm.complete("Say hi")).text)

        async for event in llm.stream("Count to five"):
            if event.type == "text":
                print(event.text, end="")
    """

    _client_cls = AsyncLLMClient
    _client: AsyncLLMClient

    def _fetches(self, transport: Any) -> tuple[Any, Any]:
        """The async twin, and classified for the same reason as the sync one.

        A one-queue `NetworkEngine` rather than a bare transport: it is the
        async side's retry loop, and a caller who did not build an engine should
        not thereby lose error classification.
        """
        from ..network.engine import NetworkEngine, NetworkEngineConfig
        from ..transport import ahttp_transport, as_async_fetch, as_async_fetch_stream

        chosen = transport or ahttp_transport()
        engine = NetworkEngine(
            NetworkEngineConfig(
                hooks=self._hooks_for_executor(),
                fetch=as_async_fetch(chosen),
                fetch_stream=as_async_fetch_stream(chosen),
            )
        )
        return engine.fetch, as_async_fetch_stream(chosen)

    def _engine_fetches(self, engine: Any) -> tuple[Any, Any]:
        return engine.afetch, engine.afetch_stream

    def _engine_fetches_with(self, engine: Any, transport: Any) -> tuple[Any, Any]:
        pair: tuple[Any, Any] = engine.afetches_for(transport)
        return pair

    async def complete(self, input_: Any, **options: Any) -> Completion:
        """One completion."""
        return self._completion(await self._client.complete(input_, _options(options)))

    async def structured_complete(
        self, input_: Any, schema: Mapping[str, Any], **options: Any
    ) -> Any:
        """A completion parsed against `schema`, returning the parsed object."""
        return await self._client.structured_complete(input_, schema, _options(options))

    async def stream(self, input_: Any, **options: Any) -> AsyncIterator[Event]:
        """Stream a completion, yielding events as they arrive."""
        async for raw in self._client.stream(input_, _options(options)):
            yield to_event(raw)

    async def retrieve_file(self, file: Any) -> Any:
        """The bytes of a hosted-tool output file, plus its name and type."""
        return await self._client.retrieve_file(_file_ref(file))

    async def stream_file(self, file: Any) -> Any:
        """A hosted-tool output file as an async iterator of bytes."""
        return await self._client.stream_file(_file_ref(file))


def _file_ref(file: Any) -> Mapping[str, Any]:
    """Accept the `FileOutput` dataclass a `Completion` handed back, or a raw dict.

    The dataclass is what a caller actually holds -- `result.files[0]` -- and
    requiring them to unpack it would make the ergonomic type useless at the one
    place it is passed back in.
    """
    if isinstance(file, Mapping):
        return file
    return {
        "id": file.id,
        "name": file.name,
        "mimeType": file.mime_type,
        "data": file.data,
        "url": file.url,
        "source": file.source,
        "ref": file.ref,
    }


__all__ = ["LLM", "AsyncLLM", "default_adapter_factory"]
