"""`Engine` -- the shared home for hooks, keys, catalog and the HTTP machinery.

Where TypeScript has `createEngine({...})` returning a bag of plugin instances,
Python has a class, per the API contract's "no `create_llm()` *and* `LLM()`"
rule. It is a thin coordinator: classes never consult it, and only the facades
(`LLM`, `AsyncLLM`) resolve their fetch and hooks against one.

**It holds BOTH cores.** A synchronous `RequestExecutor` and an asynchronous
`NetworkEngine`, sharing one HookBus, one catalog and one set of keys. That is
not a wrapper in either direction -- it is the same configuration handed to two
real implementations, so `LLM(engine=e)` and `AsyncLLM(engine=e)` see the same
retry policy, the same subscribers and the same cost ledger.

Hooks are decorators, and the contract says why::

    `engine.hooks.on('onCompletion', fn)` is a string-keyed registry -- natural
    in TypeScript, where the string literal is what types the handler. Python
    has no such mechanism, so a string key buys nothing and costs autocomplete.

Unsubscribing is the awkward part of decorator APIs, because the obvious design
(return the unsubscribe function) rebinds the name and throws away the handler.
So the decorator returns the function UNCHANGED with `.unsubscribe()` attached::

    @engine.on_completion
    def track(ctx): ...

    track.unsubscribe()      # `track` is still the function you wrote
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from typing import Any

from ..bus.hook_bus import HookBus
from ..catalog.catalog import ModelCatalog, resolve_catalog
from ..cost_collector import CostCollector
from ..network.engine import NetworkEngine, NetworkEngineConfig
from ..network.executor import RequestExecutor
from ..network.retry import RetryConfig, merge_retry
from ..network.types import HttpRequest, HttpResponse
from ..persistence import FilePersistence, MemoryPersistence, Persistence
from ..results import Usage

Handler = Callable[..., Any]

#: The process-wide default, for callers who build an engine and never pass it.
#: `Engine(register_as_default=False)` opts out -- which every test and every
#: example that builds more than one engine does, because a global that the last
#: constructor silently wins is a bug waiting for a second engine.
_default: Engine | None = None


def default_engine() -> Engine | None:
    """The engine registered as the default, if any."""
    return _default


def clear_default_engine() -> None:
    """Forget the registered default.

    Exists for tests and for a process that rebuilds its configuration: a global
    that only ever accumulates is one a later caller inherits without asking,
    and `register_as_default=False` only helps the engine that opts out.
    """
    global _default
    _default = None


def resolve_persistence(
    config: Persistence | Mapping[str, Any] | None,
) -> Persistence:
    """A store, from a store, a `{"type": ...}` config, or nothing at all."""
    if config is None:
        return MemoryPersistence()
    # Already a store: recognised by shape, so an application's own class works
    # without importing anything from here.
    if not isinstance(config, Mapping):
        return config
    kind = config.get("type")
    if kind == "memory":
        return MemoryPersistence()
    if kind == "file":
        directory = config.get("dir")
        if not directory:
            raise ValueError('Engine: persistence type "file" requires a `dir` field')
        return FilePersistence(str(directory))
    raise ValueError(f"Engine: unknown persistence type {kind!r}")


class Engine:
    """Shared configuration for everything that makes HTTP calls."""

    def __init__(
        self,
        *,
        transport: Any = None,
        async_transport: Any = None,
        api_keys: Mapping[str, str] | None = None,
        retry: Any = None,
        queues: Mapping[str, Any] | None = None,
        hooks: HookBus | None = None,
        catalog: ModelCatalog | str | None = None,
        persistence: Persistence | Mapping[str, Any] | None = None,
        check_response_shapes: bool = False,
        register_as_default: bool = True,
        plugins: Sequence[Any] = (),
    ) -> None:
        self.hooks = hooks or HookBus()
        self.catalog = resolve_catalog(catalog)
        #: Durable storage for whatever wants it. Always present, defaulting to
        #: memory: a plugin that had to check for `None` before every write
        #: would grow a second, subtly different in-process store to fall back
        #: to, and the two would disagree about what was saved.
        self.persistence: Persistence = resolve_persistence(persistence)
        #: The ledger. Built here rather than on demand because it has to be
        #: SUBSCRIBED before the first call to record it -- a collector created
        #: when someone first reads `engine.cost` would report an empty run.
        self.cost = CostCollector(self.hooks, self.catalog)
        #: Attached AFTER the bus exists and BEFORE anything is emitted:
        #: a plugin that subscribed later would miss the events that
        #: happened while the engine was being built.
        self.plugins = list(plugins)
        for plugin in self.plugins:
            attach = getattr(plugin, "attach", None)
            if callable(attach):
                attach(self)
        self.api_keys: dict[str, str] = dict(api_keys or {})
        self.check_response_shapes = check_response_shapes
        self._retry_override = retry
        self._queue_settings: dict[str, Any] = dict(queues or {})
        self._transport = transport
        self._async_transport = async_transport
        self._executor = RequestExecutor(self.hooks)
        self._network: NetworkEngine | None = None

        if register_as_default:
            global _default
            _default = self

    # -- retry policy --------------------------------------------------------

    def retry_for(self, queue_name: str, request_retry: Any = None) -> RetryConfig:
        """The policy for one request: engine, then queue, then request.

        Nested groups merge at every layer, so overriding one backoff knob keeps
        the rest -- otherwise a one-line tweak silently resets the schedule.
        """
        queue_retry = (self._queue_settings.get(queue_name) or {}).get("retry")
        return merge_retry(self._retry_override, queue_retry, request_retry)

    # -- the synchronous side ------------------------------------------------

    def request(
        self,
        *,
        url: str,
        provider: str = "",
        model: str = "",
        body: Any = None,
        method: str = "POST",
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        retry: Any = None,
        queue_name: str | None = None,
    ) -> HttpResponse:
        """Make one HTTP call, synchronously, under this engine's retry policy."""
        req: HttpRequest = {
            "url": url,
            "method": method,
            "headers": dict(headers or {}),
            "body": body,
            "provider": provider,
            "model": model,
        }
        if timeout is not None:
            req["timeout"] = timeout
        if retry is not None:
            req["retry"] = retry
        name = queue_name or f"{provider}/{model}"
        return self._executor.execute(req, self._send, self.retry_for(name, retry))

    def fetch(self, req: HttpRequest, options: Any = None) -> HttpResponse:
        """The `fetch` a synchronous `LLM` is injected with."""
        name = _queue_name_of(req, options)
        return self._executor.execute(
            req, self._send, self.retry_for(name, req.get("retry"))
        )

    def fetches_for(self, transport: Any) -> tuple[Any, Any]:
        """The fetch pair for a client that brought its OWN transport.

        An engine holds a transport as a FLEET default; a client given one has
        said something more specific. Both still matter, and they are different
        jobs -- the engine queues, rate-limits and retries, the transport puts
        the bytes on the wire -- so a client with both goes through this engine's
        policy and sends through its own transport.

        Silently ignoring one of them is the failure worth avoiding: preferring
        the engine sends a test's stub traffic to the real provider, and
        preferring the transport drops the whole fleet's rate limiting for that
        client without saying so.
        """
        from ..transport import as_fetch, as_fetch_stream

        send = as_fetch(transport)

        def fetch(req: HttpRequest, options: Any = None) -> HttpResponse:
            name = _queue_name_of(req, options)
            return self._executor.execute(
                req, lambda r: send(r), self.retry_for(name, req.get("retry"))
            )

        return fetch, as_fetch_stream(transport)

    def realtime_connection(self, request: Any, timeout: float | None = None) -> Any:
        """A live socket, on this engine's hook bus.

        The socket sibling of `fetch`, and NOT queued: a live connection has no
        per-call retry, no rate limit and no idempotency key, because the call
        IS the connection and retrying it would restart the conversation. What
        the engine still owns is the hooks, which is why this exists instead of
        adapters opening their own sockets.
        """
        from ..realtime.connection import RealtimeConnection

        kwargs: dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return RealtimeConnection(request, self.hooks, **kwargs)

    def fetch_stream(self, req: HttpRequest, options: Any = None) -> Iterator[Any]:
        """The `fetch_stream` a synchronous `LLM` is injected with.

        Not retried: a stream that failed part-way has already delivered events
        the caller acted on, and starting again would repeat them. The queue's
        other protections do not apply to a connection the caller holds open.
        """
        from ..transport import as_fetch_stream, http_transport

        return as_fetch_stream(self._transport or http_transport())(req, options)

    @property
    def _send(self) -> Callable[[HttpRequest], HttpResponse]:
        from ..transport import as_fetch, http_transport

        return as_fetch(self._transport or http_transport())

    # -- the asynchronous side -----------------------------------------------

    @property
    def network(self) -> NetworkEngine:
        """The queued async engine, built on first use.

        Lazy because an `AsyncClient` binds to the event loop it is created on:
        building one in `__init__` would bind it to whatever loop happened to be
        running then, which is usually none.
        """
        if self._network is None:
            from ..transport import ahttp_transport, as_async_fetch, as_async_fetch_stream

            transport = self._async_transport or ahttp_transport()
            self._network = NetworkEngine(
                NetworkEngineConfig(
                    hooks=self.hooks,
                    fetch=as_async_fetch(transport),
                    fetch_stream=as_async_fetch_stream(transport),
                    queues=self._queue_settings,
                    retry=self._retry_override,
                )
            )
        return self._network

    async def afetch(self, req: HttpRequest, options: Any = None) -> HttpResponse:
        """The `fetch` an `AsyncLLM` is injected with -- queued and retried."""
        return await self.network.fetch(req, options)

    def afetches_for(self, transport: Any) -> tuple[Any, Any]:
        """`fetches_for`, on the asynchronous side.

        The network engine is what queues here, so the transport is swapped for
        this client alone rather than on the shared engine -- setting it there
        would redirect the whole fleet.
        """
        from ..transport import as_async_fetch, as_async_fetch_stream

        send = as_async_fetch(transport)

        async def fetch(req: HttpRequest, options: Any = None) -> HttpResponse:
            # On the request rather than the options: the queue hands the
            # request to its worker and the options do not travel with it.
            return await self.network.fetch({**dict(req), "send": send}, options)

        return fetch, as_async_fetch_stream(transport)

    def afetch_stream(self, req: HttpRequest, options: Any = None) -> AsyncIterator[Any]:
        """The `fetch_stream` an `AsyncLLM` is injected with."""
        from ..transport import ahttp_transport, as_async_fetch_stream

        return as_async_fetch_stream(self._async_transport or ahttp_transport())(req, options)

    # -- hooks ---------------------------------------------------------------

    # -- emitting --------------------------------------------------------------
    #
    # Typed emitters for the handful of events an APPLICATION raises itself: a
    # budget warning of its own, a tool call it dispatched outside the agent
    # loop. Named parameters rather than a payload dict, and snake_case like
    # every other name here, so the wire spelling stays inside the library.

    def emit(self, name: str, payload: Mapping[str, Any]) -> None:
        """Emit by name, for an event with no typed emitter."""
        self.hooks.emit_sync(_camel(name), dict(payload))

    def emit_warning(
        self, *, source: str, code: str, message: str, details: Any = None
    ) -> None:
        """Raise an `on_warning` of your own."""
        self.emit(
            "on_warning",
            {
                "source": source,
                "code": code,
                "message": message,
                "details": details or {},
            },
        )

    def emit_completion(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        response: Mapping[str, Any] | None = None,
        ctx: Mapping[str, Any] | None = None,
    ) -> None:
        """Raise an `on_completion` for a call this engine did not make.

        `usage` is built here so a subscriber reads `ctx.usage.output_tokens`
        whoever emitted it -- a synthetic event whose usage had a different shape
        from a real one would make every subscriber special-case the source.
        """
        self.emit(
            "on_completion",
            {
                "provider": provider,
                "model": model,
                "usage": Usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                ),
                "response": dict(response or {}),
                "request": {},
                "ctx": dict(ctx or {}),
            },
        )

    def emit_tool_call_start(
        self,
        *,
        tool_name: str,
        call_id: str,
        step: int = 0,
        run_id: str | None = None,
        agent_id: str | None = None,
        arguments: Mapping[str, Any] | None = None,
    ) -> None:
        """Raise an `on_tool_call_start` for a tool dispatched outside a loop."""
        self.emit(
            "on_tool_call_start",
            {
                "runId": run_id,
                "agentId": agent_id,
                "step": step,
                "callId": call_id,
                "toolName": tool_name,
                "arguments": dict(arguments or {}),
            },
        )

    def emit_request(
        self,
        *,
        url: str,
        method: str = "POST",
        headers: Mapping[str, str] | None = None,
        provider: str = "",
        model: str = "",
    ) -> None:
        """Raise an `on_request_start` for a call made outside this engine."""
        self.emit(
            "on_request_start",
            {
                "url": url,
                "method": method,
                "headers": dict(headers or {}),
                "provider": provider,
                "model": model,
            },
        )

    def emit_retry(
        self,
        *,
        provider: str,
        model: str,
        attempt: int,
        reason: str,
        delay_ms: float = 0.0,
    ) -> None:
        """Raise an `on_retry`."""
        self.emit(
            "on_retry",
            {
                "provider": provider,
                "model": model,
                "attempt": attempt,
                "reason": reason,
                "delayMs": delay_ms,
            },
        )

    # -- subscribing -----------------------------------------------------------

    def on(self, name: str, handler: Handler) -> Callable[[], None]:
        """Subscribe by name, for genuinely dynamic subscription.

        Takes the snake_case name (`"on_completion"`), like every other name in
        this API. The escape hatch a plugin loader needs; the decorators below
        are the path the docs lead with.
        """
        return self.hooks.on(_camel(name), handler)

    def on_any(self, handler: Handler) -> Handler:
        """Subscribe to EVERY event, as one tagged value."""
        unsubscribe = self.hooks.on_any(handler)
        handler.unsubscribe = unsubscribe  # type: ignore[attr-defined]
        return handler

    def _decorate(self, hook: str, handler: Handler) -> Handler:
        """Subscribe and hand the function BACK, with `.unsubscribe()` on it.

        Returning the handler unchanged is what makes this usable as a
        decorator: returning the unsubscribe function instead would rebind the
        name and throw the handler away.
        """
        unsubscribe = self.hooks.on(hook, handler)
        handler.unsubscribe = unsubscribe  # type: ignore[attr-defined]
        return handler

    def on_warning(self, handler: Handler) -> Handler:
        """Subscribe to `onWarning`."""
        return self._decorate("onWarning", handler)

    def on_internal_error(self, handler: Handler) -> Handler:
        """Subscribe to `onInternalError`."""
        return self._decorate("onInternalError", handler)

    def on_enqueue(self, handler: Handler) -> Handler:
        """Subscribe to `onEnqueue`."""
        return self._decorate("onEnqueue", handler)

    def on_dequeue(self, handler: Handler) -> Handler:
        """Subscribe to `onDequeue`."""
        return self._decorate("onDequeue", handler)

    def on_queue_timeout(self, handler: Handler) -> Handler:
        """Subscribe to `onQueueTimeout`."""
        return self._decorate("onQueueTimeout", handler)

    def on_rate_limit_update(self, handler: Handler) -> Handler:
        """Subscribe to `onRateLimitUpdate`."""
        return self._decorate("onRateLimitUpdate", handler)

    def on_request_start(self, handler: Handler) -> Handler:
        """Subscribe to `onRequestStart`."""
        return self._decorate("onRequestStart", handler)

    def on_request_complete(self, handler: Handler) -> Handler:
        """Subscribe to `onRequestComplete`."""
        return self._decorate("onRequestComplete", handler)

    def on_model_error(self, handler: Handler) -> Handler:
        """Subscribe to `onModelError`."""
        return self._decorate("onModelError", handler)

    def on_rate_limit_hit(self, handler: Handler) -> Handler:
        """Subscribe to `onRateLimitHit`."""
        return self._decorate("onRateLimitHit", handler)

    def on_retry(self, handler: Handler) -> Handler:
        """Subscribe to `onRetry`."""
        return self._decorate("onRetry", handler)

    def on_stream_chunk(self, handler: Handler) -> Handler:
        """Subscribe to `onStreamChunk`."""
        return self._decorate("onStreamChunk", handler)

    def on_realtime_open(self, handler: Handler) -> Handler:
        """Subscribe to `onRealtimeOpen`."""
        return self._decorate("onRealtimeOpen", handler)

    def on_realtime_frame(self, handler: Handler) -> Handler:
        """Subscribe to `onRealtimeFrame`."""
        return self._decorate("onRealtimeFrame", handler)

    def on_realtime_close(self, handler: Handler) -> Handler:
        """Subscribe to `onRealtimeClose`."""
        return self._decorate("onRealtimeClose", handler)

    def on_realtime_error(self, handler: Handler) -> Handler:
        """Subscribe to `onRealtimeError`."""
        return self._decorate("onRealtimeError", handler)

    def on_client_create(self, handler: Handler) -> Handler:
        """Subscribe to `onClientCreate`."""
        return self._decorate("onClientCreate", handler)

    def on_client_destroy(self, handler: Handler) -> Handler:
        """Subscribe to `onClientDestroy`."""
        return self._decorate("onClientDestroy", handler)

    def on_message_resolve(self, handler: Handler) -> Handler:
        """Subscribe to `onMessageResolve`."""
        return self._decorate("onMessageResolve", handler)

    def on_before_submit(self, handler: Handler) -> Handler:
        """Subscribe to `onBeforeSubmit`."""
        return self._decorate("onBeforeSubmit", handler)

    def on_completion(self, handler: Handler) -> Handler:
        """Subscribe to `onCompletion`."""
        return self._decorate("onCompletion", handler)

    def on_agent_create(self, handler: Handler) -> Handler:
        """Subscribe to `onAgentCreate`."""
        return self._decorate("onAgentCreate", handler)

    def on_agent_destroy(self, handler: Handler) -> Handler:
        """Subscribe to `onAgentDestroy`."""
        return self._decorate("onAgentDestroy", handler)

    def on_run_start(self, handler: Handler) -> Handler:
        """Subscribe to `onRunStart`."""
        return self._decorate("onRunStart", handler)

    def on_step_start(self, handler: Handler) -> Handler:
        """Subscribe to `onStepStart`."""
        return self._decorate("onStepStart", handler)

    def on_step_complete(self, handler: Handler) -> Handler:
        """Subscribe to `onStepComplete`."""
        return self._decorate("onStepComplete", handler)

    def on_tool_call_start(self, handler: Handler) -> Handler:
        """Subscribe to `onToolCallStart`."""
        return self._decorate("onToolCallStart", handler)

    def on_tool_call_complete(self, handler: Handler) -> Handler:
        """Subscribe to `onToolCallComplete`."""
        return self._decorate("onToolCallComplete", handler)

    def on_tool_call_error(self, handler: Handler) -> Handler:
        """Subscribe to `onToolCallError`."""
        return self._decorate("onToolCallError", handler)

    def on_tool_search(self, handler: Handler) -> Handler:
        """Subscribe to `onToolSearch`."""
        return self._decorate("onToolSearch", handler)

    def on_run_complete(self, handler: Handler) -> Handler:
        """Subscribe to `onRunComplete`."""
        return self._decorate("onRunComplete", handler)

    def on_run_error(self, handler: Handler) -> Handler:
        """Subscribe to `onRunError`."""
        return self._decorate("onRunError", handler)

    def on_guardrail_triggered(self, handler: Handler) -> Handler:
        """Subscribe to `onGuardrailTriggered`."""
        return self._decorate("onGuardrailTriggered", handler)

    def on_approval_requested(self, handler: Handler) -> Handler:
        """Subscribe to `onApprovalRequested`."""
        return self._decorate("onApprovalRequested", handler)

    def on_approval_resolved(self, handler: Handler) -> Handler:
        """Subscribe to `onApprovalResolved`."""
        return self._decorate("onApprovalResolved", handler)

    def on_server_request(self, handler: Handler) -> Handler:
        """Subscribe to `onServerRequest`."""
        return self._decorate("onServerRequest", handler)

    def on_server_response(self, handler: Handler) -> Handler:
        """Subscribe to `onServerResponse`."""
        return self._decorate("onServerResponse", handler)

    def on_auth_fail(self, handler: Handler) -> Handler:
        """Subscribe to `onAuthFail`."""
        return self._decorate("onAuthFail", handler)

    def on_cost_entry(self, handler: Handler) -> Handler:
        """Subscribe to `onCostEntry`."""
        return self._decorate("onCostEntry", handler)

    def on_budget_warning(self, handler: Handler) -> Handler:
        """Subscribe to `onBudgetWarning`."""
        return self._decorate("onBudgetWarning", handler)

    def on_budget_exceeded(self, handler: Handler) -> Handler:
        """Subscribe to `onBudgetExceeded`."""
        return self._decorate("onBudgetExceeded", handler)

    def on_context_measure(self, handler: Handler) -> Handler:
        """Subscribe to `onContextMeasure`."""
        return self._decorate("onContextMeasure", handler)

    def on_media_generated(self, handler: Handler) -> Handler:
        """Subscribe to `onMediaGenerated`."""
        return self._decorate("onMediaGenerated", handler)

    def on_media_error(self, handler: Handler) -> Handler:
        """Subscribe to `onMediaError`."""
        return self._decorate("onMediaError", handler)

    def on_media_progress(self, handler: Handler) -> Handler:
        """Subscribe to `onMediaProgress`."""
        return self._decorate("onMediaProgress", handler)

    def on_internal_tool_call_start(self, handler: Handler) -> Handler:
        """Subscribe to `onInternalToolCallStart`."""
        return self._decorate("onInternalToolCallStart", handler)

    def on_internal_tool_call_complete(self, handler: Handler) -> Handler:
        """Subscribe to `onInternalToolCallComplete`."""
        return self._decorate("onInternalToolCallComplete", handler)

    def on_internal_tool_call_error(self, handler: Handler) -> Handler:
        """Subscribe to `onInternalToolCallError`."""
        return self._decorate("onInternalToolCallError", handler)

    def on_mcp_connect(self, handler: Handler) -> Handler:
        """Subscribe to `onMcpConnect`."""
        return self._decorate("onMcpConnect", handler)

    def on_mcp_tool_call(self, handler: Handler) -> Handler:
        """Subscribe to `onMcpToolCall`."""
        return self._decorate("onMcpToolCall", handler)

    def on_mcp_error(self, handler: Handler) -> Handler:
        """Subscribe to `onMcpError`."""
        return self._decorate("onMcpError", handler)


def _camel(name: str) -> str:
    """`on_completion` -> `onCompletion`, the name the bus is keyed by."""
    head, *rest = name.split("_")
    return head + "".join(word.title() for word in rest)


def _queue_name_of(req: HttpRequest, options: Any) -> str:
    options = options or {}
    if options.get("queueName"):
        return str(options["queueName"])
    return f"{req.get('provider')}/{req.get('model')}"


__all__ = ["Engine", "clear_default_engine", "default_engine"]
