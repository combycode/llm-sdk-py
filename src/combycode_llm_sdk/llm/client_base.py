"""What the sync and async clients share -- which is everything but the waiting.

`unified-library-ts/src/llm/client.ts` is one class, because JavaScript has one
colour of function. Python has two, and the API contract
(`examples/README.md`) is explicit that both are real:

> **Neither wraps the other.** Sync is not `asyncio.run()` around async (that
> breaks inside a running loop, which is exactly where notebook and web users
> live), and async is not a thread pool around sync.

So there are two clients. This module is what stops that being two ports of the
same logic: every step that does not wait lives here ONCE, and each client is a
short script of those steps with its own `await`s (or absence of them) between.

The seams are named for the points where the two genuinely differ -- a hook
emit, the fetch, the moderation calls. Read `complete` in either client and the
sequence is the same list of calls in the same order; that is the property this
file exists to make checkable, and `test_client_parity.py` asserts it by running
both over identical inputs and comparing.

**Nothing here awaits.** A method that needs to wait belongs in the two clients,
not in this base -- that is the whole rule, and the reason it can be checked by
grepping this file for `async`.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, MutableMapping
from functools import lru_cache
from typing import Any

from ..bus.hook_bus import HookBus
from ..catalog.catalog import ModelCatalog
from ..network.types import HttpRequest, HttpResponse
from ..types.request_context import RequestContext
from ..wire.interpreter import js_json
from .client_config import LLMClientConfig
from .client_internal import (
    PRIORITY_BACKGROUND,
    PRIORITY_INTERACTIVE,
    ClientRouting,
    build_assistant_message,
    build_context,
    extract_system,
    normalize_input,
    resolve_adapter,
    resolve_api,
)
from .moderation.runner import moderation_model, resolve_moderation_mode
from .moderation.types import MODERATION_DEFAULT_INTERVAL, MODERATION_DEFAULT_STRATEGY
from .response_shape import ResponseShapeChecker, load_response_shapes
from .server_state import resolve_server_state
from .types.messages import Message
from .types.provider import ProviderHttpRequest
from .types.response import empty_usage

#: Loaded once per process, not once per client: it is a 150 KB read and the
#: book is immutable. The CHECKER is per-client, because it remembers what it has
#: already reported.
_shapes = lru_cache(maxsize=1)(load_response_shapes)


def now_ms() -> float:
    """`performance.now()` -- a monotonic clock in milliseconds.

    `perf_counter`, not `time.time`: latency measured across a system clock
    adjustment is how a request gets reported as having taken minus four
    seconds.
    """
    return time.perf_counter() * 1000


#: The per-call option keys that map straight onto `NormalizedRequest` under the
#: same name. Written once rather than as twenty assignments repeated in
#: `complete` and `stream`, because the two lists drifting is exactly how
#: `structured` came to be sent on one path and not the other.
_PASSTHROUGH_OPTIONS = (
    "maxTokens",
    "temperature",
    "topP",
    "topK",
    "seed",
    "presencePenalty",
    "frequencyPenalty",
    "stop",
    "tools",
    "toolChoice",
    "thinking",
    "cache",
    "serviceTier",
    "moderation",
    "providerOptions",
    "audio",
    "outputModalities",
    "previousResponseId",
    "timeout",
    "signal",
)


class BaseLLMClient:
    """Construction, request assembly, and the phase seams both clients drive."""

    #: Names this class in its own error messages, so a sync client does not
    #: report itself as the async one.
    _label = "LLMClient"

    def __init__(self, config: LLMClientConfig) -> None:
        label = type(self)._label
        if not config.get("provider"):
            raise ValueError(f"{label}: provider is required")
        if not config.get("model"):
            raise ValueError(f"{label}: model is required")
        if not config.get("apiKey"):
            raise ValueError(f"{label}: apiKey is required")
        if not config.get("adapter") and not config.get("fetch"):
            raise ValueError(f"{label}: adapter (or factory) is required")
        if not config.get("fetch"):
            raise ValueError(f"{label}: fetch is required (typically engine.fetch)")

        self.id: str = str(uuid.uuid4())
        self.session_id: str = config.get("sessionId") or f"sess_{str(uuid.uuid4())[:12]}"
        self.provider: str = config["provider"]
        self.model: str = config["model"]
        self.system: str | None = config.get("system")
        self._api_key: str = config["apiKey"]
        self.hooks: HookBus = config.get("hooks") or HookBus()
        self._fetch = config["fetch"]
        self._fetch_stream = config.get("fetchStream")
        # The bundled catalog, not an empty one: this is where the model's
        # wire-spec pin comes from, and without it every request falls back to
        # deriving the spec from the model id -- which is the fallback for models
        # this build has never heard of, not the normal path.
        #
        # Assigned BEFORE `api` is resolved, because the API a model is callable
        # on is one of the things the catalog knows.
        self.catalog: ModelCatalog = config.get("catalog") or ModelCatalog.with_provider_defaults()
        self.api: str = resolve_api(
            config["provider"],
            config.get("api"),
            self.catalog.get_preferred_api(config["provider"], config["model"]),
        )
        self.mode: str = config.get("mode") or "foreground"
        self.batchable: bool = bool(config.get("batchable"))
        self._priority: int = (
            config["priority"]
            if config.get("priority") is not None
            else (PRIORITY_BACKGROUND if self.mode == "background" else PRIORITY_INTERACTIVE)
        )

        self._adapter = resolve_adapter(config, self.api)
        # One checker per client, because it remembers what it has already
        # reported: the same drift on every request for the rest of the process
        # is how a diagnostic gets ignored by the person it is for.
        self._shape_checker: ResponseShapeChecker | None = (
            ResponseShapeChecker(self.hooks, self.provider, self.api, _shapes())
            if config.get("checkResponseShapes")
            else None
        )

        self.routing = ClientRouting(
            queue_name=config.get("queueName") or f"{config['provider']}/{config['model']}",
            config_name=config.get("configName") or f"{config['provider']}/{config['model']}",
            cache_name=config.get("cacheName") or "default",
        )
        self._cache_key_fn = config.get("cacheKeyFn")

        self.hooks.emit_sync(
            "onClientCreate",
            {
                "clientId": self.id,
                "provider": self.provider,
                "model": self.model,
                "mode": self.mode,
                "batchable": self.batchable,
            },
        )

    def destroy(self) -> None:
        self.hooks.emit_sync(
            "onClientDestroy",
            {"clientId": self.id, "provider": self.provider, "model": self.model},
        )

    def assistant_message(self, response: Mapping[str, Any]) -> Message:
        """Build an assistant history message from a response, stamped with
        provenance (id, createdAt, origin).

        On a stateful API (responses / interactions) the origin carries the
        server-state id, so a later turn can continue server-side instead of
        resending the transcript. Append the result to your messages between
        turns.
        """
        return build_assistant_message(
            response, {"provider": self.provider, "model": self.model, "api": self.api}
        )

    # -- moderation helpers --------------------------------------------------

    def _resolve_moderation_key(self, mod: Mapping[str, Any]) -> str:
        """The OpenAI key the emulated moderation path needs.

        Reuses the client's own key when the provider is OpenAI; otherwise an
        explicit one is required. Raises when none is resolvable -- moderation is
        report-only, but the call still needs a key to reach the endpoint.
        """
        key = mod.get("apiKey") or (self._api_key if self.provider == "openai" else None)
        if not key:
            raise ValueError(
                "moderation: emulated moderation requires an OpenAI API key (the only "
                "public moderations endpoint). Pass moderation.apiKey, or use the "
                "OpenAI provider."
            )
        return str(key)

    def _moderation_config(self, mod: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "apiKey": self._resolve_moderation_key(mod),
            "model": moderation_model(mod),
            "fetch": self._fetch,
        }

    def _moderation_plan(self, options: Mapping[str, Any]) -> tuple[Any, dict[str, Any] | None]:
        """`(request, config)` for emulated moderation, resolved BEFORE the call.

        The key is resolved up front so a missing one fails fast (the documented
        contract) and never discards a billed completion or skips
        `onCompletion`. The moderation calls themselves run after the response,
        because they need the result text, and reuse this config.
        """
        mod = options.get("moderation")
        if mod and resolve_moderation_mode(self.provider, mod) == "emulate":
            return mod, self._moderation_config(mod)
        return mod, None

    def _stream_moderation_plan(self, options: Mapping[str, Any]) -> dict[str, Any]:
        """The emulated-moderation decisions a stream needs, in one shape.

        Native moderation needs none of this -- the adapter emits `moderation`
        events off the final chunk.
        """
        mod = options.get("moderation")
        emulate = bool(mod) and resolve_moderation_mode(self.provider, mod or {}) == "emulate"
        stream_opts: Mapping[str, Any] = (mod or {}).get("stream") or {}
        return {
            "mod": mod,
            "emulate": emulate,
            "input": bool(mod) and emulate and (mod or {}).get("input", True),
            "output": bool(mod) and emulate and (mod or {}).get("output", True),
            "strategy": stream_opts.get("strategy") or MODERATION_DEFAULT_STRATEGY,
            "interval": stream_opts.get("interval") or MODERATION_DEFAULT_INTERVAL,
        }

    @staticmethod
    def _moderation_report(input_entry: Any, output_entry: Any) -> dict[str, Any]:
        """Assemble the emulated report. Absent halves stay absent."""
        report: dict[str, Any] = {"source": "emulated"}
        if input_entry is not None:
            report["input"] = input_entry
        if output_entry is not None:
            report["output"] = output_entry
        return report

    @staticmethod
    def _merge_moderation(
        prev: Mapping[str, Any] | None,
        phase: str,
        result: Any,
        source: str,
    ) -> dict[str, Any]:
        """Fold a moderation stream event into the accumulating report.

        A report is only extended when the SOURCE matches: a native input result
        and an emulated output one are two different claims, and merging them
        would label whichever arrived first as the source of both.
        """
        nxt: dict[str, Any] = (
            dict(prev) if prev and prev.get("source") == source else {"source": source}
        )
        nxt["input" if phase == "input" else "output"] = result
        return nxt

    # -- file retrieval ------------------------------------------------------

    def _retrieve_context(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "apiKey": self._api_key,
            "fetch": self._fetch,
            "baseURL": self._adapter.base_url(),
        }

    # -- request assembly ----------------------------------------------------

    def _limit_thinking(self, normalized: MutableMapping[str, Any]) -> str | None:
        """Drop `thinking={"mode": "off"}` where the model cannot honour it.

        Dropped rather than sent, because the wire spec turns `off` into a real
        field -- `thinkingBudget: 0` on Gemini 2.5, `thinkingLevel: "MINIMAL"`
        on 3.x -- and the models the catalog flags answer 400 to it: `Budget 0
        is invalid. This model only works in thinking mode.` The caller gets a
        request that works, plus a warning that their instruction could not be
        followed.

        What this replaced was worse than a 400. `off` was accepted, the spec
        emitted nothing at all, and Google applied its own default -- so the
        model reasoned, the caller was told nothing, and only the bill said
        otherwise. Measured before the fix: 387 thought tokens on 2.5-flash and
        684 on 2.5-pro, for requests that had asked for none.

        Asked of the CATALOG, and only an explicit `False` counts: a model
        nobody has measured is not evidence that off is impossible.
        """
        thinking = normalized.get("thinking")
        if not isinstance(thinking, Mapping) or thinking.get("mode") != "off":
            return None
        entry = self.catalog.get(self.provider, self.model) if self.catalog else None
        reasoning = dict(entry).get("reasoning") if entry else None
        if not isinstance(reasoning, Mapping) or reasoning.get("canDisable") is not False:
            return None
        normalized.pop("thinking", None)
        return (
            f"{self.provider}/{self.model} cannot switch reasoning off -- "
            "thinking={'mode': 'off'} was dropped and the model will reason as it defaults to."
        )

    def _note_unsupported_builtins(
        self, normalized: Mapping[str, Any], req: ProviderHttpRequest
    ) -> None:
        """Say when a hosted tool the caller asked for is not going to be sent.

        A provider that cannot run a builtin simply has no mapping for it, so
        the spec omits it and the request goes out without -- correctly. What
        was missing is the telling: `builtin_tools=["code_interpreter"]` against
        OpenRouter produced `tools: []` and not one word, which is the silent
        loss of a capability this library's tool-constraint mechanic exists to
        prevent. Measured live 2026-09-04.

        Asked of the CATALOG rather than the spec: which hosted tools a provider
        runs is a fact about the provider, and the spec layer has no way to know
        it. A model the catalog does not carry is left alone -- an unknown model
        is not evidence that a tool is unsupported.
        """
        wanted = [
            str(tool.get("type"))
            for tool in (normalized.get("tools") or [])
            if isinstance(tool, Mapping) and tool.get("type") and not tool.get("function")
        ]
        if not wanted:
            return
        info = self.catalog.get(self.provider, self.model)
        if info is None:
            return
        capabilities = info.get("capabilities") or {}
        supported = capabilities.get("builtinTools")
        if supported is None:
            return
        for name in wanted:
            if name in supported:
                continue
            note = (
                f"{self.provider} does not run the hosted tool {name!r}, so it was "
                f"not sent. Supported here: {', '.join(supported) or 'none'}."
            )
            if req.notes is None:
                req.notes = []
            if note not in req.notes:
                req.notes.append(note)

    def _attach_build_notes(
        self, result: dict[str, Any], req: ProviderHttpRequest
    ) -> dict[str, Any]:
        """Put the build notes on the result, for `Completion.warnings`.

        The same notes `_report_build_notes` emits as `onWarning`, in the same
        shape. Both, not one: a hook subscriber wants them as they happen, and a
        caller holding only the result must not have to have subscribed. They
        share this shape so the two never describe one adjustment differently.
        """
        if req.notes:
            result["warnings"] = [
                {
                    "source": "llm",
                    "code": "request_adjusted",
                    "message": note,
                    "details": {"provider": self.provider, "model": self.model},
                }
                for note in req.notes
            ]
        return result

    def _report_build_notes(self, req: ProviderHttpRequest, ctx: RequestContext) -> None:
        """Anything the spec left out on purpose reaches the caller as a warning.

        Said once per request; the build already de-duplicates within one.
        """
        for note in req.notes or ():
            self.hooks.emit_sync(
                "onWarning",
                {
                    "source": "llm",
                    "code": "request_adjusted",
                    "message": note,
                    "details": {"provider": self.provider, "model": self.model, "ctx": ctx},
                },
            )

    def _compose_system(self, options: Mapping[str, Any], from_messages: str | None) -> str | None:
        """`options.system`, then any `role='system'` messages, then the client's
        own -- in that priority order, joined by a blank line. Empty strings drop
        out, and an all-empty stack yields None rather than `''`."""
        parts = [
            s
            for s in (options.get("system"), from_messages, self.system)
            if isinstance(s, str) and s
        ]
        return "\n\n".join(parts) or None

    def _normalize(
        self, input_: Any, options: Mapping[str, Any], *, structured: bool
    ) -> dict[str, Any]:
        """Build the NormalizedRequest from fixed config plus per-call options.

        `structured` is the one asymmetry between the two paths and it is the
        TypeScript's: `stream()` does not carry `options.structured`, because a
        streamed structured output has no parse step to enforce it.
        """
        raw_messages = normalize_input(input_)
        # Universal normalization: pull any `role='system'` messages out of the
        # input array and merge them into the top-level system field. Some
        # providers (Anthropic) reject `role='system'` in the messages array;
        # doing it here makes per-call system prompts work across all providers.
        extracted = extract_system(raw_messages)
        info = self.catalog.get(self.provider, self.model)
        normalized: dict[str, Any] = {
            "model": self.model,
            "wireSpec": info.get("wireSpec") if info else None,
            "messages": extracted["messages"],
            "system": self._compose_system(options, extracted.get("system")),
        }
        # ABSENT, not None. TypeScript writes `maxTokens: options.maxTokens` and
        # an unset one becomes `undefined`, which the spec treats as missing and
        # never emits. Python's `None` is a value: filling every unset option
        # with it put `"temperature": null` on the wire, and Anthropic answers
        # `temperature: Input should be a valid number` -- a 400 for a parameter
        # the caller never set. Found by a live call; no recorded test could see
        # it, because those build the request dict themselves.
        for key in _PASSTHROUGH_OPTIONS:
            if options.get(key) is not None:
                normalized[key] = options[key]
        if structured and options.get("structured") is not None:
            normalized["structured"] = options["structured"]
        return normalized

    def _begin(
        self, input_: Any, options: Mapping[str, Any], *, structured: bool
    ) -> tuple[RequestContext, dict[str, Any]]:
        """Everything before the first hook: the context and the normalized request."""
        ctx = build_context(self, options)
        return ctx, self._normalize(input_, options, structured=structured)

    def _resolve_ctx(
        self, normalized: Mapping[str, Any], options: Mapping[str, Any]
    ) -> dict[str, Any]:
        """The mutable context handed to `onMessageResolve` handlers.

        Built here and applied by `_apply_resolve`, so the two clients differ
        only in how they emit the hook between the two.
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "messages": normalized["messages"],
            "system": normalized["system"],
            "history": options.get("history"),
            "abort": None,
            "abortReason": None,
        }

    @staticmethod
    def _apply_resolve(
        resolve_ctx: Mapping[str, Any], normalized: dict[str, Any], what: str
    ) -> None:
        """Honour an abort, then re-anchor to what the handlers left.

        Reading the values back is not optional: a handler that REPLACES the
        message list rather than mutating it is invisible otherwise -- which is
        what a context guard handing back a compacted transcript does.
        """
        if resolve_ctx.get("abort"):
            reason = resolve_ctx.get("abortReason")
            from ..network.errors import RequestAborted

            raise RequestAborted(
                f"{what} aborted by onMessageResolve handler" + (f": {reason}" if reason else "")
            )
        normalized["messages"] = resolve_ctx["messages"]
        normalized["system"] = resolve_ctx["system"]

    def _apply_server_state(self, normalized: dict[str, Any], options: Mapping[str, Any]) -> None:
        """Unless the caller passed an explicit `previousResponseId` (manual
        mode) or opted out (`stateful=False`), decide whether to continue
        server-side (send the id plus only the new turn) or resend full
        history."""
        if normalized.get("previousResponseId") or options.get("stateful") is False:
            return
        decision = resolve_server_state(
            messages=normalized["messages"],
            provider=self.provider,
            model=self.model,
            catalog=self.catalog,
            stateful=True,
            now=time.time() * 1000,
        )
        normalized["previousResponseId"] = decision.get("previousResponseId")
        normalized["messages"] = decision["messages"]

    def _build_submit(
        self, normalized: dict[str, Any], ctx: RequestContext, options: Mapping[str, Any]
    ) -> tuple[ProviderHttpRequest, dict[str, Any], dict[str, Any]]:
        """Server state, the provider request, and the `onBeforeSubmit` context.

        Returns `(provider_req, submit_ctx, stats)`. The caller emits the hook
        with `submit_ctx`, which a cache plugin or batcher may fill in to
        short-circuit the call.
        """
        self._apply_server_state(normalized, options)
        thinking_note = self._limit_thinking(normalized)
        provider_req = self._adapter.build_request(normalized)
        if thinking_note:
            provider_req.notes = [*(provider_req.notes or []), thinking_note]
        self._note_unsupported_builtins(normalized, provider_req)
        self._report_build_notes(provider_req, ctx)

        # Compute cacheKey if a custom builder was provided.
        if self._cache_key_fn:
            ctx["cacheKey"] = ctx.get("cacheKey") or self._cache_key_fn(normalized, ctx)

        submit_ctx: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "clientId": self.id,
            "mode": self.mode,
            "batchable": self.batchable,
            "request": provider_req.body,
            "ctx": ctx,
            "intercepted": False,
            "result": None,
        }
        return provider_req, submit_ctx, self._request_stats(normalized)

    def _http_request(
        self,
        provider_req: ProviderHttpRequest,
        normalized: Mapping[str, Any],
        ctx: RequestContext,
        *,
        stream: bool = False,
    ) -> HttpRequest:
        url = self._adapter.base_url() + (provider_req.path or self._adapter.completion_path())
        req: HttpRequest = {
            "url": url,
            "headers": {**self._adapter.auth_headers(), **(provider_req.headers or {})},
            "body": provider_req.body,
            "timeout": normalized.get("timeout"),
            "signal": normalized.get("signal"),
            "provider": self.provider,
            "model": self.model,
            # Every trace field, not a hand-picked three: `traceparent` rides
            # with the ids, and picking fields here is what left the HTTP spans
            # rooting a trace of their own while the LLM span they belong to had
            # joined the caller's.
            "trace": {
                "sessionId": ctx.get("sessionId"),
                "requestId": ctx.get("requestId"),
                "callId": ctx.get("callId"),
                "traceparent": ctx.get("traceparent"),
            },
        }
        if stream:
            req["stream"] = True
        return req

    def _fetch_options(self, stats: Mapping[str, Any], ctx: RequestContext) -> dict[str, Any]:
        return {
            "queueName": self.routing.queue_name,
            "priority": self._priority,
            "estimatedTokens": stats["estimatedInputTokens"],
            "ctx": ctx,
        }

    def _stream_fetch_options(self, ctx: RequestContext) -> dict[str, Any]:
        """No token estimate: a stream's queue slot is taken before its size is
        known, and passing a made-up number would price the queue on a guess."""
        return {"queueName": self.routing.queue_name, "priority": self._priority, "ctx": ctx}

    @staticmethod
    def _request_stats(normalized: Mapping[str, Any]) -> dict[str, Any]:
        """The `request` block on `onCompletion`, plus the token estimate the
        queue is given. One function so the streamed and buffered paths cannot
        report a call two different ways."""
        input_chars = len(js_json(normalized["messages"]))
        return {
            "inputChars": input_chars,
            # `-(-x // 4)` is `Math.ceil(x / 4)` without importing math for it.
            "estimatedInputTokens": -(-input_chars // 4),
            "messageCount": len(normalized["messages"]),
            "hasTools": len(normalized.get("tools") or ()) > 0,
        }

    @staticmethod
    def _intercepted(submit_ctx: Mapping[str, Any]) -> bool:
        """Whether a hook answered the call itself.

        Both flags: `intercepted` alone with no result would produce an empty
        200 that parses into an empty completion, which is worse than the call
        it replaced.
        """
        return bool(submit_ctx.get("intercepted")) and submit_ctx.get("result") is not None

    @staticmethod
    def _intercepted_response(raw_result: Any) -> HttpResponse:
        return {"status": 200, "headers": {}, "body": raw_result}

    def _finish(self, response: Mapping[str, Any], latency_ms: float) -> dict[str, Any]:
        """Shape-check the body, then parse it.

        Checked BEFORE parsing, so a body the parser silently tolerates is still
        reported.
        """
        if self._shape_checker:
            self._shape_checker.check_response(response.get("body"))
        parsed: dict[str, Any] = self._adapter.parse_response(response.get("body"), latency_ms)
        return parsed

    def _completion_ctx(
        self,
        response: Mapping[str, Any],
        stats: Mapping[str, Any],
        request_body: Any,
        response_body: Any,
        ctx: RequestContext,
    ) -> dict[str, Any]:
        """The `onCompletion` payload. One builder, so a streamed call and a
        buffered one cannot describe themselves differently."""
        return {
            "provider": self.provider,
            "model": self.model,
            "response": response,
            "request": {
                "estimatedInputTokens": stats["estimatedInputTokens"],
                "inputChars": stats["inputChars"],
                "messageCount": stats["messageCount"],
                "hasTools": stats["hasTools"],
            },
            "requestBody": request_body,
            "responseBody": response_body,
            "ctx": ctx,
        }

    # -- streaming -----------------------------------------------------------

    def _require_fetch_stream(self) -> Any:
        """The streaming fetch, or a clear error naming what is missing.

        Returns it rather than only checking, so the caller holds a value the
        type checker can see is present -- the alternative is every stream
        method re-proving it.
        """
        if not self._fetch_stream:
            raise RuntimeError(
                f"{type(self)._label}.stream: no fetchStream function configured"
            )
        return self._fetch_stream

    def _stream_begin(
        self, input_: Any, options: Mapping[str, Any]
    ) -> tuple[RequestContext, dict[str, Any]]:
        self._require_fetch_stream()
        return self._begin(input_, options, structured=False)

    def _stream_request(
        self, normalized: dict[str, Any], ctx: RequestContext
    ) -> tuple[ProviderHttpRequest, HttpRequest]:
        thinking_note = self._limit_thinking(normalized)
        provider_req = self._adapter.build_request(normalized)
        if thinking_note:
            provider_req.notes = [*(provider_req.notes or []), thinking_note]
        self._note_unsupported_builtins(normalized, provider_req)
        self._report_build_notes(provider_req, ctx)
        # `enable_streaming` is OPTIONAL on the adapter protocol, and probed the
        # way the TypeScript writes `adapter.enableStreaming?.(...)`.
        enable = getattr(self._adapter, "enable_streaming", None)
        if enable is not None:
            enable(provider_req, normalized)
        return provider_req, self._http_request(provider_req, normalized, ctx, stream=True)

    def _parse_frames(self, sse_event: Mapping[str, Any], parse: Any) -> list[dict[str, Any]]:
        """One SSE frame -> unified events, shape-checked first.

        Per event, and per event TYPE: an event type the parser does not handle
        is skipped in silence, so the reply is simply missing a piece.
        """
        if self._shape_checker:
            self._shape_checker.check_stream_event(sse_event)
        events: list[dict[str, Any]] = parse(sse_event)
        return events


class StreamAccumulator:
    """What a stream adds up to, so the final `onCompletion` describes the same
    call `complete` would have.

    Its own class rather than eleven locals in `stream`, because every field here
    exists to make a streamed response match a buffered one -- and a field added
    to one and not the other is how "which call style you used" starts changing
    the answer.
    """

    def __init__(self) -> None:
        self.text = ""
        self.thinking = ""
        self.usage: dict[str, Any] = empty_usage()
        self.finish_reason: str = "stop"
        self.moderation: dict[str, Any] | None = None
        self.files: list[Any] = []
        self.builtin_tool_calls: list[Any] = []
        # Deduped by url: Google repeats its grounding chunks on more than one
        # late chunk, and a model that cites one page twice is still one source.
        self.citations_by_url: dict[str, Any] = {}

    def absorb(self, event: Mapping[str, Any], merge: Any) -> None:
        kind = event.get("type")
        if kind == "text":
            self.text += event.get("text") or ""
        elif kind == "thinking":
            self.thinking += event.get("text") or ""
        elif kind == "usage":
            self.usage = event["usage"]
        elif kind == "done":
            self.finish_reason = event["finishReason"]
        elif kind == "file":
            # Hosted-tool output file (a code-execution artifact) -- collected
            # for the final response so streamed `files` matches complete().
            self.files.append(event["file"])
        elif kind == "citation":
            # Same reason as `file`: streamed `citations` must match what
            # complete() returns.
            self.citations_by_url[event["citation"]["url"]] = event["citation"]
        elif kind == "builtin_tool_end":
            # Durable trail of provider-run builtin tools (parity with
            # complete()) -- collected on END, which carries the full payload
            # (code/output/query).
            call: dict[str, Any] = {"tool": event.get("tool")}
            for key in ("id", "code", "output", "query", "url"):
                if event.get(key):
                    call[key] = event[key]
            self.builtin_tool_calls.append(call)
        elif kind == "moderation":
            self.moderation = merge(
                self.moderation, event["phase"], event["result"], event["source"]
            )

    def to_response(self, model: str, latency_ms: float) -> dict[str, Any]:
        response: dict[str, Any] = {
            "id": f"stream_{str(uuid.uuid4())[:12]}",
            "model": model,
            "content": [{"type": "text", "text": self.text}] if self.text else [],
            "finishReason": self.finish_reason,
            "usage": self.usage,
            "text": self.text,
            "toolCalls": [],
            "thinking": self.thinking or None,
            "media": [],
        }
        # The four optional fields stay ABSENT when empty (CONSTITUTION.md R3),
        # exactly as the buffered parse leaves them.
        if self.files:
            response["files"] = self.files
        if self.builtin_tool_calls:
            response["builtinToolCalls"] = self.builtin_tool_calls
        if self.citations_by_url:
            response["citations"] = list(self.citations_by_url.values())
        if self.moderation:
            response["moderation"] = self.moderation
        response["latencyMs"] = latency_ms
        response["raw"] = None
        return response


def repair_turns(text: str, err: Exception) -> list[dict[str, Any]]:
    """The two turns a structured repair appends.

    Shared by both clients so the re-prompt cannot drift between them -- it is
    the instruction that decides whether the retry works, and two copies of it
    would be two prompts under one name.
    """
    return [
        {"role": "assistant", "content": text},
        {
            "role": "user",
            "content": (
                "Your previous reply was not valid JSON for the required "
                f"schema ({err}). Reply with ONLY the JSON object matching "
                "the schema -- no prose, no code fences."
            ),
        },
    ]


__all__ = ["BaseLLMClient", "StreamAccumulator", "now_ms", "repair_turns"]
