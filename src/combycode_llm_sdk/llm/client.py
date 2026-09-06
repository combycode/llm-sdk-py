"""LLMClient -- Layer 2, the synchronous core.

Transposed from `unified-library-ts/src/llm/client.ts`, whose single class this
and `AsyncLLMClient` split between them. Everything that does not wait lives in
`client_base.py`; what is left here is the sequence.

Read this file beside `async_client.py`: `complete` is the same list of calls in
the same order, and the only differences are the five points where the async one
says `await`. That is deliberate and it is checked -- `test_client_parity.py`
runs both over identical inputs and asserts identical output, so the two cannot
drift into two behaviours under one name.

Format adapter only. Does NOT own a queue, a retry policy, or a cache. It
receives `fetch` (and optionally `fetch_stream`) as injected functions, and both
are ORDINARY callables here: a sync client given an async fetch will get a
coroutine object where it expected a response, which is a mistake worth making
loudly rather than papering over with a loop.

Public methods:

- `complete(input, options)`      -> CompletionResponse
- `structured_complete(input, schema, options)` -> the parsed object
- `for` over `stream(input, options)` -> StreamEvent
- `assistant_message(response)`   -> a history Message with provenance
- `retrieve_file(file)` / `stream_file(file)`
- `destroy()`                     -> emits the lifecycle hook

Hooks reach handlers through `HookBus.emit_sync` on this path. Synchronous
handlers run exactly as they do on the async client; an `async def` handler
cannot be awaited from here and is reported rather than silently skipped -- see
`HookBus.emit_sync`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from ..network.types import HttpResponse
from .client_base import BaseLLMClient, StreamAccumulator, now_ms, repair_turns
from .client_internal import normalize_input, parse_structured
from .files.retrieve import FileStream, RetrievedFile, retrieve_file, stream_file
from .moderation.runner import (
    emit_moderation_zero_cost,
    moderation_input_text,
    run_moderation,
    wrap_moderated_stream_sync,
)
from .output_errors import InvalidFinalOutputError


class LLMClient(BaseLLMClient):
    """The sync core. Its twin is `AsyncLLMClient`; neither wraps the other."""

    _label = "LLMClient"

    # -- file retrieval ------------------------------------------------------

    def retrieve_file(self, file: Mapping[str, Any]) -> RetrievedFile:
        """Fetch a hosted-tool output file (e.g. one from `response['files']`):
        its bytes plus `name` / `mimeType` / `size`.

        Resolves inline `data`, a `url`, or a provider file `id` -- all through
        this client's provider, auth and engine.
        """
        return retrieve_file(file, self._retrieve_context())

    def stream_file(self, file: Mapping[str, Any]) -> FileStream:
        """Stream a hosted-tool output file: an iterator of bytes plus
        best-effort `name` / `mimeType` / `size` from the response headers."""
        return stream_file(file, self._retrieve_context())

    # -- complete ------------------------------------------------------------

    def complete(self, input_: Any, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Submit a request. Returns the parsed CompletionResponse."""
        options = options or {}
        ctx, normalized = self._begin(input_, options, structured=True)

        # Let plugins (FilesRegistry, ContextGuard) mutate messages or abort.
        resolve_ctx = self._resolve_ctx(normalized, options)
        self.hooks.emit_sync("onMessageResolve", resolve_ctx)
        self._apply_resolve(resolve_ctx, normalized, "Request")

        provider_req, submit_ctx, stats = self._build_submit(normalized, ctx, options)
        # A cache plugin or batcher may intercept and short-circuit here.
        self.hooks.emit_sync("onBeforeSubmit", submit_ctx)

        mod, moderation_cfg = self._moderation_plan(options)
        start = now_ms()

        if self._intercepted(submit_ctx):
            response: HttpResponse = self._intercepted_response(submit_ctx["result"])
        else:
            response = self._fetch(
                self._http_request(provider_req, normalized, ctx),
                self._fetch_options(stats, ctx),
            )
        latency_ms = now_ms() - start

        result = self._attach_build_notes(self._finish(response, latency_ms), provider_req)

        # Emulated inline moderation (non-OpenAI providers, or forced). Native
        # results are already on `result['moderation']` from the adapter.
        # Report-only: attach, never block.
        if mod and moderation_cfg:
            result["moderation"] = self._run_emulated_moderation(
                mod, moderation_cfg, normalized, result
            )

        self.hooks.emit_sync(
            "onCompletion",
            self._completion_ctx(result, stats, provider_req.body, response.get("body"), ctx),
        )
        return result

    def _run_emulated_moderation(
        self,
        mod: Mapping[str, Any],
        cfg: Mapping[str, Any],
        normalized: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Input and output run one after the other.

        The async client runs them concurrently, because the moderations
        endpoint is free and there is a round-trip to save. Doing the same here
        would mean threads, which the API contract rules out -- so this path is
        one round-trip slower and identical in every observable way. That is the
        only behavioural difference between the two cores, and the parity test
        compares their OUTPUT, which this does not change.
        """
        do_input = mod.get("input", True)
        do_output = mod.get("output", True)
        input_entry = (
            run_moderation(moderation_input_text(normalized["messages"]), cfg)
            if do_input
            else None
        )
        output_entry = run_moderation(result.get("text") or "", cfg) if do_output else None
        if do_input:
            emit_moderation_zero_cost(self.hooks, cfg["model"])
        if do_output:
            emit_moderation_zero_cost(self.hooks, cfg["model"])
        return self._moderation_report(input_entry, output_entry)

    # -- structured ----------------------------------------------------------

    def structured_complete(
        self,
        input_: Any,
        schema: Mapping[str, Any],
        options: Mapping[str, Any] | None = None,
    ) -> Any:
        """Run `complete` with a JSON Schema enforced via `structured`.

        Strips any leading/trailing markdown fences from the model reply, then
        parses it. Raises `InvalidFinalOutputError` if the parse fails -- callers
        should catch and retry, or opt into `structured.repairAttempts`.
        """
        options = options or {}
        structured = {**(options.get("structured") or {}), "schema": schema}
        repair_attempts = max(0, int(structured.get("repairAttempts") or 0))
        messages = normalize_input(input_)

        # Attempt 0 = the original call; each further attempt appends the parse
        # error and re-prompts (opt-in `repairAttempts`, default 0). The typed
        # InvalidFinalOutputError is re-raised once repairs are exhausted.
        attempt = 0
        while True:
            res = self.complete(messages, {**options, "structured": structured})
            try:
                return parse_structured(res.get("text") or "")
            except InvalidFinalOutputError as err:
                if attempt >= repair_attempts:
                    raise
                messages.extend(repair_turns(res.get("text") or "", err))
                attempt += 1

    # -- stream --------------------------------------------------------------

    def stream(
        self, input_: Any, options: Mapping[str, Any] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Stream a completion, yielding unified events as they arrive.

        Accumulates the stream so a single `onCompletion` fires at the end (as
        `complete` does), which is what lets CostCollector and ContextMeasurer
        price and measure streamed calls too. Usage arrives once near the end
        (OpenAI's `include_usage` final chunk, Anthropic `message_delta`, Google
        `usageMetadata`).
        """
        options = options or {}
        ctx, normalized = self._stream_begin(input_, options)

        resolve_ctx = self._resolve_ctx(normalized, options)
        self.hooks.emit_sync("onMessageResolve", resolve_ctx)
        self._apply_resolve(resolve_ctx, normalized, "Stream")

        provider_req, http_req = self._stream_request(normalized, ctx)

        start = now_ms()
        acc = StreamAccumulator()

        # One parser instance per stream -- it holds any per-stream state (e.g.
        # Google's code-execution latch) in its closure, isolated from concurrent
        # streams.
        parse = self._adapter.create_stream_parser()
        fetch_stream = self._require_fetch_stream()
        fetch_options = self._stream_fetch_options(ctx)

        def raw_events() -> Iterator[dict[str, Any]]:
            for sse_event in fetch_stream(http_req, fetch_options):
                yield from self._parse_frames(sse_event, parse)

        plan = self._stream_moderation_plan(options)
        event_stream: Iterator[dict[str, Any]] = raw_events()

        if plan["output"]:
            cfg = self._moderation_config(plan["mod"])
            hooks = self.hooks

            def moderate(text: str) -> dict[str, Any]:
                entry = run_moderation(text, cfg)
                emit_moderation_zero_cost(hooks, cfg["model"])
                return entry

            event_stream = wrap_moderated_stream_sync(
                event_stream, plan["strategy"], plan["interval"], moderate
            )

        if plan["input"]:
            cfg = self._moderation_config(plan["mod"])
            entry = run_moderation(moderation_input_text(normalized["messages"]), cfg)
            emit_moderation_zero_cost(self.hooks, cfg["model"])
            acc.moderation = self._merge_moderation(acc.moderation, "input", entry, "emulated")
            yield {"type": "moderation", "phase": "input", "result": entry, "source": "emulated"}

        for event in event_stream:
            acc.absorb(event, self._merge_moderation)
            yield event

        # Normal completion (no raise): emit onCompletion. Aborted or errored
        # streams raise out of the loop above and emit nothing -- a cost is a
        # COMPLETED call.
        self.hooks.emit_sync(
            "onCompletion",
            self._completion_ctx(
                acc.to_response(self.model, now_ms() - start),
                self._request_stats(normalized),
                provider_req.body,
                None,
                ctx,
            ),
        )


__all__ = ["LLMClient"]
