"""AsyncLLMClient -- Layer 2, the coroutine core.

Transposed from `unified-library-ts/src/llm/client.ts`, whose single class this
and `LLMClient` (sync) split between them. Everything that does not wait lives in
`client_base.py`; what is left here is the sequence and its `await`s.

Format adapter only. Does NOT own a queue, a retry policy, or a cache. It
receives `fetch` (and optionally `fetch_stream`) as injected functions. The
semantic layer is fixed at construction: `provider`, `model`, `api_key` and
`system` are immutable per instance.

Public methods:

- `await complete(input, options)`      -> CompletionResponse
- `await structured_complete(input, schema, options)` -> the parsed object
- `async for` over `stream(input, options)` -> StreamEvent
- `assistant_message(response)`         -> a history Message with provenance
- `await retrieve_file(file)` / `await stream_file(file)`
- `destroy()`                           -> emits the lifecycle hook

Input shapes (`str | list[ContentPart] | list[Message]`):

- `str`               -> `[{'role': 'user', 'content': input}]`
- `list[ContentPart]` -> `[{'role': 'user', 'content': parts}]`
- `list[Message]`     -> used as the full messages array (REPLACE)

Hooks emitted: `onClientCreate` (in the constructor), `onMessageResolve`,
`onBeforeSubmit`, `onCompletion`, `onWarning`, `onClientDestroy`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

from ..network.types import HttpResponse
from .client_base import BaseLLMClient, StreamAccumulator, now_ms, repair_turns
from .client_internal import normalize_input, parse_structured
from .files.retrieve import FileStream, RetrievedFile, aretrieve_file, astream_file
from .moderation.runner import (
    arun_moderation,
    awrap_moderated_stream,
    emit_moderation_zero_cost,
    moderation_input_text,
)
from .output_errors import InvalidFinalOutputError


class AsyncLLMClient(BaseLLMClient):
    """The async core. `LLM`'s twin is `LLMClient`; neither wraps the other."""

    _label = "AsyncLLMClient"

    # -- file retrieval ------------------------------------------------------

    async def retrieve_file(self, file: Mapping[str, Any]) -> RetrievedFile:
        """Fetch a hosted-tool output file (e.g. one from `response['files']`):
        its bytes plus `name` / `mimeType` / `size`.

        Resolves inline `data`, a `url`, or a provider file `id` -- all through
        this client's provider, auth and engine.
        """
        return await aretrieve_file(file, self._retrieve_context())

    async def stream_file(self, file: Mapping[str, Any]) -> FileStream:
        """Stream a hosted-tool output file: an async iterator of bytes plus
        best-effort `name` / `mimeType` / `size` from the response headers."""
        return await astream_file(file, self._retrieve_context())

    # -- complete ------------------------------------------------------------

    async def complete(
        self, input_: Any, options: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Submit a request. Returns the parsed CompletionResponse."""
        options = options or {}
        ctx, normalized = self._begin(input_, options, structured=True)

        # Let plugins (FilesRegistry, ContextGuard) mutate messages or abort.
        resolve_ctx = self._resolve_ctx(normalized, options)
        await self.hooks.emit("onMessageResolve", resolve_ctx)
        self._apply_resolve(resolve_ctx, normalized, "Request")

        provider_req, submit_ctx, stats = self._build_submit(normalized, ctx, options)
        # A cache plugin or batcher may intercept and short-circuit here.
        await self.hooks.emit("onBeforeSubmit", submit_ctx)

        mod, moderation_cfg = self._moderation_plan(options)
        start = now_ms()

        if self._intercepted(submit_ctx):
            # An interceptor answered. TypeScript hands back a Promise; here it
            # may be a value OR an awaitable, so a synchronous cache hit does not
            # have to fabricate a coroutine.
            raw_result = submit_ctx["result"]
            if hasattr(raw_result, "__await__"):
                raw_result = await raw_result
            response: HttpResponse = self._intercepted_response(raw_result)
        else:
            response = await self._fetch(
                self._http_request(provider_req, normalized, ctx),
                self._fetch_options(stats, ctx),
            )
        latency_ms = now_ms() - start

        result = self._attach_build_notes(self._finish(response, latency_ms), provider_req)

        # Emulated inline moderation (non-OpenAI providers, or forced). Native
        # results are already on `result['moderation']` from the adapter.
        # Report-only: attach, never block.
        if mod and moderation_cfg:
            result["moderation"] = await self._run_emulated_moderation(
                mod, moderation_cfg, normalized, result
            )

        await self.hooks.emit(
            "onCompletion",
            self._completion_ctx(
                result, stats, provider_req.body, response.get("body"), ctx
            ),
        )
        return result

    async def _run_emulated_moderation(
        self,
        mod: Mapping[str, Any],
        cfg: Mapping[str, Any],
        normalized: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Input and output run CONCURRENTLY -- the moderations endpoint is free,
        so there is nothing to save by serialising them and a round-trip to lose.

        This is the one place the two cores differ in more than syntax: the sync
        client runs the same two calls one after the other, because concurrency
        there would mean threads.
        """
        do_input = mod.get("input", True)
        do_output = mod.get("output", True)
        input_entry, output_entry = await asyncio.gather(
            arun_moderation(moderation_input_text(normalized["messages"]), cfg)
            if do_input
            else _none(),
            arun_moderation(result.get("text") or "", cfg) if do_output else _none(),
        )
        if do_input:
            emit_moderation_zero_cost(self.hooks, cfg["model"])
        if do_output:
            emit_moderation_zero_cost(self.hooks, cfg["model"])
        return self._moderation_report(input_entry, output_entry)

    # -- structured ----------------------------------------------------------

    async def structured_complete(
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
            res = await self.complete(messages, {**options, "structured": structured})
            try:
                return parse_structured(res.get("text") or "")
            except InvalidFinalOutputError as err:
                if attempt >= repair_attempts:
                    raise
                messages.extend(repair_turns(res.get("text") or "", err))
                attempt += 1

    # -- stream --------------------------------------------------------------

    async def stream(
        self, input_: Any, options: Mapping[str, Any] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
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
        await self.hooks.emit("onMessageResolve", resolve_ctx)
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

        async def raw_events() -> AsyncIterator[dict[str, Any]]:
            async for sse_event in fetch_stream(http_req, fetch_options):
                for ev in self._parse_frames(sse_event, parse):
                    yield ev

        plan = self._stream_moderation_plan(options)
        event_stream: AsyncIterator[dict[str, Any]] = raw_events()

        if plan["output"]:
            cfg = self._moderation_config(plan["mod"])
            hooks = self.hooks

            async def moderate(text: str) -> dict[str, Any]:
                entry = await arun_moderation(text, cfg)
                emit_moderation_zero_cost(hooks, cfg["model"])
                return entry

            event_stream = awrap_moderated_stream(
                event_stream, plan["strategy"], plan["interval"], moderate
            )

        if plan["input"]:
            cfg = self._moderation_config(plan["mod"])
            entry = await arun_moderation(moderation_input_text(normalized["messages"]), cfg)
            emit_moderation_zero_cost(self.hooks, cfg["model"])
            acc.moderation = self._merge_moderation(acc.moderation, "input", entry, "emulated")
            yield {"type": "moderation", "phase": "input", "result": entry, "source": "emulated"}

        async for event in event_stream:
            acc.absorb(event, self._merge_moderation)
            yield event

        # Normal completion (no raise): emit onCompletion. Aborted or errored
        # streams raise out of the loop above and emit nothing -- a cost is a
        # COMPLETED call.
        await self.hooks.emit(
            "onCompletion",
            self._completion_ctx(
                acc.to_response(self.model, now_ms() - start),
                self._request_stats(normalized),
                provider_req.body,
                None,
                ctx,
            ),
        )


async def _none() -> None:
    """An awaitable that resolves to None -- `Promise.resolve(undefined)`."""


__all__ = ["AsyncLLMClient"]
