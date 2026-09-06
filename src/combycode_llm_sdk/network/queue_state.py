"""One queue: its rate limiter, its concurrency slots, its retry policy.

Transposed from `unified-library-ts/src/network/queue-state.ts`.

This is the per-queue body `NetworkEngine` creates lazily, one per `queueName`
(`anthropic/claude-haiku-4.5`, `openai/gpt-5.4-nano`, `shared/cheap`). It knows
nothing about LLMs -- the semantic layer decides the name and fills in
`provider`/`model` for the hooks.

Every decision about WHETHER to retry, and for how long, comes from
`network/retry.py`, which the synchronous `RequestExecutor` uses too. What lives
here is the queueing: waiting for a rate-limit slot, holding a concurrency
permit, and running a loop that drains the heap.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from ..bus.hook_bus import HookBus
from .errors import LLMError, NetworkError, classify_error
from .rate_limiter import RateLimiter, RateLimiterConfig
from .request_queue import (
    QueueConfig,
    QueueEntry,
    QueueFullError,
    RequestQueue,
    Semaphore,
    body_size_of,
)
from .retry import RetryConfig, calculate_backoff, honored_retry_after_ms, should_retry
from .types import HttpRequest, HttpResponse, SSEEvent

#: Lower is more urgent. A RETRY outranks new work: it already holds a slice of
#: the caller's total budget, and making it queue behind fresh requests is how a
#: retry sequence times out without ever being attempted twice.
PRIORITY_RETRY = 0
PRIORITY_INTERACTIVE = 1
PRIORITY_BACKGROUND = 2
PRIORITY_LOW = 3


def _now_ms() -> float:
    return time.perf_counter() * 1000


def _replayable(body: Any) -> bool:
    return not (hasattr(body, "__next__") or hasattr(body, "__anext__"))


@dataclass
class QueueStateConfig:
    queue_name: str
    fetch: Callable[..., Any]
    hooks: HookBus
    limits: RateLimiterConfig
    retry: RetryConfig
    queue: QueueConfig | None = None


class QueueState:
    """`class QueueState` (queue-state.ts:36)."""

    def __init__(self, config: QueueStateConfig) -> None:
        self.queue_name = config.queue_name
        self._fetch = config.fetch
        self._hooks = config.hooks
        self._retry = config.retry
        self._limiter = RateLimiter(config.limits)
        self._semaphore = Semaphore(config.limits.concurrent)
        self._queue = RequestQueue(config.queue)
        self._running = False
        self._worker: asyncio.Task[None] | None = None
        #: Lifetime round-trips, so an idle queue still shows past activity where
        #: depth and in-flight read 0 between bursts.
        self._processed = 0
        self._peak_depth = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "queueName": self.queue_name,
            "depth": len(self._queue),
            "inFlight": self._semaphore.in_flight,
            "waiting": self._semaphore.waiting,
            "rateLimitWaitMs": self._limiter.wait_time_ms(1),
            "running": self._running,
            "processed": self._processed,
            "peakDepth": self._peak_depth,
        }

    # -- submission ----------------------------------------------------------

    async def submit(
        self,
        req: HttpRequest,
        priority: int = PRIORITY_INTERACTIVE,
        estimated_tokens: float = 0,
    ) -> HttpResponse:
        """Queue a request and wait for its result."""
        depth = len(self._queue) + 1
        self._peak_depth = max(self._peak_depth, depth)
        self._hooks.emit_sync(
            "onEnqueue",
            {
                "provider": req.get("provider"),
                "model": req.get("model"),
                "queueName": self.queue_name,
                "priority": priority,
                "queueLength": depth,
                "estimatedTokens": estimated_tokens,
                "trace": req.get("trace"),
            },
        )
        future = self._queue.enqueue(req, priority, estimated_tokens, 0)
        self._ensure_processing()
        return await future

    async def submit_stream(
        self,
        req: HttpRequest,
        priority: int = PRIORITY_INTERACTIVE,
        estimated_tokens: float = 0,
    ) -> AsyncIterator[SSEEvent]:
        """Run a streaming request, yielding SSE events.

        Streams do NOT go through the heap. A queued request is one the loop can
        run and settle; a stream is a connection the CALLER holds open and
        consumes at its own pace, so parking it in a queue would mean the loop
        waiting on the caller. It still takes a rate-limit slot and a concurrency
        permit -- the parts that protect the provider -- and releases the permit
        when the caller is done.
        """
        await self._wait_for_capacity(estimated_tokens)
        await self._semaphore.acquire()

        self._hooks.emit_sync(
            "onRequestStart",
            {
                "provider": req.get("provider"),
                "model": req.get("model"),
                "queueName": self.queue_name,
                "url": req.get("url"),
                "method": req.get("method") or "POST",
                "bodySize": body_size_of(req.get("body")),
                "attempt": 0,
                "idempotencyKey": str(uuid.uuid4()),
                "streaming": True,
                "trace": req.get("trace"),
            },
        )
        started = _now_ms()
        try:
            send_stream = req.pop("send", None) or self._fetch
            async for event in send_stream(req, None):
                yield event
        finally:
            # Released in a `finally` because a consumer that breaks out of the
            # loop early -- an abort, or enough of the answer -- must not leak
            # the permit and shrink the queue's concurrency for good.
            self._semaphore.release()
            self._processed += 1
            self._hooks.emit_sync(
                "onRequestComplete",
                {
                    "provider": req.get("provider"),
                    "model": req.get("model"),
                    "queueName": self.queue_name,
                    "status": 200,
                    "attempt": 0,
                    "latencyMs": _now_ms() - started,
                    "streaming": True,
                    "trace": req.get("trace"),
                },
            )

    # -- the loop ------------------------------------------------------------

    def _ensure_processing(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker = asyncio.ensure_future(self._process_loop())

    async def _process_loop(self) -> None:
        """Drain the heap until it is empty, then stop.

        Stopping rather than idling: a task parked on an event keeps the loop
        alive, which stops an `asyncio.run` from ever returning. `_ensure_processing`
        starts a new one on the next submit.
        """
        try:
            while True:
                entry = self._queue.dequeue()
                if entry is None:
                    return
                await self._wait_for_capacity(entry.estimated_tokens)
                await self._semaphore.acquire()
                asyncio.ensure_future(self._run(entry))
        finally:
            self._running = False

    async def _run(self, entry: QueueEntry) -> None:
        """One attempt, and whatever follows from how it went."""
        try:
            await self._execute(entry)
        except BaseException as exc:
            # Last-resort net. `_execute` settles its own future, so reaching
            # here means something unforeseen -- and an unsettled future is a
            # caller awaiting for ever, which is worse than any exception.
            if not entry.future.done():
                entry.future.set_exception(exc)
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise
        finally:
            self._semaphore.release()
            # A slot just freed may be what the next entry was waiting for.
            self._ensure_processing()

    async def _execute(self, entry: QueueEntry) -> None:
        req = entry.request
        started = _now_ms()
        self._hooks.emit_sync(
            "onRequestStart",
            {
                "provider": req.get("provider"),
                "model": req.get("model"),
                "queueName": self.queue_name,
                "url": req.get("url"),
                "method": req.get("method") or "POST",
                "bodySize": body_size_of(req.get("body")),
                "attempt": entry.attempt,
                "streaming": False,
                "trace": req.get("trace"),
            },
        )
        try:
            response = await self._attempt(req)
        except LLMError as error:
            self._on_error(entry, error, started)
            return

        headers = response.get("headers") or {}
        self._limiter.update_from_headers(headers)
        self._processed += 1
        self._hooks.emit_sync(
            "onRequestComplete",
            {
                "provider": req.get("provider"),
                "model": req.get("model"),
                "queueName": self.queue_name,
                "status": response.get("status"),
                "attempt": entry.attempt,
                "latencyMs": _now_ms() - started,
                "streaming": False,
                "trace": req.get("trace"),
            },
        )
        if not entry.future.done():
            entry.future.set_result(response)

    async def _attempt(self, req: HttpRequest) -> HttpResponse:
        try:
            # A client may bring its own transport while still sharing this
            # queue -- a stub in a test, a proxy for one tenant. The queue's
            # admission, rate limiting and retry are the shared part; who puts
            # the bytes on the wire is not. Popped so it never reaches HTTP.
            send = req.pop("send", None) or self._fetch
            response: HttpResponse = await send(req, None)
        except LLMError:
            raise
        except Exception as exc:
            raise NetworkError(
                str(exc) or type(exc).__name__,
                provider=req.get("provider") or "",
                model=req.get("model") or "",
                retryable=True,
                raw=exc,
            ) from exc

        status = int(response.get("status") or 0)
        if status >= 400:
            raise classify_error(
                req.get("provider") or "",
                status,
                response.get("body"),
                response.get("headers") or {},
                model=req.get("model") or "",
            )
        return response

    def _on_error(self, entry: QueueEntry, error: LLMError, started: float) -> None:
        """Decide, announce, and either re-queue or fail.

        The decision comes from `retry.should_retry` -- the same call the
        synchronous executor makes -- so the two paths cannot disagree about
        what is worth another attempt.
        """
        req = entry.request
        willing = should_retry(
            error,
            entry.attempt,
            self._retry,
            elapsed_ms=_now_ms() - started,
            request_max_retries=_request_max_retries(req),
            replayable_body=_replayable(req.get("body")),
        )
        self._hooks.emit_sync(
            "onModelError",
            {
                "provider": req.get("provider"),
                "model": req.get("model"),
                "queueName": self.queue_name,
                "error": error,
                "headers": {},
                "attempt": entry.attempt,
                "willRetry": willing,
                "trace": req.get("trace"),
            },
        )

        if error.kind == "rate_limit":
            # Pause the whole queue, not just this request: a 429 is about the
            # key, so letting the next queued request straight through would
            # earn another one.
            honored = honored_retry_after_ms(error, self._retry)
            if honored:
                self._limiter.pause(honored)
            self._hooks.emit_sync(
                "onRateLimitHit",
                {
                    "provider": req.get("provider"),
                    "model": req.get("model"),
                    "queueName": self.queue_name,
                    "retryAfterMs": error.retry_after_ms,
                    "trace": req.get("trace"),
                },
            )

        if not willing:
            if not entry.future.done():
                entry.future.set_exception(error)
            return

        backoff = calculate_backoff(entry.attempt, error, self._retry)
        self._hooks.emit_sync(
            "onRetry",
            {
                "provider": req.get("provider"),
                "model": req.get("model"),
                "queueName": self.queue_name,
                "attempt": entry.attempt + 1,
                "backoffMs": backoff,
                "reason": error.kind,
                "trace": req.get("trace"),
            },
        )
        asyncio.ensure_future(self._requeue_after(entry, backoff))

    async def _requeue_after(self, entry: QueueEntry, backoff_ms: float) -> None:
        """Wait out the backoff, then put the request back at RETRY priority.

        The caller's future is carried over rather than replaced, so a retry is
        invisible to whoever is awaiting the original submit.
        """
        await asyncio.sleep(backoff_ms / 1000)
        try:
            future = self._queue.enqueue(
                entry.request, PRIORITY_RETRY, entry.estimated_tokens, entry.attempt + 1
            )
        except QueueFullError as exc:
            # The queue filled up while this request was backing off. Failing it
            # is the honest outcome: it cannot be re-queued and pretending
            # otherwise would leave the caller waiting on a future nothing owns.
            if not entry.future.done():
                entry.future.set_exception(exc)
            return
        future.add_done_callback(lambda done: _forward(done, entry.future))
        self._ensure_processing()

    async def _wait_for_capacity(self, estimated_tokens: float) -> None:
        """Sleep until the rate limiter will take this request.

        A loop rather than one sleep: another request may consume the slot this
        one waited for, and waking to find it gone must mean waiting again, not
        proceeding anyway.
        """
        while not self._limiter.can_proceed(estimated_tokens):
            wait_ms = self._limiter.wait_time_ms(estimated_tokens)
            self._hooks.emit_sync(
                "onRateLimitUpdate",
                {
                    "queueName": self.queue_name,
                    "waitMs": wait_ms,
                    "rpmAvailable": self._limiter.rpm_available,
                    "tpmAvailable": self._limiter.tpm_available,
                },
            )
            await asyncio.sleep(max(wait_ms, 1) / 1000)


def _forward(done: asyncio.Future[HttpResponse], target: asyncio.Future[HttpResponse]) -> None:
    if target.done():
        return
    error = done.exception()
    if error is not None:
        target.set_exception(error)
    else:
        target.set_result(done.result())


def _request_max_retries(req: HttpRequest) -> int | None:
    override = req.get("retry")
    if override is None:
        return None
    if isinstance(override, dict):
        value = override.get("max_retries", override.get("maxRetries"))
    else:
        value = getattr(override, "max_retries", None)
    return int(value) if value is not None else None


__all__ = [
    "PRIORITY_BACKGROUND",
    "PRIORITY_INTERACTIVE",
    "PRIORITY_LOW",
    "PRIORITY_RETRY",
    "QueueState",
    "QueueStateConfig",
]
