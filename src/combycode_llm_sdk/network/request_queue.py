"""A priority queue of pending requests. Lower priority number = more urgent.

Transposed from `unified-library-ts/src/network/request-queue.ts` and
`semaphore.ts`.

The TypeScript hand-rolls a min-heap because JavaScript has none; Python has
`heapq`, so the heap itself is the stdlib's and what is transposed is the
ORDERING and the expiry rule -- ties broken by enqueue time so equal priorities
stay FIFO, and an entry past its deadline rejected rather than run late.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .types import HttpRequest, HttpResponse


def _now_ms() -> float:
    return time.perf_counter() * 1000


@dataclass
class QueueEntry:
    """One pending request and the future its caller is waiting on."""

    id: str
    request: HttpRequest
    priority: int
    enqueued_at: float
    deadline: float
    estimated_tokens: float
    attempt: int
    future: asyncio.Future[HttpResponse] = field(repr=False)


@dataclass(frozen=True)
class QueueConfig:
    #: Max pending requests. A new one beyond this is refused rather than queued
    #: -- an unbounded queue turns a provider outage into memory exhaustion.
    max_size: int = 200
    #: Longest a request may WAIT before being run. Distinct from the request's
    #: own timeout, which bounds the call itself.
    timeout_ms: float = 30_000


class QueueFullError(RuntimeError):
    """The queue is at `max_size` and refusing new work."""


class RequestQueue:
    """A min-heap of entries, ordered by (priority, enqueue time)."""

    def __init__(self, config: QueueConfig | None = None) -> None:
        self._config = config or QueueConfig()
        self._heap: list[tuple[int, int, int, QueueEntry]] = []
        #: A monotonic tiebreaker. Two entries enqueued in the same millisecond
        #: would otherwise compare by their QueueEntry, which is not orderable --
        #: and making it orderable would let a heap reshuffle equal-priority work.
        self._counter = itertools.count()
        self._arrivals: asyncio.Event = asyncio.Event()

    def enqueue(
        self,
        request: HttpRequest,
        priority: int,
        estimated_tokens: float,
        attempt: int = 0,
    ) -> asyncio.Future[HttpResponse]:
        """Add a request; the future resolves when it has been run."""
        if len(self._heap) >= self._config.max_size:
            raise QueueFullError(f"Queue full ({self._config.max_size} pending)")
        now = _now_ms()
        entry = QueueEntry(
            id=str(uuid.uuid4()),
            request=request,
            priority=priority,
            enqueued_at=now,
            deadline=now + self._config.timeout_ms,
            estimated_tokens=estimated_tokens,
            attempt=attempt,
            future=asyncio.get_event_loop().create_future(),
        )
        heapq.heappush(
            self._heap, (priority, int(entry.enqueued_at), next(self._counter), entry)
        )
        self._arrivals.set()
        return entry.future

    def dequeue(self) -> QueueEntry | None:
        """The most urgent entry that has not expired, or None."""
        self._purge_expired()
        if not self._heap:
            return None
        return heapq.heappop(self._heap)[3]

    def peek(self) -> QueueEntry | None:
        self._purge_expired()
        return self._heap[0][3] if self._heap else None

    async def wait_for_item(self) -> None:
        """Block until something is queued. Returns at once when it already is."""
        if self._heap:
            return
        self._arrivals.clear()
        await self._arrivals.wait()

    def __len__(self) -> int:
        return len(self._heap)

    def _purge_expired(self) -> None:
        """Reject everything past its deadline, then re-heapify what is left.

        A request that waited out its deadline is rejected rather than run: the
        caller has by then almost certainly given up, and running it spends a
        rate-limit slot on a result nobody reads.
        """
        now = _now_ms()
        expired = [item[3] for item in self._heap if now > item[3].deadline]
        if not expired:
            return
        self._heap = [item for item in self._heap if now <= item[3].deadline]
        # Filtering breaks the heap invariant, so rebuild it rather than trusting
        # that removing from the middle left a heap behind.
        heapq.heapify(self._heap)
        for entry in expired:
            if not entry.future.done():
                entry.future.set_exception(
                    TimeoutError(
                        f"Request timed out in queue after "
                        f"{round(now - entry.enqueued_at)}ms"
                    )
                )


class Semaphore:
    """Counting semaphore for concurrency control.

    `asyncio.Semaphore` would do, but it exposes no counts -- and `in_flight`,
    `waiting` and `available` are what the queue snapshot reports to metrics.
    """

    def __init__(self, maximum: int) -> None:
        self._max = maximum
        self._current = 0
        self._waiters: list[asyncio.Future[None]] = []

    async def acquire(self) -> None:
        if self._current < self._max:
            self._current += 1
            return
        waiter: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        self._waiters.append(waiter)
        await waiter

    def release(self) -> None:
        self._current -= 1
        while self._waiters:
            waiter = self._waiters.pop(0)
            if not waiter.done():
                # The slot passes straight to the waiter rather than being freed
                # and re-taken, so a third caller cannot cut in between.
                self._current += 1
                waiter.set_result(None)
                return

    @property
    def in_flight(self) -> int:
        return self._current

    @property
    def waiting(self) -> int:
        return len(self._waiters)

    @property
    def available(self) -> int:
        return max(0, self._max - self._current)


def body_size_of(body: Any) -> int:
    """Telemetry body size, null-safe.

    A body-less request (GET/HEAD/DELETE) has no body at all, and serialising
    `None` to measure it would report 4 for the string "null".
    """
    if body is None:
        return 0
    if isinstance(body, (bytes, bytearray)):
        return len(body)
    if isinstance(body, str):
        return len(body)
    from ..wire.interpreter import js_json

    try:
        return len(js_json(body))
    except (TypeError, ValueError):
        return 0


__all__ = [
    "QueueConfig",
    "QueueEntry",
    "QueueFullError",
    "RequestQueue",
    "Semaphore",
    "body_size_of",
]
