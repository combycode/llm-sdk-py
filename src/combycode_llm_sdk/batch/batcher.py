"""Calls that arrive one at a time, submitted as one job.

Batch pricing is roughly half, and the cost is latency -- so the only callers
who should be batching are the ones who can wait. What `AutoBatcher` adds is
that they do not have to KNOW about each other: unrelated callers each hand it
one item, and when the collection window closes the whole set goes as a single
provider job.

Two decisions carry this file:

- **No thread and no timer.** The window is checked whenever the batcher is
  touched -- `add()`, `tick()`, `wait()`. A background thread would be one this
  library started in every importing process, unstoppable by a caller who does
  not know it exists and untestable without sleeping.
- **Results are correlated by ID, never by position.** Providers reorder;
  Anthropic measurably returns a two-item batch back to front. Pairing by
  position would hand each caller the other's answer, and both answers would
  look perfectly correct.

Transposed from `unified-library-ts/src/plugins/batch/batcher.ts`, whose
`Batcher` intercepts `onBeforeSubmit` and settles Promises. The Python face is
explicit instead -- a caller `add()`s an item and holds a ticket -- because
there is no promise to intercept and a ticket is a thing you can look at.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any, cast

from ..catalog.catalog import resolve_catalog
from ..helpers.llm import default_adapter_factory
from ..llm.client_internal import resolve_api
from ..network.errors import LLMError
from ..results import Completion, build_cost
from .adapters import BATCH_ADAPTERS
from .types import (
    TERMINAL,
    BatchRequest,
    BatchResult,
    BatchStatus,
    BatchStrategy,
    PendingBatchJob,
)

#: Which adapter serves which provider. A provider absent here hosts no batch
#: API at all -- OpenRouter fronts other providers' models but has none of its
#: own -- and `AutoBatcher` says so by name rather than failing later.
ADAPTERS: dict[str, Any] = dict(BATCH_ADAPTERS)


class BatchTicket:
    """One caller's claim on one answer.

    Holds no answer until the batch is read, and REFUSES rather than inventing
    one. An empty completion here would read as a model that declined to answer,
    which is a different and much quieter kind of wrong.
    """

    __slots__ = ("_completion", "_error", "custom_id", "item")

    def __init__(self, custom_id: str, item: Mapping[str, Any]) -> None:
        self.custom_id = custom_id
        self.item = dict(item)
        self._completion: Completion | None = None
        self._error: str | None = None

    @property
    def settled(self) -> bool:
        return self._completion is not None or self._error is not None

    @property
    def failed(self) -> bool:
        return self._error is not None

    def result(self) -> Completion:
        """The answer, or a refusal naming why there is not one yet."""
        if self._completion is not None:
            return self._completion
        if self._error is not None:
            raise LLMError(f"batch item {self.custom_id} failed: {self._error}")
        raise LLMError(
            f"batch item {self.custom_id} has no answer yet. The batch has not been "
            "submitted, or has not finished -- call wait() first."
        )

    def _settle(self, completion: Completion) -> None:
        self._completion = completion

    def _fail(self, error: str) -> None:
        self._error = error

    def __repr__(self) -> str:
        state = "failed" if self.failed else ("ready" if self.settled else "waiting")
        return f"<BatchTicket {self.custom_id} {state}>"


def _monotonic() -> float:
    """Seconds. The window is a duration, so it is measured on a clock that
    only goes forwards -- not on wall time, which a clock change can move."""
    return time.monotonic()


class AutoBatcher:
    """Collects items and submits them as one provider batch job."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        strategy: BatchStrategy | None = None,
        clock: Callable[[], float] = _monotonic,
        transport: Any = None,
        engine: Any = None,
        catalog: Any = None,
        persistence: Any = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
        **options: Any,
    ) -> None:
        provider, _, model_name = model.partition("/")
        if not model_name:
            raise ValueError(
                f"AutoBatcher: name the provider, as 'provider/model' -- got {model!r}"
            )
        adapter_cls = ADAPTERS.get(provider)
        if adapter_cls is None:
            raise ValueError(
                f"AutoBatcher: {provider!r} hosts no batch API. "
                f"Available: {', '.join(sorted(ADAPTERS))}."
            )

        self.provider = provider
        #: One catalog, used to resolve the model id AND to price the result, so
        #: the two cannot disagree about which model this is.
        self._catalog = resolve_catalog(
            catalog if catalog is not None else (engine.catalog if engine else None)
        )
        #: The id the PROVIDER answers to, not our unified alias. Resolved here,
        #: once, rather than per item: a batch submitted under an alias fails
        #: every item with `not_found_error` -- and fails ninety seconds later,
        #: which is the worst place to find out. Every other entry point in this
        #: library resolves through the catalog; this one did not, and only a
        #: live job showed it.
        self.model = self._catalog.resolve_model_id(provider, model_name)
        self.strategy = strategy or BatchStrategy()
        self._clock = clock
        self._store = persistence
        self._max_tokens = max_tokens
        self._system = system
        self._options = options

        key = api_key or (engine.api_keys.get(provider) if engine is not None else None)
        if not key:
            raise ValueError(
                f'AutoBatcher: no API key for provider "{provider}". Pass api_key= '
                "or an engine that has one."
            )
        built: dict[str, Any] = {"api_key": key}
        if base_url:
            built["base_url"] = base_url
        # Google's batch URL names the model, so its adapter cannot be built
        # without one. Asked of the adapter rather than branched on the
        # provider name, so a fifth provider with the same shape needs no
        # change here.
        if getattr(adapter_cls, "needs_model", False):
            built["model"] = self.model
        self._adapter = adapter_cls(**built)
        # Which API surface a batched item is written for. Read from the
        # library's own per-provider default rather than a table here, so a
        # batched request cannot drift from the one `complete()` would send:
        # anthropic messages, openai and xai responses, google generate. The
        # xAI batch wire agrees -- its request is a tagged union whose
        # variant key is literally `responses`.
        self.api = resolve_api(cast("Any", provider))
        self._completion = default_adapter_factory()(provider, key, self.api, base_url)
        self._fetch = self._resolve_fetch(engine, transport)

        self._collecting: list[BatchTicket] = []
        self._opened_at: float | None = None
        self._job: PendingBatchJob | None = None
        self._tickets: dict[str, BatchTicket] = {}

    # -- state ---------------------------------------------------------------

    @property
    def pending(self) -> int:
        """How many items are collected and not yet submitted."""
        return len(self._collecting)

    @property
    def batch_id(self) -> str | None:
        """The submitted job's id, once there is one."""
        return self._job.batch_id if self._job is not None else None

    @property
    def submitted(self) -> bool:
        return self._job is not None

    # -- collecting ----------------------------------------------------------

    def add(self, item: Mapping[str, Any], *, custom_id: str | None = None) -> BatchTicket:
        """Take one caller's item and hand back its ticket.

        `{"prompt": "..."}` is the short form; `{"messages": [...]}` is the long
        one. Per-item `max_tokens`, `system` and `temperature` override the
        batcher's own.

        `custom_id` names the item yourself, for a caller who has an id of their
        own to correlate with -- a row key, a request id. It must be unique
        within the batch, and a collision is refused rather than silently
        overwriting the earlier ticket, which would leave one caller holding a
        ticket that can never settle.
        """
        chosen = custom_id or item.get("custom_id") or f"req_{uuid.uuid4().hex[:12]}"
        if chosen in self._tickets:
            raise ValueError(
                f"AutoBatcher: custom_id {chosen!r} is already in this batch. "
                "Ids correlate answers to callers, so they must be unique."
            )
        ticket = BatchTicket(str(chosen), item)
        self._collecting.append(ticket)
        self._tickets[ticket.custom_id] = ticket
        if self._opened_at is None:
            self._opened_at = self._clock()
        # A full batch goes now: waiting out the window past `max_batch_size`
        # only delays a job that cannot grow.
        if len(self._collecting) >= self.strategy.max_batch_size:
            self.tick()
        return ticket

    def tick(self) -> str | None:
        """Submit if the window says so. Returns the batch id, or None.

        The whole scheduling mechanism: a caller who touches the batcher gives
        it the chance to notice its window has closed, and a caller who does not
        touch it has nothing waiting on a timer.
        """
        if not self._collecting or self._opened_at is None:
            return None
        waited = self._clock() - self._opened_at
        if not self.strategy.should_submit(len(self._collecting), waited):
            return None
        return self.submit()

    def submit(self) -> str:
        """Submit what is collected now, whatever the window says."""
        if not self._collecting:
            raise LLMError("AutoBatcher: nothing to submit")
        batch = list(self._collecting)
        requests = [BatchRequest(t.custom_id, self._body_for(t.item)) for t in batch]
        batch_id: str = self._adapter.submit(requests, self._fetch)
        self._collecting.clear()
        self._opened_at = None
        self._job = PendingBatchJob(
            batch_id=batch_id,
            provider=self.provider,
            created_at=self._clock(),
            custom_ids=[t.custom_id for t in batch],
        )
        if self._store is not None:
            self._store.set(f"batch:{batch_id}", self._job.as_row())
        return batch_id

    # -- reading -------------------------------------------------------------

    def status(self) -> BatchStatus:
        """Ask the provider how far along the job is."""
        if self._job is None:
            raise LLMError("AutoBatcher: nothing has been submitted")
        state: BatchStatus = self._adapter.get_status(self._job.batch_id, self._fetch)
        return state

    def collect(self) -> int:
        """Read the finished job and settle every ticket. Returns how many.

        Settled BY ID. Providers reorder, so position is not a correlation --
        it is a coincidence that usually holds.
        """
        if self._job is None:
            raise LLMError("AutoBatcher: nothing has been submitted")
        results = self._adapter.get_results(self._job.batch_id, self._fetch)
        settled = 0
        for result in results:
            ticket = self._tickets.get(result.custom_id)
            if ticket is None:
                # An id we never sent. Dropped rather than guessed at: matching
                # it to a waiting ticket is exactly the mistake this design is
                # built to avoid.
                continue
            self._settle(ticket, result)
            settled += 1
        if self._store is not None:
            self._store.delete(f"batch:{self._job.batch_id}")
        return settled

    def wait(self, *, poll_seconds: float | None = None, timeout_seconds: float = 300.0) -> int:
        """Submit if needed, poll until the job ends, then settle every ticket.

        The timeout is measured on REAL time, not the injected clock: this call
        genuinely blocks, so what a caller means by "give up after five seconds"
        is five seconds of their life. The injected clock governs the collection
        window, which is a different question.
        """
        if self._job is None:
            self.submit()
        interval = self.strategy.poll_seconds if poll_seconds is None else poll_seconds
        deadline = time.monotonic() + timeout_seconds
        while True:
            state = self.status()
            if state.status in TERMINAL:
                return self.collect()
            if time.monotonic() >= deadline:
                raise LLMError(
                    f"batch {self.batch_id} did not finish within {timeout_seconds:g}s "
                    f"(last status: {state.status})"
                )
            if interval > 0:
                time.sleep(interval)

    def cancel(self) -> None:
        if self._job is not None:
            self._adapter.cancel(self._job.batch_id, self._fetch)

    def tickets(self) -> list[BatchTicket]:
        return list(self._tickets.values())

    # -- internal ------------------------------------------------------------

    def _settle(self, ticket: BatchTicket, result: BatchResult) -> None:
        if not result.success or result.response is None:
            ticket._fail(result.error or "the provider reported no result")
            return
        # A batch result IS an ordinary completion response -- the batch API
        # only changed how it was delivered -- so it is read by the same
        # parser that reads a live one, and a provider that changes its
        # response shape changes both at once.
        parsed = self._completion.parse_response(dict(result.response), 0.0)
        ticket._settle(
            Completion.of(
                parsed,
                provider=self.provider,
                model=self.model,
                api=self.api,
                cost=build_cost(self._catalog, self.provider, self.model, parsed),
            )
        )

    def _body_for(self, item: Mapping[str, Any]) -> dict[str, Any]:
        """One item as the request body it would have been sent as on its own.

        Built by the PROVIDER's adapter. Building every provider's items with
        one adapter is not a near miss -- Google rejects the batch outright
        (`Unknown name "messages"`), and OpenAI accepts the job and then fails
        every item inside it, which costs the whole wait before it says so.
        """
        messages = item.get("messages")
        if messages is None:
            prompt = item.get("prompt")
            if prompt is None:
                raise ValueError(
                    "AutoBatcher.add: an item needs a 'prompt' or 'messages'; "
                    f"got keys {sorted(item)}"
                )
            messages = [{"role": "user", "content": str(prompt)}]

        request: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            **{k: v for k, v in self._options.items() if v is not None},
        }
        for key, value in (
            ("maxTokens", item.get("max_tokens", self._max_tokens)),
            ("system", item.get("system", self._system)),
            ("temperature", item.get("temperature")),
        ):
            if value is not None:
                request[key] = value

        built = self._completion.build_request(request)
        body: dict[str, Any] = dict(built.body or {})
        return body

    def _resolve_fetch(self, engine: Any, transport: Any) -> Any:
        """Who sends. An engine keeps its queue; a transport does the sending."""
        if engine is not None and transport is not None:
            pair: tuple[Any, Any] = engine.fetches_for(transport)
            return pair[0]
        if engine is not None:
            fetch: Any = engine.fetch
            return fetch
        from ..transport import as_fetch, http_transport

        return as_fetch(transport or http_transport())

    def __repr__(self) -> str:
        return (
            f"<AutoBatcher {self.provider}/{self.model} pending={self.pending} "
            f"batch={self.batch_id}>"
        )


__all__ = ["ADAPTERS", "AutoBatcher", "BatchTicket"]
