"""Many prompts at the provider's batch price, in one call.

    results = batch(model=..., api_key=..., requests=[{"prompt": "..."}, ...])

Two shapes, because batch jobs take minutes to hours and a caller cannot always
block for one:

- `batch()` submits and waits, and hands back the answers in the order the
  requests were given.
- `submit_batch()` returns a handle immediately. Poll it, persist its id, come
  back tomorrow.

`AutoBatcher` is the third shape and a different question: it exists for callers
who arrive ONE AT A TIME and do not know about each other. Here the caller
already has the whole list.

Transposed from `unified-library-ts/src/helpers/batch.ts`.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..batch.batcher import AutoBatcher, BatchTicket
from ..batch.types import TERMINAL, BatchStatus, BatchStrategy
from ..network.errors import LLMError
from ..results import Completion

#: How often a wait looks, and how long it waits, when the caller says nothing.
DEFAULT_POLL_SECONDS = 15.0
DEFAULT_TIMEOUT_SECONDS = 3600.0


@dataclass(frozen=True)
class BatchItemResult:
    """One request's answer, tagged with the id it went out under."""

    custom_id: str
    success: bool
    #: The reply text. `""` when the item failed -- never None, so a caller
    #: reading `.text` on a failed row gets an empty string rather than a
    #: `TypeError` several lines later.
    text: str
    completion: Completion | None = None
    error: str | None = None


class BatchJob:
    """A submitted job, and the two things you can do with one: look, or wait."""

    def __init__(self, batcher: AutoBatcher, order: Sequence[BatchTicket]) -> None:
        self._batcher = batcher
        self._order = list(order)

    @property
    def id(self) -> str:
        batch_id = self._batcher.batch_id
        if batch_id is None:
            raise LLMError("this job was never submitted")
        return batch_id

    @property
    def provider(self) -> str:
        return self._batcher.provider

    @property
    def model(self) -> str:
        return self._batcher.model

    def status(self) -> BatchStatus:
        """Where the provider says the job has got to."""
        return self._batcher.status()

    def results(self) -> list[BatchItemResult]:
        """The answers. Refuses while the job is still running.

        Not an empty list: "not finished" and "finished with nothing" are
        different answers, and only one of them means the work is done.
        """
        state = self.status()
        if state.status not in TERMINAL:
            raise LLMError(
                f"batch {self.id} is not finished (status: {state.status}). "
                "Call wait(), or poll status() yourself."
            )
        self._batcher.collect()
        return self._collected()

    def wait(
        self,
        *,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        on_progress: Any = None,
    ) -> list[BatchItemResult]:
        """Poll until the job ends, then hand back every answer."""
        deadline = time.monotonic() + timeout_seconds
        while True:
            state = self.status()
            if on_progress is not None:
                on_progress(state)
            if state.status in TERMINAL:
                self._batcher.collect()
                return self._collected()
            if time.monotonic() >= deadline:
                raise LLMError(
                    f"batch {self.id} did not finish within {timeout_seconds:g}s "
                    f"(last status: {state.status})"
                )
            if poll_seconds > 0:
                time.sleep(poll_seconds)

    def cancel(self) -> None:
        self._batcher.cancel()

    def _collected(self) -> list[BatchItemResult]:
        """Every ticket, in the order the caller supplied them.

        The ORDER is the caller's, and the correlation is by id -- those are two
        different things and both matter. A provider returns results in its own
        order (Anthropic reverses a two-item batch), so the ids settle the
        tickets and this list then re-imposes the order the caller asked in.
        """
        out: list[BatchItemResult] = []
        for ticket in self._order:
            if ticket.failed:
                out.append(
                    BatchItemResult(
                        custom_id=ticket.custom_id,
                        success=False,
                        text="",
                        error=_reason(ticket),
                    )
                )
            elif ticket.settled:
                completion = ticket.result()
                out.append(
                    BatchItemResult(
                        custom_id=ticket.custom_id,
                        success=True,
                        text=completion.text,
                        completion=completion,
                    )
                )
            else:
                # The job ended and this item is not in the results. Reported as
                # a failure rather than dropped: a caller comparing lengths must
                # not silently receive a shorter list than they asked for.
                out.append(
                    BatchItemResult(
                        custom_id=ticket.custom_id,
                        success=False,
                        text="",
                        error="the provider returned no result for this item",
                    )
                )
        return out

    def __repr__(self) -> str:
        return f"<BatchJob {self._batcher.batch_id} {len(self._order)} item(s)>"


def _reason(ticket: BatchTicket) -> str:
    try:
        ticket.result()
    except LLMError as exc:
        return str(exc)
    return "unknown"


def submit_batch(
    *,
    model: str,
    requests: Sequence[Mapping[str, Any]],
    api_key: str | None = None,
    engine: Any = None,
    catalog: Any = None,
    transport: Any = None,
    base_url: str | None = None,
    max_tokens: int | None = None,
    system: str | None = None,
    **options: Any,
) -> BatchJob:
    """Submit a batch and return its handle immediately.

    Each request is `{"prompt": ...}` or `{"messages": [...]}`, optionally with
    its own `custom_id`, `max_tokens`, `system` or `temperature`. An item with
    no id is given `req-<index>`, so a caller who supplies none still gets ids
    that mean something when they appear in a provider's console.
    """
    if not requests:
        raise ValueError("submit_batch: `requests` is empty; there is nothing to submit")

    batcher = AutoBatcher(
        model=model,
        api_key=api_key,
        engine=engine,
        catalog=catalog,
        transport=transport,
        base_url=base_url,
        max_tokens=max_tokens,
        system=system,
        # The window is the wrong mechanism here: the caller handed over the
        # whole list at once, so there is nothing to wait for.
        strategy=BatchStrategy(window_seconds=0.0, min_batch_size=1),
        **options,
    )
    order = [
        batcher.add(request, custom_id=str(request.get("custom_id") or f"req-{index}"))
        for index, request in enumerate(requests)
    ]
    batcher.submit()
    return BatchJob(batcher, order)


def batch(
    *,
    model: str,
    requests: Sequence[Mapping[str, Any]],
    api_key: str | None = None,
    engine: Any = None,
    catalog: Any = None,
    transport: Any = None,
    base_url: str | None = None,
    max_tokens: int | None = None,
    system: str | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    on_progress: Any = None,
    **options: Any,
) -> list[BatchItemResult]:
    """Submit a batch, wait for it, and return every answer in request order."""
    job = submit_batch(
        model=model,
        requests=requests,
        api_key=api_key,
        engine=engine,
        catalog=catalog,
        transport=transport,
        base_url=base_url,
        max_tokens=max_tokens,
        system=system,
        **options,
    )
    return job.wait(
        poll_seconds=poll_seconds, timeout_seconds=timeout_seconds, on_progress=on_progress
    )


__all__ = [
    "DEFAULT_POLL_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "BatchItemResult",
    "BatchJob",
    "batch",
    "submit_batch",
]
