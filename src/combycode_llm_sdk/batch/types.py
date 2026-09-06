"""What a batch is made of, and when one is worth making.

The type that carries the weight here is `BatchResult.custom_id`. A provider is
free to return results in any order, and Anthropic measurably does return a
two-item batch back to front -- so a batcher that paired results with callers by
POSITION would hand each caller the other's answer, with both answers looking
entirely correct. Every result carries the id its request went out with, and
that id is the only thing allowed to settle a caller's ticket.

Transposed from `unified-library-ts/src/plugins/batch/types.ts`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

#: Batch status, normalised across providers.
PENDING = "pending"
PROCESSING = "processing"
COMPLETED = "completed"
FAILED = "failed"
EXPIRED = "expired"
CANCELLED = "cancelled"

#: A batch is finished when it reaches one of these, however it got there.
TERMINAL = frozenset({COMPLETED, FAILED, EXPIRED, CANCELLED})


@dataclass(frozen=True)
class BatchRequest:
    """One caller's request, tagged so its answer can find its way home."""

    custom_id: str
    body: Mapping[str, Any]


@dataclass(frozen=True)
class BatchResult:
    """One answer, tagged with the id of the request it belongs to."""

    custom_id: str
    success: bool
    response: Mapping[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class BatchStatus:
    """How far along a submitted job is."""

    id: str
    status: str
    total: int = 0
    completed: int = 0
    failed: int = 0
    pending: int = 0

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL


@dataclass(frozen=True)
class BatchStrategy:
    """When to submit, and how eagerly to look for the answer.

    The window is the whole trade: batch pricing is roughly half, and the cost
    is latency. A caller who cannot wait should not be batching at all, which is
    what `min_batch_size` expresses -- one item is not a batch, it is a
    `complete()` with extra steps.
    """

    #: How long to collect before submitting.
    window_seconds: float = 10.0
    #: Fewer than this and the window has not earned its wait.
    min_batch_size: int = 2
    #: More than this and the batch is submitted immediately.
    max_batch_size: int = 1000
    #: How long to wait before the first status check.
    first_poll_seconds: float = 5.0
    #: And between the ones after it.
    poll_seconds: float = 15.0

    def should_submit(self, size: int, waited: float) -> bool:
        """Whether a collection of this size, held this long, should go now."""
        # Not redundant with `min_batch_size`: a caller who sets that to 0 has
        # said "submit whatever you have", and an EMPTY batch is not that.
        if size <= 0:
            return False
        if size >= self.max_batch_size:
            return True
        return waited >= self.window_seconds and size >= self.min_batch_size

    def first_poll_after(self, size: int) -> float:
        """When to look first. A bigger batch takes longer, so look later."""
        return self.first_poll_seconds + min(size, 100) * 0.05


class BatchProviderAdapter(Protocol):
    """What a provider must do to be batchable here."""

    @property
    def name(self) -> str: ...

    def submit(self, requests: list[BatchRequest], fetch: Any) -> str: ...

    def get_status(self, batch_id: str, fetch: Any) -> BatchStatus: ...

    def get_results(self, batch_id: str, fetch: Any) -> list[BatchResult]: ...

    def cancel(self, batch_id: str, fetch: Any) -> None: ...


@dataclass
class PendingBatchJob:
    """A submitted job, as the store holds it between restarts."""

    batch_id: str
    provider: str
    created_at: float
    custom_ids: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        return {
            "batchId": self.batch_id,
            "provider": self.provider,
            "createdAt": self.created_at,
            "customIds": list(self.custom_ids),
        }

    @staticmethod
    def of(row: Mapping[str, Any]) -> PendingBatchJob:
        return PendingBatchJob(
            batch_id=str(row.get("batchId") or ""),
            provider=str(row.get("provider") or ""),
            created_at=float(row.get("createdAt") or 0.0),
            custom_ids=[str(c) for c in (row.get("customIds") or [])],
        )


__all__ = [
    "CANCELLED",
    "COMPLETED",
    "EXPIRED",
    "FAILED",
    "PENDING",
    "PROCESSING",
    "TERMINAL",
    "BatchProviderAdapter",
    "BatchRequest",
    "BatchResult",
    "BatchStatus",
    "BatchStrategy",
    "PendingBatchJob",
]
