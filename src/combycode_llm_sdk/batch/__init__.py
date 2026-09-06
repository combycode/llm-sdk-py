"""Provider batch APIs: half the price, in exchange for waiting.

`AutoBatcher` collects items from callers that know nothing about each other and
submits them as one job, then settles each caller's ticket BY ID -- because
providers reorder results, and pairing by position hands each caller the other's
answer with both looking correct.

All four batch APIs have an adapter, and no two of them agree about anything:
Anthropic inlines every request, OpenAI uploads a JSONL file and then creates a
batch over it, xAI creates the batch and then adds requests to it, and Google
returns the answers inline on the job resource. OpenRouter is refused by name --
it fronts other providers' models but hosts no batch API of its own.

Transposed from `unified-library-ts/src/plugins/batch/` and
`llm/providers/*/batch.ts`.
"""

from __future__ import annotations

from .adapters import (
    BATCH_ADAPTERS,
    AnthropicBatchAdapter,
    GoogleBatchAdapter,
    OpenAIBatchAdapter,
    XaiBatchAdapter,
)
from .batcher import ADAPTERS, AutoBatcher, BatchTicket
from .types import (
    CANCELLED,
    COMPLETED,
    EXPIRED,
    FAILED,
    PENDING,
    PROCESSING,
    TERMINAL,
    BatchProviderAdapter,
    BatchRequest,
    BatchResult,
    BatchStatus,
    BatchStrategy,
    PendingBatchJob,
)

__all__ = [
    "ADAPTERS",
    "BATCH_ADAPTERS",
    "CANCELLED",
    "COMPLETED",
    "EXPIRED",
    "FAILED",
    "PENDING",
    "PROCESSING",
    "TERMINAL",
    "AnthropicBatchAdapter",
    "AutoBatcher",
    "BatchProviderAdapter",
    "BatchRequest",
    "BatchResult",
    "BatchStatus",
    "BatchStrategy",
    "BatchTicket",
    "GoogleBatchAdapter",
    "OpenAIBatchAdapter",
    "PendingBatchJob",
    "XaiBatchAdapter",
]
