"""Where the vectors live, when they live here.

The interface is deliberately four methods. Anything larger would be a database
API, and the point is that a caller can put pgvector, Qdrant or Weaviate behind
it without this library knowing. `InMemoryVectorStore` is the bundled default:
cosine similarity, in-process, no dependency.

Transposed from `unified-library-ts/src/plugins/retrieval/vector-store.ts`.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

#: Guards the cosine denominator. A zero vector -- an empty chunk that embedded
#: to nothing -- would otherwise divide by zero rather than simply not match.
COSINE_EPSILON = 1e-10


@dataclass
class VectorEntry:
    """One embedded chunk."""

    id: str
    doc_id: str
    corpus_id: str
    vector: Sequence[float]
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    citation: str | None = None


class VectorStore(Protocol):
    """Pluggable storage. Four methods, so a real database can satisfy it."""

    def upsert(self, entry: VectorEntry) -> None: ...

    def remove_by_doc_id(self, corpus_id: str, doc_id: str) -> None: ...

    def remove_by_corpus_id(self, corpus_id: str) -> None: ...

    def query(
        self, corpus_id: str, vector: Sequence[float], top_k: int
    ) -> list[tuple[VectorEntry, float]]: ...

    def count(self, corpus_id: str | None = None) -> int: ...


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Similarity in [-1, 1], or 0.0 when either side has no magnitude.

    Length mismatch returns 0.0 rather than raising: it means the corpus holds
    vectors from two different embedding models, which is a bad state but not
    one worth crashing a search over -- the mismatched entries simply never win.
    """
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    denominator = math.sqrt(norm_a) * math.sqrt(norm_b)
    return dot / denominator if denominator > COSINE_EPSILON else 0.0


class InMemoryVectorStore:
    """Vectors in this process, and gone when it ends."""

    def __init__(self) -> None:
        self._entries: dict[str, VectorEntry] = {}
        self._lock = threading.Lock()

    def upsert(self, entry: VectorEntry) -> None:
        with self._lock:
            self._entries[entry.id] = entry

    def remove_by_doc_id(self, corpus_id: str, doc_id: str) -> None:
        with self._lock:
            for key in [
                k
                for k, e in self._entries.items()
                if e.corpus_id == corpus_id and e.doc_id == doc_id
            ]:
                del self._entries[key]

    def remove_by_corpus_id(self, corpus_id: str) -> None:
        with self._lock:
            for key in [k for k, e in self._entries.items() if e.corpus_id == corpus_id]:
                del self._entries[key]

    def query(
        self, corpus_id: str, vector: Sequence[float], top_k: int
    ) -> list[tuple[VectorEntry, float]]:
        """The `top_k` nearest, best first."""
        with self._lock:
            candidates = [e for e in self._entries.values() if e.corpus_id == corpus_id]
        scored = [(entry, cosine_similarity(vector, entry.vector)) for entry in candidates]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[: max(0, top_k)]

    def count(self, corpus_id: str | None = None) -> int:
        with self._lock:
            if corpus_id is None:
                return len(self._entries)
            return sum(1 for e in self._entries.values() if e.corpus_id == corpus_id)

    def __repr__(self) -> str:
        return f"<InMemoryVectorStore {len(self._entries)} chunk(s)>"


__all__ = [
    "COSINE_EPSILON",
    "InMemoryVectorStore",
    "VectorEntry",
    "VectorStore",
    "cosine_similarity",
]
