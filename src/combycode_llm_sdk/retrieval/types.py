"""Retrieval (RAG): the shape every backend shares.

One lifecycle, four backends: create a corpus, add documents, wait for
indexing, then search or hand the model a tool. Local runs entirely here with a
vector store in memory; the three hosted ones keep the index at the provider.

Two design rules the interface exists to enforce:

**Divergence lives in capabilities, not in the contract.** Backends differ in
real ways -- xAI can be searched directly, OpenAI's vector store cannot -- and
those differences are DECLARED so a caller can degrade gracefully instead of
discovering them from a runtime error.

**Anti-lock-in.** A `CorpusRef` carries the original source alongside the
opaque provider id, and a `DocumentRef` keeps the text it was built from. The
documents are the rebuildable source of truth; nothing here depends on ever
exporting a hosted store's embeddings, because no provider lets you.

Transposed from `unified-library-ts/src/plugins/retrieval/types.ts`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

IndexState = Literal["pending", "indexing", "ready", "error"]

CitationFormat = Literal["label:offset", "file_id", "gemini", "collections-uri", "none"]


@dataclass(frozen=True)
class DocumentSource:
    """What the caller supplied. Everything else is derived from it."""

    text: str
    #: A filename or URL. Kept for citation, and for naming the upload.
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CorpusRef:
    """A corpus somewhere, plus enough to rebuild it if that somewhere goes away."""

    id: str
    name: str
    backend: str
    #: Provider-specific fields that are not part of the contract.
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentRef:
    """One document in a corpus, and the source it came from."""

    id: str
    corpus_id: str
    source: DocumentSource
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IndexCounts:
    total: int = 0
    indexed: int = 0
    failed: int = 0


@dataclass(frozen=True)
class IndexStatus:
    """Indexing state, normalised across four providers that all name it differently."""

    state: IndexState
    #: Absent where a backend does not report file-level counts. None rather
    #: than zeros: "not reported" and "none indexed" are different answers.
    counts: IndexCounts | None = None


@dataclass(frozen=True)
class RetrievalHit:
    """One matched chunk."""

    text: str
    score: float
    doc_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    #: Human-readable, in whatever form the backend can produce -- see
    #: `RetrievalCapabilities.citation_format` for which.
    citation: str | None = None


@dataclass(frozen=True)
class RetrievalCapabilities:
    """What a backend can actually do, declared rather than discovered.

    A caller reads this to degrade gracefully. Without it the only way to learn
    that OpenAI's vector store cannot be searched directly is to try.
    """

    #: Whether the caller controls chunking, or the provider does it invisibly.
    user_chunking: bool = False
    search_modes: Sequence[str] = ()
    expiration: bool = False
    #: Whether `search()` works at all. False on backends that only ever hand
    #: the model a tool and let the provider do the retrieval server-side.
    direct_search: bool = False
    id_field: str = "id"
    citation_format: CitationFormat = "none"


@dataclass(frozen=True)
class ChunkingOptions:
    max_tokens: int | None = None
    overlap_tokens: int | None = None


@dataclass(frozen=True)
class CreateCorpusOptions:
    name: str
    chunking: ChunkingOptions | None = None
    #: How long a hosted corpus lives untouched. Hosted stores bill for
    #: storage, so a corpus nobody deletes is a standing charge.
    expires_after_days: int | None = None
    embedding_model: str | None = None


class RetrievalBackend(Protocol):
    """The five stages, the same shape on every backend."""

    capabilities: RetrievalCapabilities

    def create_corpus(self, options: CreateCorpusOptions) -> CorpusRef: ...

    def add_document(
        self,
        corpus: CorpusRef,
        source: DocumentSource,
        metadata: Mapping[str, Any] | None = None,
    ) -> DocumentRef: ...

    def index_status(self, corpus: CorpusRef) -> IndexStatus: ...

    def remove_document(self, corpus: CorpusRef, doc_id: str) -> None: ...

    def delete_corpus(self, corpus: CorpusRef) -> None: ...

    def list_corpora(self) -> Sequence[CorpusRef]: ...

    def search(
        self,
        corpora: Sequence[CorpusRef],
        query: str,
        max_results: int | None = None,
        min_score: float | None = None,
    ) -> Sequence[RetrievalHit]: ...

    def as_tool(self, corpora: Sequence[CorpusRef], **options: Any) -> Any: ...


__all__ = [
    "ChunkingOptions",
    "CitationFormat",
    "CorpusRef",
    "CreateCorpusOptions",
    "DocumentRef",
    "DocumentSource",
    "IndexCounts",
    "IndexState",
    "IndexStatus",
    "RetrievalBackend",
    "RetrievalCapabilities",
    "RetrievalHit",
]
