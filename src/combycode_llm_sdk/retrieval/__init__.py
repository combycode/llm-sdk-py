"""Retrieval (RAG): corpora, indexing, search.

`Corpus` is the entry point -- one handle over four backends. `local` indexes
here and works on every provider; the three hosted ones keep the index at the
provider and hand back a tool spec it runs itself.

Transposed from `unified-library-ts/src/plugins/retrieval/`.
"""

from __future__ import annotations

from .chunker import (
    DEFAULT_CHUNK_MAX_TOKENS,
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    TextChunk,
    chunk_text,
)
from .corpus import HOSTED_BACKENDS, Corpus
from .hosted import (
    HostedGoogleRetrievalBackend,
    HostedOpenAIRetrievalBackend,
    HostedXaiRetrievalBackend,
    normalise_state,
)
from .local import LocalRetrievalBackend, format_hits
from .types import (
    ChunkingOptions,
    CorpusRef,
    CreateCorpusOptions,
    DocumentRef,
    DocumentSource,
    IndexCounts,
    IndexState,
    IndexStatus,
    RetrievalBackend,
    RetrievalCapabilities,
    RetrievalHit,
)
from .vector_store import InMemoryVectorStore, VectorEntry, VectorStore, cosine_similarity

__all__ = [
    "DEFAULT_CHUNK_MAX_TOKENS",
    "DEFAULT_CHUNK_OVERLAP_TOKENS",
    "HOSTED_BACKENDS",
    "ChunkingOptions",
    "Corpus",
    "CorpusRef",
    "CreateCorpusOptions",
    "DocumentRef",
    "DocumentSource",
    "HostedGoogleRetrievalBackend",
    "HostedOpenAIRetrievalBackend",
    "HostedXaiRetrievalBackend",
    "InMemoryVectorStore",
    "IndexCounts",
    "IndexState",
    "IndexStatus",
    "LocalRetrievalBackend",
    "RetrievalBackend",
    "RetrievalCapabilities",
    "RetrievalHit",
    "TextChunk",
    "VectorEntry",
    "VectorStore",
    "chunk_text",
    "cosine_similarity",
    "format_hits",
    "normalise_state",
]
