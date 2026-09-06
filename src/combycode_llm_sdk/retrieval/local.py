"""Retrieval that runs here: chunk, embed, store, search.

The only backend where the index is ours. Documents are chunked, each chunk
embedded through the same `embed()` path as everything else, and the vectors go
into a `VectorStore` -- in memory by default, or whatever the caller brings.

`as_tool()` returns a REAL tool the agent loop executes client-side, which is
what makes local retrieval work on Anthropic and every other provider that has
no hosted vector store. The three hosted backends can only hand the provider a
tool spec and let it search server-side.

Transposed from `unified-library-ts/src/plugins/retrieval/local.ts`.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from .chunker import DEFAULT_CHUNK_MAX_TOKENS, DEFAULT_CHUNK_OVERLAP_TOKENS, chunk_text
from .types import (
    CorpusRef,
    CreateCorpusOptions,
    DocumentRef,
    DocumentSource,
    IndexCounts,
    IndexStatus,
    RetrievalCapabilities,
    RetrievalHit,
)
from .vector_store import InMemoryVectorStore, VectorEntry, VectorStore

#: How many passages a search returns when nobody says. Five is enough to
#: answer from and few enough to fit in a prompt beside the question.
DEFAULT_SEARCH_TOP_K = 5

#: Everything, by default. A threshold that silently drops the only relevant
#: passage is worse than a low-scoring hit the model can ignore.
DEFAULT_MIN_SCORE = 0.0

#: The same name OpenAI's hosted tool uses, deliberately: the model has seen it
#: in training, and the semantics are the same even though this one runs here.
LOCAL_TOOL_NAME = "file_search"
LOCAL_TOOL_DESCRIPTION = (
    "Search the local knowledge base for passages relevant to the query."
)


def format_hits(hits: Sequence[RetrievalHit]) -> str:
    """Hits as the text a model reads.

    Numbered and cited, because the model is being asked to answer FROM these
    and a reader of its answer needs to be able to check it.
    """
    if not hits:
        return "No relevant passages found."
    lines = []
    for index, hit in enumerate(hits, start=1):
        citation = f" [{hit.citation}]" if hit.citation else ""
        lines.append(f"[{index}]{citation} (score {hit.score:.3f})\n{hit.text}")
    return "\n\n".join(lines)


class LocalRetrievalBackend:
    """Chunk, embed and search, all in this process."""

    capabilities = RetrievalCapabilities(
        user_chunking=True,
        search_modes=("cosine",),
        expiration=False,
        direct_search=True,
        id_field="id",
        citation_format="label:offset",
    )

    def __init__(
        self,
        *,
        embed_adapter: Any,
        fetch: Any,
        embedding_model: str,
        vector_store: VectorStore | None = None,
        max_tokens: int | None = None,
        overlap_tokens: int | None = None,
    ) -> None:
        self._embed = embed_adapter
        self._fetch = fetch
        self._model = embedding_model
        self._store: VectorStore = vector_store or InMemoryVectorStore()
        self._max_tokens = max_tokens or DEFAULT_CHUNK_MAX_TOKENS
        self._overlap_tokens = (
            overlap_tokens if overlap_tokens is not None else DEFAULT_CHUNK_OVERLAP_TOKENS
        )
        self._corpora: dict[str, CorpusRef] = {}

    # -- lifecycle -----------------------------------------------------------

    def create_corpus(self, options: CreateCorpusOptions) -> CorpusRef:
        ref = CorpusRef(id=str(uuid.uuid4()), name=options.name, backend="local")
        self._corpora[ref.id] = ref
        return ref

    def add_document(
        self,
        corpus: CorpusRef,
        source: DocumentSource,
        metadata: Mapping[str, Any] | None = None,
    ) -> DocumentRef:
        doc_id = str(uuid.uuid4())
        merged = {**dict(source.metadata or {}), **dict(metadata or {})}
        chunks = chunk_text(source.text, self._max_tokens, self._overlap_tokens)

        # One embed call for the whole document: the batch shapes exist for
        # exactly this, and a call per chunk would be N round trips.
        result = self._embed.embed(self._model, [c.text for c in chunks], self._fetch)
        vectors = list(result.embeddings)

        for index, chunk in enumerate(chunks):
            self._store.upsert(
                VectorEntry(
                    id=f"{doc_id}:{index}",
                    doc_id=doc_id,
                    corpus_id=corpus.id,
                    vector=vectors[index] if index < len(vectors) else (),
                    text=chunk.text,
                    metadata=merged,
                    # Where in the document, not just which one: a citation
                    # that names only the file cannot be checked.
                    citation=f"{source.label}:{chunk.offset}" if source.label else None,
                )
            )

        return DocumentRef(id=doc_id, corpus_id=corpus.id, source=source)

    def index_status(self, corpus: CorpusRef) -> IndexStatus:
        # Nothing is asynchronous here: `add_document` returned once the
        # vectors were stored, so the index is ready by construction.
        count = self._store.count(corpus.id)
        return IndexStatus(
            state="ready", counts=IndexCounts(total=count, indexed=count, failed=0)
        )

    def wait_indexed(self, corpus: CorpusRef, timeout: float = 0.0) -> IndexStatus:
        """Already indexed. Present so the four backends share one lifecycle."""
        return self.index_status(corpus)

    def remove_document(self, corpus: CorpusRef, doc_id: str) -> None:
        self._store.remove_by_doc_id(corpus.id, doc_id)

    def delete_corpus(self, corpus: CorpusRef) -> None:
        self._store.remove_by_corpus_id(corpus.id)
        self._corpora.pop(corpus.id, None)

    def list_corpora(self) -> list[CorpusRef]:
        return list(self._corpora.values())

    # -- searching -----------------------------------------------------------

    def search(
        self,
        corpora: Sequence[CorpusRef],
        query: str,
        max_results: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalHit]:
        top_k = max_results or DEFAULT_SEARCH_TOP_K
        floor = DEFAULT_MIN_SCORE if min_score is None else min_score

        result = self._embed.embed(self._model, [query], self._fetch)
        vectors = list(result.embeddings)
        vector = vectors[0] if vectors else ()

        hits: list[RetrievalHit] = []
        for corpus in corpora:
            for entry, score in self._store.query(corpus.id, vector, top_k):
                if score >= floor:
                    hits.append(
                        RetrievalHit(
                            text=entry.text,
                            score=score,
                            doc_id=entry.doc_id,
                            metadata=entry.metadata,
                            citation=entry.citation,
                        )
                    )
        # Re-ranked ACROSS corpora before truncating: taking the top five from
        # each and concatenating would return five weak hits from one corpus
        # ahead of a strong one from another.
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def as_tool(self, corpora: Sequence[CorpusRef], max_results: int | None = None) -> Any:
        """A real tool the loop runs, so this works on every provider."""
        from ..helpers.tool import Tool

        top_k = max_results or DEFAULT_SEARCH_TOP_K
        held = list(corpora)
        backend = self

        def file_search(query: str) -> str:
            """Search the local knowledge base for relevant passages."""
            return format_hits(backend.search(held, query, max_results=top_k))

        file_search.__name__ = LOCAL_TOOL_NAME
        file_search.__doc__ = LOCAL_TOOL_DESCRIPTION
        return Tool(
            file_search,
            {
                "name": LOCAL_TOOL_NAME,
                "description": LOCAL_TOOL_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query text."}
                    },
                    "required": ["query"],
                },
            },
        )

    def __repr__(self) -> str:
        return f"<LocalRetrievalBackend {len(self._corpora)} corpus(es)>"


__all__ = [
    "DEFAULT_MIN_SCORE",
    "DEFAULT_SEARCH_TOP_K",
    "LOCAL_TOOL_DESCRIPTION",
    "LOCAL_TOOL_NAME",
    "LocalRetrievalBackend",
    "format_hits",
]
