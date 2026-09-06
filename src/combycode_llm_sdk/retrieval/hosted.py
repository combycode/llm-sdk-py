"""Retrieval where the provider keeps the index.

Three backends, one shape. Each creates a store, uploads documents, reports
indexing progress, and hands back a TOOL SPEC rather than a tool -- the search
happens inside the provider's own call, so there is nothing for us to execute.
That is the line between these and `LocalRetrievalBackend`, and it is why
`capabilities.direct_search` exists: OpenAI and Google cannot be searched from
here at all, and saying so up front is better than a 404.

The differences that survive into the code are real ones:

- **OpenAI** uploads a file, then attaches it to the vector store: two calls,
  because a file exists independently of the stores that reference it.
- **Google** uploads and imports, and its indexing is a long-running operation
  that has to be polled.
- **xAI** is the only one that can be searched directly, and the only one that
  needs a SECOND key -- collection management is a different credential from
  inference, which is a real operational trap.

Transposed from `unified-library-ts/src/plugins/retrieval/hosted-*.ts`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

from ..llm.wire_multipart import MultipartFile, encode_multipart, to_form_data
from ..llm.wire_transforms import make_registry
from ..util.hash import fnv1a32_hex
from ..wire.interpreter import build_from_spec
from ..wire.retrieval_specs import retrieval_spec
from .types import (
    CorpusRef,
    CreateCorpusOptions,
    DocumentRef,
    DocumentSource,
    IndexCounts,
    IndexState,
    IndexStatus,
    RetrievalCapabilities,
    RetrievalHit,
)

#: The tool type OpenAI and xAI both use for a hosted store.
FILE_SEARCH_TOOL_TYPE = "file_search"

#: How long to wait for a hosted index, and how often to ask. Indexing a small
#: document is seconds; the ceiling is there so a stuck corpus fails rather
#: than hanging a caller forever.
DEFAULT_INDEX_TIMEOUT = 120.0
DEFAULT_INDEX_POLL = 2.0

#: What each provider's own status words mean in ours.
_STATES: Mapping[str, IndexState] = {
    "completed": "ready",
    "active": "ready",
    "ready": "ready",
    "succeeded": "ready",
    "in_progress": "indexing",
    "processing": "indexing",
    "pending": "pending",
    "queued": "pending",
    "failed": "error",
    "error": "error",
    "cancelled": "error",
    "expired": "error",
}


def normalise_state(raw: Any) -> IndexState:
    """A provider's status word as ours.

    An unrecognised word reads as `indexing`, not `ready`: telling a caller the
    corpus is searchable when it may not be produces an empty answer that looks
    like a missing document.
    """
    return _STATES.get(str(raw or "").lower(), "indexing")


def document_file(source: DocumentSource) -> MultipartFile:
    """The document as an upload part.

    The fallback name is a CONTENT HASH, not a random id: a retried upload has
    to arrive under the same name or it becomes a second document, and a
    request nobody can reproduce cannot be asserted in a test either.
    """
    return MultipartFile(
        filename=source.label or f"doc-{fnv1a32_hex(source.text)}.txt",
        data=source.text.encode("utf-8"),
        mime_type="text/plain",
    )


def _body(response: Any) -> Mapping[str, Any]:
    body = response.get("body") if isinstance(response, Mapping) else None
    if isinstance(body, (str, bytes)):
        try:
            body = json.loads(body)
        except ValueError:
            return {}
    return body if isinstance(body, Mapping) else {}


def _status(response: Any) -> int:
    value = response.get("status") if isinstance(response, Mapping) else None
    return value if isinstance(value, int) else 0


def _rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [v for v in value if isinstance(v, Mapping)]


class _HostedBackend:
    """The parts all three hosted backends share."""

    provider = ""
    backend_name = ""
    base_url = ""
    capabilities = RetrievalCapabilities()

    def __init__(self, *, api_key: str, fetch: Any, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._fetch = fetch
        self._base_url = base_url or self.base_url
        self._registry = make_registry({})

    def _request(
        self, spec_id: str, payload: Mapping[str, Any], file: MultipartFile | None = None
    ) -> dict[str, Any]:
        """One request from its spec, plus the routing fields the wire never sees."""
        built = build_from_spec(
            retrieval_spec(spec_id),
            dict(payload),
            self._registry,
            self.provider,
            None,
            self._config(),
        )
        request: dict[str, Any] = {
            "url": built.url,
            "method": built.method or "POST",
            "headers": dict(built.headers or {}),
            "provider": self.provider,
            # Named, not derived: these endpoints are model-agnostic and a
            # queue called `openai/` would be nobody's.
            "model": "retrieval",
            "responseType": "json",
        }
        multipart = getattr(built, "multipart", None)
        if multipart and file is not None:
            encoded, content_type = encode_multipart(to_form_data(multipart, file))
            request["body"] = encoded
            request["rawBody"] = True
            request["headers"]["content-type"] = content_type
        elif not getattr(built, "no_body", False):
            request["body"] = built.body
        return request

    def _config(self) -> dict[str, Any]:
        """What the specs may read. A backend with more planes overrides this."""
        return {"baseURL": self._base_url, "apiKey": self._api_key}

    def _send(self, spec_id: str, payload: Mapping[str, Any], what: str,
              file: MultipartFile | None = None) -> Mapping[str, Any]:
        response = self._fetch(self._request(spec_id, payload, file))
        if _status(response) >= 400:
            raise RuntimeError(
                f"{self.backend_name}: {what} failed ({_status(response)}): {_body(response)}"
            )
        return _body(response)

    # -- lifecycle -----------------------------------------------------------

    def wait_indexed(
        self,
        corpus: CorpusRef,
        timeout: float = DEFAULT_INDEX_TIMEOUT,
        poll: float = DEFAULT_INDEX_POLL,
    ) -> IndexStatus:
        """Block until the provider says the corpus is searchable.

        Searching an unindexed corpus returns nothing, which is
        indistinguishable from a document that was never added -- so this waits
        rather than letting the caller discover it as an empty answer.
        """
        deadline = time.monotonic() + timeout
        status = self.index_status(corpus)
        while status.state in ("pending", "indexing") and time.monotonic() < deadline:
            time.sleep(min(poll, max(0.0, deadline - time.monotonic())))
            status = self.index_status(corpus)
        if status.state in ("pending", "indexing"):
            raise TimeoutError(
                f"{self.backend_name}: corpus {corpus.id} was still {status.state} "
                f"after {timeout:g}s."
            )
        return status

    def index_status(self, corpus: CorpusRef) -> IndexStatus:
        raise NotImplementedError

    def search(
        self,
        corpora: Sequence[CorpusRef],
        query: str,
        max_results: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalHit]:
        raise RuntimeError(
            f"{self.backend_name}: direct search is not supported -- the index lives at "
            "the provider and is only reachable through a completion. Use as_tool() and "
            "pass the result to complete(tools=[...])."
        )


class HostedOpenAIRetrievalBackend(_HostedBackend):
    """OpenAI vector stores: upload a file, then attach it."""

    provider = "openai"
    backend_name = "hostedOpenAI"
    base_url = "https://api.openai.com"
    capabilities = RetrievalCapabilities(
        user_chunking=True,
        search_modes=("hybrid",),
        expiration=True,
        direct_search=False,
        id_field="id",
        citation_format="file_id",
    )

    def create_corpus(self, options: CreateCorpusOptions) -> CorpusRef:
        data = self._send(
            "openai/retrieval.createCorpus",
            {"name": options.name, "expiresAfterDays": options.expires_after_days},
            "createCorpus",
        )
        return CorpusRef(
            id=str(data.get("id") or ""),
            name=str(data.get("name") or options.name),
            backend=self.backend_name,
        )

    def add_document(
        self,
        corpus: CorpusRef,
        source: DocumentSource,
        metadata: Mapping[str, Any] | None = None,
    ) -> DocumentRef:
        # Two steps because a file exists independently of the stores that
        # reference it -- the same upload can be attached to several.
        uploaded = self._send(
            "openai/retrieval.uploadFile", {}, "file upload", document_file(source)
        )
        file_id = str(uploaded.get("id") or "")
        self._send(
            "openai/retrieval.attachDocument",
            {"corpusId": corpus.id, "fileId": file_id, "metadata": dict(metadata or {})},
            "attach file",
        )
        return DocumentRef(
            id=file_id, corpus_id=corpus.id, source=source, extra={"file_id": file_id}
        )

    def index_status(self, corpus: CorpusRef) -> IndexStatus:
        response = self._fetch(
            self._request("openai/retrieval.indexStatus", {"corpusId": corpus.id})
        )
        if _status(response) >= 400:
            return IndexStatus(state="error")
        data = _body(response)
        counts_raw = data.get("file_counts")
        counts = (
            IndexCounts(
                total=int(counts_raw.get("total") or 0),
                indexed=int(counts_raw.get("completed") or 0),
                failed=int(counts_raw.get("failed") or 0),
            )
            if isinstance(counts_raw, Mapping)
            else None
        )
        return IndexStatus(state=normalise_state(data.get("status")), counts=counts)

    def remove_document(self, corpus: CorpusRef, doc_id: str) -> None:
        self._send(
            "openai/retrieval.removeDocument",
            {"corpusId": corpus.id, "docId": doc_id},
            "removeDocument",
        )

    def delete_corpus(self, corpus: CorpusRef) -> None:
        self._send(
            "openai/retrieval.deleteCorpus", {"corpusId": corpus.id}, "deleteCorpus"
        )

    def list_corpora(self) -> list[CorpusRef]:
        response = self._fetch(self._request("openai/retrieval.listCorpora", {}))
        if _status(response) >= 400:
            return []
        return [
            CorpusRef(
                id=str(item.get("id") or ""),
                name=str(item.get("name") or ""),
                backend=self.backend_name,
            )
            for item in _rows(_body(response).get("data"))
        ]

    def as_tool(
        self,
        corpora: Sequence[CorpusRef],
        max_results: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """A hosted-tool entry, not a callable tool.

        `{type, params}` is this library's shape for every hosted tool -- the
        wire spec spreads `params` into the provider's own field set, which is
        why `vector_store_ids` goes inside rather than beside.
        """
        params: dict[str, Any] = {"vector_store_ids": [c.id for c in corpora]}
        if max_results is not None:
            params["max_num_results"] = max_results
        if filters is not None:
            params["filters"] = dict(filters)
        return {"type": FILE_SEARCH_TOOL_TYPE, "params": params}


class HostedGoogleRetrievalBackend(_HostedBackend):
    """Gemini File Search: upload, import, then poll a long-running operation."""

    provider = "google"
    backend_name = "hostedGoogle"
    base_url = "https://generativelanguage.googleapis.com"
    capabilities = RetrievalCapabilities(
        user_chunking=False,
        search_modes=("semantic",),
        expiration=False,
        direct_search=False,
        id_field="name",
        citation_format="gemini",
    )

    def create_corpus(self, options: CreateCorpusOptions) -> CorpusRef:
        data = self._send(
            "google/retrieval.createCorpus", {"name": options.name}, "createCorpus"
        )
        store = str(data.get("name") or "")
        return CorpusRef(id=store, name=options.name, backend=self.backend_name)

    def add_document(
        self,
        corpus: CorpusRef,
        source: DocumentSource,
        metadata: Mapping[str, Any] | None = None,
    ) -> DocumentRef:
        uploaded = self._send(
            "google/retrieval.uploadFile", {}, "file upload", document_file(source)
        )
        holder = uploaded.get("file")
        file_name = str(
            (holder.get("name") if isinstance(holder, Mapping) else None)
            or uploaded.get("name")
            or ""
        )
        operation = self._send(
            "google/retrieval.importFile",
            {"corpusId": corpus.id, "fileName": file_name, "metadata": dict(metadata or {})},
            "import file",
        )
        return DocumentRef(
            id=file_name,
            corpus_id=corpus.id,
            source=source,
            extra={"operation": operation.get("name")},
        )

    def index_status(self, corpus: CorpusRef) -> IndexStatus:
        response = self._fetch(
            self._request("google/retrieval.indexStatus", {"corpusId": corpus.id})
        )
        if _status(response) >= 400:
            return IndexStatus(state="error")
        data = _body(response)
        # Gemini reports a store that exists as usable; there is no per-file
        # count to report, so `counts` stays absent rather than guessed.
        return IndexStatus(state=normalise_state(data.get("state") or "active"))

    def remove_document(self, corpus: CorpusRef, doc_id: str) -> None:
        self._send(
            "google/retrieval.removeDocument",
            {"corpusId": corpus.id, "docId": doc_id},
            "removeDocument",
        )

    def delete_corpus(self, corpus: CorpusRef) -> None:
        self._send(
            "google/retrieval.deleteCorpus", {"corpusId": corpus.id}, "deleteCorpus"
        )

    def list_corpora(self) -> list[CorpusRef]:
        response = self._fetch(self._request("google/retrieval.listCorpora", {}))
        if _status(response) >= 400:
            return []
        data = _body(response)
        rows = _rows(data.get("fileSearchStores") or data.get("stores"))
        return [
            CorpusRef(
                id=str(item.get("name") or ""),
                name=str(item.get("displayName") or item.get("name") or ""),
                backend=self.backend_name,
            )
            for item in rows
        ]

    def as_tool(
        self, corpora: Sequence[CorpusRef], max_results: int | None = None
    ) -> dict[str, Any]:
        """Gemini names its stores rather than passing ids in a list field."""
        return {
            "type": FILE_SEARCH_TOOL_TYPE,
            "params": {"fileSearchStoreNames": [c.id for c in corpora]},
        }


class HostedXaiRetrievalBackend(_HostedBackend):
    """xAI Collections: the only hosted backend that can be searched directly.

    And the only one needing a SECOND credential: collection management uses a
    management key, while search and inference use the ordinary API key. A
    caller who has only the inference key gets a 401 from the management calls
    alone, which is a confusing way to find out.
    """

    provider = "xai"
    backend_name = "hostedXai"
    #: Versioned, like every x.ai route. The shared specs join `/collections`
    #: and `/files` onto these, so the version belongs HERE -- omitting it sends
    #: the request to an unversioned path that answers with an nginx 404 page
    #: rather than an API error.
    base_url = "https://api.x.ai/v1"
    capabilities = RetrievalCapabilities(
        user_chunking=False,
        search_modes=("hybrid",),
        expiration=False,
        direct_search=True,
        id_field="id",
        citation_format="collections-uri",
    )

    #: Collections are managed on a DIFFERENT HOST from the one that serves
    #: files and search. Both are supplied and each spec picks the pair it
    #: needs -- sending either credential to the other plane is a 401.
    management_base_url = "https://management-api.x.ai/v1"

    def __init__(
        self,
        *,
        api_key: str,
        fetch: Any,
        management_api_key: str | None = None,
        base_url: str | None = None,
        management_base_url: str | None = None,
    ) -> None:
        super().__init__(api_key=api_key, fetch=fetch, base_url=base_url)
        self._management_key = management_api_key or api_key
        self._management_base_url = management_base_url or self.management_base_url

    def _config(self) -> dict[str, Any]:
        return {
            "baseURL": self._base_url,
            "apiKey": self._api_key,
            "managementBaseURL": self._management_base_url,
            "managementApiKey": self._management_key,
        }

    def create_corpus(self, options: CreateCorpusOptions) -> CorpusRef:
        data = self._send(
            "xai/retrieval.createCorpus", {"name": options.name}, "createCorpus"
        )
        return CorpusRef(
            id=str(data.get("collection_id") or data.get("id") or ""),
            name=str(data.get("name") or options.name),
            backend=self.backend_name,
        )

    def add_document(
        self,
        corpus: CorpusRef,
        source: DocumentSource,
        metadata: Mapping[str, Any] | None = None,
    ) -> DocumentRef:
        uploaded = self._send(
            "xai/retrieval.uploadFile", {}, "file upload", document_file(source)
        )
        file_id = str(uploaded.get("file_id") or uploaded.get("id") or "")
        self._send(
            "xai/retrieval.attachDocument",
            {"corpusId": corpus.id, "fileId": file_id, "metadata": dict(metadata or {})},
            "attach file",
        )
        return DocumentRef(id=file_id, corpus_id=corpus.id, source=source)

    def index_status(self, corpus: CorpusRef) -> IndexStatus:
        """Whether these documents can actually be SEARCHED yet.

        Read from the documents, not the collection. `documents_count` reaches 1
        the moment a document is attached -- measured about five seconds before
        it can be found -- so a caller that polled exactly as the guide says
        still searched an empty index, and the model answered from its own
        knowledge with nothing to explain it.
        """
        response = self._fetch(
            self._request("xai/retrieval.listDocuments", {"corpusId": corpus.id})
        )
        if _status(response) >= 400:
            return IndexStatus(state="error")

        rows = _rows(_body(response).get("documents"))
        if not rows:
            # PENDING, not ready. The listing is briefly empty right after an
            # upload, and calling an empty corpus searchable is the same bug one
            # level down: the wait returns at once and the search finds nothing.
            return IndexStatus(state="pending", counts=IndexCounts())

        # xAI's own words -- `FILE_PROCESSING_STATUS_PROCESSED`. The shared
        # `normalise_state` map does not carry them, and its default is
        # `indexing`, so routing these through it would mean a corpus that is
        # ready never says so and every wait runs to its timeout.
        states = [str(row.get("status") or "") for row in rows]
        indexed = sum(1 for s in states if s.endswith("PROCESSED"))
        failed = sum(1 for s in states if "FAILED" in s or "ERROR" in s)
        counts = IndexCounts(total=len(states), indexed=indexed, failed=failed)

        # ANY failure is an error, not just a total one: the caller asked for
        # these documents, and searching without them quietly answers from less
        # than it was given.
        if failed:
            return IndexStatus(state="error", counts=counts)
        state: IndexState = "ready" if indexed == len(states) else "indexing"
        return IndexStatus(state=state, counts=counts)

    def remove_document(self, corpus: CorpusRef, doc_id: str) -> None:
        self._send(
            "xai/retrieval.removeDocument",
            {"corpusId": corpus.id, "docId": doc_id},
            "removeDocument",
        )

    def delete_corpus(self, corpus: CorpusRef) -> None:
        self._send("xai/retrieval.deleteCorpus", {"corpusId": corpus.id}, "deleteCorpus")

    def list_corpora(self) -> list[CorpusRef]:
        response = self._fetch(self._request("xai/retrieval.listCorpora", {}))
        if _status(response) >= 400:
            return []
        rows = _rows(_body(response).get("collections"))
        return [
            CorpusRef(
                id=str(item.get("collection_id") or item.get("id") or ""),
                name=str(item.get("name") or ""),
                backend=self.backend_name,
            )
            for item in rows
        ]

    def search(
        self,
        corpora: Sequence[CorpusRef],
        query: str,
        max_results: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for corpus in corpora:
            data = self._send(
                "xai/retrieval.search",
                {"corpusId": corpus.id, "query": query, "limit": max_results},
                "search",
            )
            for row in _rows(data.get("results") or data.get("chunks")):
                score = row.get("score")
                score = float(score) if isinstance(score, (int, float)) else 0.0
                if min_score is not None and score < min_score:
                    continue
                hits.append(
                    RetrievalHit(
                        text=str(row.get("content") or row.get("text") or ""),
                        score=score,
                        doc_id=str(row.get("file_id") or row.get("document_id") or ""),
                        citation=row.get("source_uri") or row.get("uri"),
                    )
                )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: max_results or len(hits)]

    def as_tool(
        self, corpora: Sequence[CorpusRef], max_results: int | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"vector_store_ids": [c.id for c in corpora]}
        if max_results is not None:
            params["max_num_results"] = max_results
        return {"type": FILE_SEARCH_TOOL_TYPE, "params": params}


__all__ = [
    "DEFAULT_INDEX_POLL",
    "DEFAULT_INDEX_TIMEOUT",
    "FILE_SEARCH_TOOL_TYPE",
    "HostedGoogleRetrievalBackend",
    "HostedOpenAIRetrievalBackend",
    "HostedXaiRetrievalBackend",
    "document_file",
    "normalise_state",
]
