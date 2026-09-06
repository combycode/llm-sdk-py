"""One handle over four backends.

`Corpus.create(model=...)` picks the backend from the provider, so a caller
writes the same five lines whether the index lives at OpenAI, at Google, at xAI
or in this process. The differences are still there -- they are in
`backend.capabilities` -- but they stop being something you have to know before
you can start.

A context manager, deliberately. A hosted corpus is a standing charge: it
survives the process, and one nobody deletes is a bill nobody is reading. So
`with` owns what `create()` made, and leaving the block deletes it.

Transposed from `unified-library-ts/src/helpers/collection.ts`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Self

from .hosted import (
    DEFAULT_INDEX_POLL,
    DEFAULT_INDEX_TIMEOUT,
    HostedGoogleRetrievalBackend,
    HostedOpenAIRetrievalBackend,
    HostedXaiRetrievalBackend,
)
from .local import LocalRetrievalBackend
from .types import CorpusRef, CreateCorpusOptions, DocumentSource, IndexStatus, RetrievalHit

#: Which hosted backend each provider gets. Anthropic and OpenRouter host no
#: corpus API at all -- they are absent rather than mapped to something else,
#: because silently indexing a caller's documents at a provider they did not
#: name would be a data decision this library has no right to make.
HOSTED_BACKENDS: Mapping[str, type[Any]] = {
    "openai": HostedOpenAIRetrievalBackend,
    "google": HostedGoogleRetrievalBackend,
    "xai": HostedXaiRetrievalBackend,
}


class Corpus:
    """Documents a provider indexes, asked about through a tool."""

    def __init__(self, backend: Any, ref: CorpusRef, owns: bool = False) -> None:
        self.backend = backend
        self.ref = ref
        #: Whether leaving a `with` block should delete it. True only when this
        #: handle created it: adopting an existing corpus must not delete it.
        self._owns = owns

    # -- construction --------------------------------------------------------

    @staticmethod
    def create(
        *,
        name: str,
        model: str = "",
        api_key: str | None = None,
        provider: str | None = None,
        management_api_key: str | None = None,
        engine: Any = None,
        transport: Any = None,
        base_url: str | None = None,
        embedding_model: str | None = None,
        expires_after_days: int | None = None,
    ) -> Corpus:
        """Create a corpus at the provider the model belongs to."""
        from ..helpers.client_resolver import is_namespaced_model_id, parse_model_id

        provider_name = (
            parse_model_id(model)[0] if model and is_namespaced_model_id(model) else provider
        )
        if not provider_name:
            raise ValueError(
                'Corpus.create: name the provider, either as provider= or as '
                '"provider/model" in model=.'
            )
        factory = HOSTED_BACKENDS.get(provider_name)
        if factory is None:
            raise ValueError(
                f"Corpus.create: {provider_name!r} hosts no corpus API. "
                f"Available: {', '.join(sorted(HOSTED_BACKENDS))}. For a provider "
                "without one, index locally with LocalRetrievalBackend."
            )

        key = api_key or (engine.api_keys.get(provider_name) if engine is not None else None)
        if not key:
            raise ValueError(
                f'Corpus.create: no API key for provider "{provider_name}".'
            )

        options: dict[str, Any] = {"api_key": key, "fetch": _fetch(engine, transport)}
        if base_url:
            options["base_url"] = base_url
        if provider_name == "xai":
            # xAI manages collections under a DIFFERENT credential from the one
            # that runs inference. Passing the inference key here 401s on the
            # management calls only, which is a confusing way to find out.
            options["management_api_key"] = management_api_key
        backend = factory(**options)
        ref = backend.create_corpus(
            CreateCorpusOptions(
                name=name,
                embedding_model=embedding_model,
                expires_after_days=expires_after_days,
            )
        )
        return Corpus(backend, ref, owns=True)

    @staticmethod
    def local(
        *,
        name: str,
        embed_adapter: Any,
        embedding_model: str,
        fetch: Any,
        vector_store: Any = None,
    ) -> Corpus:
        """A corpus indexed HERE, which works on every provider."""
        backend = LocalRetrievalBackend(
            embed_adapter=embed_adapter,
            fetch=fetch,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )
        return Corpus(backend, backend.create_corpus(CreateCorpusOptions(name=name)), owns=True)

    # -- documents -----------------------------------------------------------

    def add_document(
        self,
        text: str,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        return self.backend.add_document(
            self.ref, DocumentSource(text=text, label=label), metadata
        )

    def remove_document(self, doc_id: str) -> None:
        self.backend.remove_document(self.ref, doc_id)

    def index_status(self) -> IndexStatus:
        status: IndexStatus = self.backend.index_status(self.ref)
        return status

    def wait_indexed(
        self, timeout: float = DEFAULT_INDEX_TIMEOUT, poll: float = DEFAULT_INDEX_POLL
    ) -> IndexStatus:
        """Block until the provider will actually search these documents.

        Not optional politeness: asking early searches an empty index and
        returns an answer built on nothing, which reads exactly like a document
        that did not contain what was asked.
        """
        status: IndexStatus = self.backend.wait_indexed(self.ref, timeout, poll)
        return status

    # -- using it ------------------------------------------------------------

    def as_tool(self, max_results: int | None = None) -> Any:
        """The tool to pass to `complete(tools=[...])`.

        Local hands back a callable this library runs; hosted hands back a spec
        the provider runs. Both go in the same list, which is the point.
        """
        return self.backend.as_tool([self.ref], max_results)

    def search(
        self,
        query: str,
        max_results: int | None = None,
        min_score: float | None = None,
    ) -> Sequence[RetrievalHit]:
        hits: Sequence[RetrievalHit] = self.backend.search(
            [self.ref], query, max_results, min_score
        )
        return hits

    def delete(self) -> None:
        self.backend.delete_corpus(self.ref)

    # -- ownership -----------------------------------------------------------

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        if self._owns:
            self.delete()

    def __repr__(self) -> str:
        return f"<Corpus {self.ref.name!r} on {self.ref.backend}>"


def _fetch(engine: Any, transport: Any) -> Any:
    from ..bus.hook_bus import HookBus
    from ..network.executor import RequestExecutor
    from ..network.retry import DEFAULT_RETRY
    from ..transport import as_fetch, http_transport

    if engine is not None and transport is None:
        return engine.fetch
    send = as_fetch(transport or http_transport())
    executor = RequestExecutor(engine.hooks if engine is not None else HookBus())

    def fetch(req: Any, options: Any = None) -> Any:
        return executor.execute(req, lambda r: send(r), DEFAULT_RETRY)

    return fetch


__all__ = ["HOSTED_BACKENDS", "Corpus"]
