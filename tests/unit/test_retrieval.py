"""Retrieval: chunking, vectors, and the four backends.

The requests are built from the real specs, so a URL assertion is an assertion
about what would go out -- which is how the missing `/v1` on both xAI planes was
found and fixed. Only the transport is stubbed.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

import pytest

from combycode_llm_sdk.retrieval import (
    Corpus,
    CorpusRef,
    CreateCorpusOptions,
    DocumentSource,
    HostedGoogleRetrievalBackend,
    HostedOpenAIRetrievalBackend,
    HostedXaiRetrievalBackend,
    InMemoryVectorStore,
    LocalRetrievalBackend,
    VectorEntry,
    chunk_text,
    cosine_similarity,
    format_hits,
    normalise_state,
)
from combycode_llm_sdk.retrieval.chunker import CHARS_PER_TOKEN_HEURISTIC
from combycode_llm_sdk.retrieval.hosted import document_file
from combycode_llm_sdk.retrieval.types import RetrievalHit


def _dropped(text: str, chunks: Any) -> int:
    """How many non-whitespace characters of the source reach no chunk at all.

    The single separator space between two windows is legitimately consumed by
    the walk, so only real content counts.
    """
    seen = bytearray(len(text))
    for chunk in chunks:
        for i in range(chunk.offset, min(chunk.offset + len(chunk.text), len(text))):
            seen[i] = 1
    return sum(1 for i, hit in enumerate(seen) if not hit and not text[i].isspace())


def responder(*bodies: Any) -> tuple[Any, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []
    queue = list(bodies)

    def fetch(request: dict[str, Any], _options: Any = None) -> dict[str, Any]:
        seen.append(request)
        return {"status": 200, "headers": {}, "body": queue.pop(0) if queue else {}}

    return fetch, seen


class FakeEmbedder:
    """Deterministic vectors, so a search assertion is about the search."""

    def __init__(self, table: dict[str, list[float]] | None = None) -> None:
        self.table = table or {}
        self.calls: list[list[str]] = []

    def embed(self, model: str, inputs: Any, fetch: Any) -> Any:
        texts = list(inputs)
        self.calls.append(texts)
        vectors = [self.table.get(t, [1.0, 0.0, 0.0]) for t in texts]
        return type("R", (), {"embeddings": vectors})()


class TestChunking:
    def test_short_text_is_one_chunk(self) -> None:
        chunks = chunk_text("hello world")
        assert len(chunks) == 1
        assert chunks[0].offset == 0

    def test_empty_text_is_no_chunks(self) -> None:
        assert chunk_text("") == []

    def test_long_text_is_split(self) -> None:
        chunks = chunk_text("word " * 2000, max_tokens=64, overlap_tokens=8)
        assert len(chunks) > 1

    def test_chunks_overlap(self) -> None:
        # A fact straddling a boundary is otherwise in neither chunk and
        # retrievable by neither -- the answer is in the corpus and the search
        # cannot find it.
        text = " ".join(f"w{i}" for i in range(2000))
        chunks = chunk_text(text, max_tokens=64, overlap_tokens=16)
        first_end = chunks[0].offset + len(chunks[0].text)
        assert chunks[1].offset < first_end, "the second chunk must start before the first ends"

    def test_boundaries_land_on_whitespace(self) -> None:
        # Cutting mid-word embeds a fragment as noise.
        chunks = chunk_text(" ".join(f"word{i}" for i in range(2000)), max_tokens=32)
        for chunk in chunks[:-1]:
            assert not chunk.text.endswith("wor"), "a chunk ended mid-word"

    def test_it_terminates_on_text_with_no_spaces(self) -> None:
        # No word boundary to snap to, so the window can shrink to nothing --
        # the step floor is what stops this looping forever.
        chunks = chunk_text("x" * 20000, max_tokens=16, overlap_tokens=8)
        assert len(chunks) > 1


class TestVectors:
    def test_identical_vectors_score_one(self) -> None:
        assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_a_zero_vector_scores_zero_rather_than_dividing(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_a_length_mismatch_scores_zero_rather_than_raising(self) -> None:
        # It means the corpus holds vectors from two embedding models. A bad
        # state, but the mismatched entries simply never win rather than
        # crashing every search.
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0

    def test_the_store_returns_the_nearest_first(self) -> None:
        store = InMemoryVectorStore()
        store.upsert(VectorEntry("a", "d1", "c", [1.0, 0.0], "near"))
        store.upsert(VectorEntry("b", "d2", "c", [0.0, 1.0], "far"))
        hits = store.query("c", [1.0, 0.0], 2)
        assert hits[0][0].text == "near"

    def test_entries_are_scoped_to_their_corpus(self) -> None:
        store = InMemoryVectorStore()
        store.upsert(VectorEntry("a", "d1", "one", [1.0], "in one"))
        store.upsert(VectorEntry("b", "d2", "two", [1.0], "in two"))
        assert [e.text for e, _ in store.query("one", [1.0], 5)] == ["in one"]

    def test_removing_a_document_leaves_the_others(self) -> None:
        store = InMemoryVectorStore()
        store.upsert(VectorEntry("a:0", "keep", "c", [1.0], "kept"))
        store.upsert(VectorEntry("b:0", "drop", "c", [1.0], "dropped"))
        store.remove_by_doc_id("c", "drop")
        assert store.count("c") == 1


class TestTheLocalBackend:
    def build(self, embedder: FakeEmbedder | None = None) -> LocalRetrievalBackend:
        return LocalRetrievalBackend(
            embed_adapter=embedder or FakeEmbedder(),
            fetch=None,
            embedding_model="m",
        )

    def test_a_document_is_chunked_and_embedded_in_one_call(self) -> None:
        # One embed call for the whole document: a call per chunk would be N
        # round trips for no reason.
        embedder = FakeEmbedder()
        backend = self.build(embedder)
        corpus = backend.create_corpus(CreateCorpusOptions(name="c"))
        backend.add_document(corpus, DocumentSource(text="hello world", label="d.txt"))
        assert len(embedder.calls) == 1

    def test_search_finds_the_nearer_document(self) -> None:
        embedder = FakeEmbedder(
            {"apples": [1.0, 0.0], "oranges": [0.0, 1.0], "fruit query": [0.9, 0.1]}
        )
        backend = self.build(embedder)
        corpus = backend.create_corpus(CreateCorpusOptions(name="c"))
        backend.add_document(corpus, DocumentSource(text="apples"))
        backend.add_document(corpus, DocumentSource(text="oranges"))
        hits = backend.search([corpus], "fruit query")
        assert hits[0].text == "apples"

    def test_a_citation_names_a_place_not_just_a_file(self) -> None:
        backend = self.build()
        corpus = backend.create_corpus(CreateCorpusOptions(name="c"))
        backend.add_document(corpus, DocumentSource(text="hello", label="note.txt"))
        hits = backend.search([corpus], "hello")
        assert hits[0].citation == "note.txt:0"

    def test_hits_are_reranked_across_corpora(self) -> None:
        # Taking the top-k from each and concatenating would put weak hits from
        # the first corpus ahead of a strong one from the second.
        embedder = FakeEmbedder({"weak": [0.5, 0.5], "strong": [1.0, 0.0], "q": [1.0, 0.0]})
        backend = self.build(embedder)
        first = backend.create_corpus(CreateCorpusOptions(name="a"))
        second = backend.create_corpus(CreateCorpusOptions(name="b"))
        backend.add_document(first, DocumentSource(text="weak"))
        backend.add_document(second, DocumentSource(text="strong"))
        hits = backend.search([first, second], "q")
        assert hits[0].text == "strong"

    def test_a_min_score_drops_the_rest(self) -> None:
        embedder = FakeEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0], "q": [1.0, 0.0]})
        backend = self.build(embedder)
        corpus = backend.create_corpus(CreateCorpusOptions(name="c"))
        backend.add_document(corpus, DocumentSource(text="a"))
        backend.add_document(corpus, DocumentSource(text="b"))
        assert len(backend.search([corpus], "q", min_score=0.5)) == 1

    def test_it_hands_back_a_runnable_tool(self) -> None:
        # The difference from every hosted backend: this one executes here, so
        # local retrieval works on providers with no vector store at all.
        backend = self.build()
        corpus = backend.create_corpus(CreateCorpusOptions(name="c"))
        backend.add_document(corpus, DocumentSource(text="the answer is 4217", label="n.txt"))
        tool = backend.as_tool([corpus])
        assert callable(tool.func)
        assert "4217" in tool.func(query="what is the part number")

    def test_deleting_a_corpus_drops_its_vectors(self) -> None:
        backend = self.build()
        corpus = backend.create_corpus(CreateCorpusOptions(name="c"))
        backend.add_document(corpus, DocumentSource(text="x"))
        backend.delete_corpus(corpus)
        assert backend.search([corpus], "x") == []

    def test_formatted_hits_are_numbered_and_cited(self) -> None:
        # The model is being asked to answer FROM these, and a reader of its
        # answer has to be able to check it.
        text = format_hits([RetrievalHit(text="body", score=0.5, doc_id="d", citation="n.txt:0")])
        assert "[1]" in text
        assert "n.txt:0" in text

    def test_no_hits_says_so_rather_than_returning_nothing(self) -> None:
        assert format_hits([]) == "No relevant passages found."


class TestHostedRequests:
    def test_openai_addresses_its_documented_endpoints(self) -> None:
        backend = HostedOpenAIRetrievalBackend(api_key="k", fetch=None)
        assert backend._request("openai/retrieval.createCorpus", {"name": "c"})["url"].endswith(
            "/v1/vector_stores"
        )

    def test_xai_uses_two_hosts_with_two_credentials(self) -> None:
        # Management and inference are separate planes on xAI. Sending either
        # key to the other is a 401, so the spec picks per endpoint.
        backend = HostedXaiRetrievalBackend(
            api_key="INFER", fetch=None, management_api_key="MANAGE"
        )
        create = backend._request("xai/retrieval.createCorpus", {"name": "c"})
        upload = backend._request("xai/retrieval.uploadFile", {})
        assert create["url"].startswith("https://management-api.x.ai/")
        assert "MANAGE" in create["headers"]["authorization"]
        assert upload["url"].startswith("https://api.x.ai/")
        assert "INFER" in upload["headers"]["authorization"]

    def test_readiness_is_read_from_the_documents_not_the_collection(self) -> None:
        # `documents_count` reaches 1 the moment a document is ATTACHED, about
        # five seconds before it can be searched. Asking the collection is how a
        # caller that polled correctly still searched an empty index.
        fetch, seen = responder({"documents": []})
        backend = HostedXaiRetrievalBackend(api_key="k", fetch=fetch)
        backend.index_status(CorpusRef(id="c1", name="c", backend="xai"))
        assert "/documents" in seen[0]["url"]

    def test_an_empty_listing_is_pending_not_ready(self) -> None:
        # The listing is briefly empty right after an upload. Calling that
        # searchable makes the wait return at once and the search find nothing,
        # which reads exactly like a document that did not contain the answer.
        fetch, _ = responder({"documents": []})
        backend = HostedXaiRetrievalBackend(api_key="k", fetch=fetch)
        assert backend.index_status(CorpusRef(id="c1", name="c", backend="xai")).state == "pending"

    def test_a_processed_document_is_ready(self) -> None:
        # xAI's own word, which the shared status map does not carry -- routing
        # it through that map would read as `indexing` and never finish.
        fetch, _ = responder({"documents": [{"status": "FILE_PROCESSING_STATUS_PROCESSED"}]})
        backend = HostedXaiRetrievalBackend(api_key="k", fetch=fetch)
        status = backend.index_status(CorpusRef(id="c1", name="c", backend="xai"))
        assert status.state == "ready"
        assert status.counts is not None
        assert (status.counts.total, status.counts.indexed) == (1, 1)

    def test_a_document_still_processing_is_not_ready(self) -> None:
        fetch, _ = responder({"documents": [{"status": "FILE_PROCESSING_STATUS_PROCESSING"}]})
        backend = HostedXaiRetrievalBackend(api_key="k", fetch=fetch)
        assert backend.index_status(CorpusRef(id="c1", name="c", backend="xai")).state == "indexing"

    def test_one_failure_among_several_is_an_error(self) -> None:
        # The caller asked for these documents; searching without them quietly
        # answers from less than it was given.
        fetch, _ = responder(
            {
                "documents": [
                    {"status": "FILE_PROCESSING_STATUS_PROCESSED"},
                    {"status": "FILE_PROCESSING_STATUS_FAILED"},
                ]
            }
        )
        backend = HostedXaiRetrievalBackend(api_key="k", fetch=fetch)
        status = backend.index_status(CorpusRef(id="c1", name="c", backend="xai"))
        assert status.state == "error"
        assert status.counts is not None
        assert status.counts.failed == 1

    def test_a_refused_listing_is_an_error(self) -> None:
        def fetch(request: dict[str, Any], _options: Any = None) -> dict[str, Any]:
            return {"status": 500, "headers": {}, "body": {}}

        backend = HostedXaiRetrievalBackend(api_key="k", fetch=fetch)
        assert backend.index_status(CorpusRef(id="c1", name="c", backend="xai")).state == "error"

    def test_both_xai_planes_are_versioned(self) -> None:
        # The version lives in the BASE URL, because the shared specs join
        # `/collections` and `/files` onto it. Defaulting to the bare host sends
        # the request to an unversioned path that answers with an nginx 404
        # page rather than an API error -- measured live 2026-09-04.
        backend = HostedXaiRetrievalBackend(api_key="k", fetch=None)
        for spec in ("xai/retrieval.createCorpus", "xai/retrieval.uploadFile"):
            url = backend._request(spec, {"name": "c"})["url"]
            assert "/v1/" in url, spec
            assert "/v1/v1/" not in url, f"{spec} doubled the version"

    def test_a_document_uploads_under_a_reproducible_name(self) -> None:
        # A content hash, not a random id: a retried upload has to arrive under
        # the same name or it becomes a second document.
        one = document_file(DocumentSource(text="same text"))
        two = document_file(DocumentSource(text="same text"))
        assert one.filename == two.filename
        assert document_file(DocumentSource(text="other")).filename != one.filename

    def test_a_label_is_used_when_given(self) -> None:
        assert document_file(DocumentSource(text="x", label="note.txt")).filename == "note.txt"


class TestHostedToolSpecs:
    def test_openai_puts_its_ids_under_params(self) -> None:
        # `{type, params}` is this library's hosted-tool shape; the wire spec
        # spreads `params` into the provider's own field set.
        spec = HostedOpenAIRetrievalBackend(api_key="k", fetch=None).as_tool(
            [CorpusRef(id="vs_1", name="c", backend="hostedOpenAI")]
        )
        assert spec["type"] == "file_search"
        assert spec["params"]["vector_store_ids"] == ["vs_1"]

    def test_google_names_its_stores_instead(self) -> None:
        spec = HostedGoogleRetrievalBackend(api_key="k", fetch=None).as_tool(
            [CorpusRef(id="fileSearchStores/s1", name="c", backend="hostedGoogle")]
        )
        assert spec["params"]["fileSearchStoreNames"] == ["fileSearchStores/s1"]


class TestIndexState:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("completed", "ready"),
            ("in_progress", "indexing"),
            ("failed", "error"),
            ("queued", "pending"),
        ],
    )
    def test_each_provider_word_maps_to_ours(self, raw: str, expected: str) -> None:
        assert normalise_state(raw) == expected

    def test_an_unknown_word_reads_as_indexing_not_ready(self) -> None:
        # Calling a corpus searchable when it may not be produces an empty
        # answer that looks like a missing document.
        assert normalise_state("something-new") == "indexing"
        assert normalise_state(None) == "indexing"


class TestCorpusHandle:
    def test_it_refuses_a_provider_with_no_corpus_api(self) -> None:
        # Indexing a caller's documents at a provider they did not name is a
        # data decision this library has no right to make.
        with pytest.raises(ValueError, match="hosts no corpus API"):
            Corpus.create(name="c", model="anthropic/claude-haiku-4.5", api_key="k")

    def test_it_needs_a_provider(self) -> None:
        with pytest.raises(ValueError, match="name the provider"):
            Corpus.create(name="c", model="gpt-5", api_key="k")

    def test_leaving_the_block_deletes_what_it_created(self) -> None:
        # A hosted corpus is a standing charge; one nobody deletes is a bill
        # nobody is reading.
        fetch, seen = responder({"id": "vs_1", "name": "c"}, {})
        backend = HostedOpenAIRetrievalBackend(api_key="k", fetch=fetch)
        ref = backend.create_corpus(CreateCorpusOptions(name="c"))
        with Corpus(backend, ref, owns=True):
            pass
        assert seen[-1]["method"] == "DELETE"

    def test_an_adopted_corpus_is_not_deleted(self) -> None:
        # It was not ours to make, so it is not ours to destroy.
        fetch, seen = responder({})
        backend = HostedOpenAIRetrievalBackend(api_key="k", fetch=fetch)
        with Corpus(backend, CorpusRef(id="vs_1", name="c", backend="hostedOpenAI")):
            pass
        assert seen == []


class TestDirectSearch:
    def test_openai_says_it_cannot_rather_than_failing_obscurely(self) -> None:
        assert HostedOpenAIRetrievalBackend.capabilities.direct_search is False
        with pytest.raises(RuntimeError, match="direct search is not supported"):
            HostedOpenAIRetrievalBackend(api_key="k", fetch=None).search([], "q")

    def test_xai_is_the_one_hosted_backend_that_can(self) -> None:
        assert HostedXaiRetrievalBackend.capabilities.direct_search is True

    def test_xai_search_reads_the_hits_back(self) -> None:
        fetch, _ = responder(
            {"results": [{"content": "part 4217", "score": 0.9, "file_id": "f1"}]}
        )
        backend = HostedXaiRetrievalBackend(api_key="k", fetch=fetch)
        hits = backend.search([CorpusRef(id="c1", name="c", backend="hostedXai")], "part")
        assert hits[0].text == "part 4217"
        assert hits[0].doc_id == "f1"


class TestFailuresNameTheBackend:
    def test_a_4xx_says_which_backend_and_which_step(self) -> None:
        def failing(request: dict[str, Any], _options: Any = None) -> dict[str, Any]:
            return {"status": 400, "body": json.dumps({"error": "nope"})}

        with pytest.raises(RuntimeError, match="hostedOpenAI: createCorpus failed"):
            HostedOpenAIRetrievalBackend(api_key="k", fetch=failing).create_corpus(
                CreateCorpusOptions(name="c")
            )


class TestNoWhitespaceDocuments:
    """CJK prose, minified JSON, one very long token.

    The chunker snaps to whitespace, and text with none used to advance
    straight to the end -- one chunk, and every character after it dropped
    without a word. Found by a test that only checked termination.
    """

    def test_the_whole_document_is_covered(self) -> None:
        text = "x" * 20000
        chunks = chunk_text(text, max_tokens=16, overlap_tokens=4)
        assert len(chunks) > 1
        # The last chunk must reach the end, or the tail was silently lost.
        last = chunks[-1]
        assert last.offset + len(last.text) >= len(text)

    def test_cjk_text_is_not_truncated_to_one_chunk(self) -> None:
        text = "这是一个没有空格的长文档。" * 500
        chunks = chunk_text(text, max_tokens=32, overlap_tokens=4)
        assert len(chunks) > 1
        assert chunks[-1].offset + len(chunks[-1].text) >= len(text)

    def test_a_space_free_run_in_the_middle_is_not_stepped_over(self) -> None:
        # The harder half: with prose on the far side of the blob there IS a
        # next space, thousands of characters away, and snapping to it used to
        # step over the whole blob. Covering only the tail case missed this.
        prose = " ".join(f"word{i}" for i in range(400))
        text = f"{prose} " + "Q" * 20000 + f" {prose}"
        chunks = chunk_text(text, max_tokens=512, overlap_tokens=64)

        assert _dropped(text, chunks) == 0, "characters reached no chunk"
        inside = [c for c in chunks if c.text.startswith("Q") and len(c.text) > 1000]
        assert len(inside) > 5, "the blob was covered but not windowed"

    def test_the_cursor_never_lands_beyond_the_chunk_it_just_emitted(self) -> None:
        # The invariant coverage rests on. One separator space is legitimately
        # consumed by the walk, so +1 is the limit.
        prose = " ".join(f"word{i}" for i in range(300))
        text = f"{prose} " + "Z" * 9000 + f" {prose}"
        chunks = chunk_text(text, max_tokens=256, overlap_tokens=32)
        for previous, following in itertools.pairwise(chunks):
            assert following.offset <= previous.offset + len(previous.text) + 1

    def test_a_document_that_turns_space_free_keeps_its_tail(self) -> None:
        # The realistic shape: prose, then an embedded blob. Everything after
        # the last space is the part that used to vanish.
        prose = " ".join(f"word{i}" for i in range(40))
        text = f"{prose} " + "q" * 2000
        chunks = chunk_text(text, max_tokens=25, overlap_tokens=4)

        assert _dropped(text, chunks) == 0, "characters reached no chunk"


class TestTheApproachToASpaceFreeRun:
    """Prose, then a base64 image, then prose -- the shape of a README.

    The step is derived from the SNAPPED window length, so a window trimmed back
    hard (its only space near its start, which is what the last window before a
    blob looks like) left a step below the overlap, and the `max(step, 1)` floor
    became the actual step. The walk then crawled a word at a time across the
    whole approach: 42 of 58 chunks under half the budget, the smallest 8
    characters, each embedded and each crowding the others out of the results.
    """

    def readme(self) -> str:
        prose = " ".join(f"word{i}" for i in range(400))
        return f"{prose} ![logo](data:image/png;base64,{'Q' * 20000}) {prose}"

    def test_no_runt_chunks_where_the_document_turns_space_free(self) -> None:
        text = self.readme()
        chunks = chunk_text(text, max_tokens=512, overlap_tokens=64)
        budget = 512 * CHARS_PER_TOKEN_HEURISTIC
        runts = [c for c in chunks[:-1] if len(c.text) < budget / 2]

        assert runts == []
        assert len(chunks) < 20
        assert _dropped(text, chunks) == 0

    def test_a_token_is_cut_only_where_no_word_is_within_a_window(self) -> None:
        # What the refusal costs, and the proof it is only ever spent inside a
        # run with no space in it: a window ending in prose always has a space in
        # its second half, so the snap is taken there.
        text = self.readme()
        budget = 512 * CHARS_PER_TOKEN_HEURISTIC
        for chunk in chunk_text(text, max_tokens=512, overlap_tokens=64):
            end = chunk.offset + len(chunk.text)
            if end == len(text) or text[end] == " " or text[end - 1] == " ":
                continue
            behind = text[max(0, end - budget // 2) : end]
            assert " " not in behind, "a word was split where one existed"

    def test_ordinary_prose_is_left_alone(self) -> None:
        text = " ".join(f"word{i}" for i in range(6000))
        chunks = chunk_text(text, max_tokens=512, overlap_tokens=64)

        assert _dropped(text, chunks) == 0
        for chunk in chunks:
            end = chunk.offset + len(chunk.text)
            assert end == len(text) or text[end] == " "


class TestTheTail:
    """The end of a document belongs to exactly one chunk.

    A walk that keeps going once a window has reached the end re-emits the
    same tail as a run of ever-shorter chunks. Nothing is lost -- but every
    duplicate is embedded at the caller's expense and then competes with the
    others for one of the few result slots a search returns.
    """

    def test_only_the_last_chunk_ends_at_the_end(self) -> None:
        text = " ".join(f"word{i}" for i in range(400))
        chunks = chunk_text(text, max_tokens=50, overlap_tokens=10)

        ending_at_end = [c for c in chunks if c.offset + len(c.text) == len(text)]
        assert len(ending_at_end) == 1
        assert ending_at_end[0] is chunks[-1]
