"""Text in, vector out.

Nothing here mocks the request BUILD -- the specs are the real ones, so an
assertion about the URL or the body is an assertion about what would actually go
out. Only the transport is stubbed, because the vectors themselves are the one
part a provider owns and a test cannot.

The two provider shapes are the thing worth pinning: OpenAI takes a whole batch
in one request, Google takes one text per request and loops. A caller embedding
500 chunks pays two orders of magnitude differently between them.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from combycode_llm_sdk.embeddings import (
    EmbedResult,
    GoogleEmbeddingAdapter,
    OpenAIEmbeddingAdapter,
    OpenRouterEmbeddingAdapter,
    embed,
    embedding_adapter,
)
from combycode_llm_sdk.transport import TransportResponse


def openai_body(dims: int = 3, count: int = 1, tokens: int = 7) -> dict[str, Any]:
    return {
        "data": [
            {"embedding": [float(i + j) for j in range(dims)], "index": i} for i in range(count)
        ],
        "usage": {"prompt_tokens": tokens},
    }


def google_body(dims: int = 3) -> dict[str, Any]:
    return {"embedding": {"values": [float(j) for j in range(dims)]}}


def recorder(body: Any, status: int = 200) -> tuple[Any, list[dict[str, Any]]]:
    """A fetch that records what it was asked to send."""
    seen: list[dict[str, Any]] = []

    def fetch(request: dict[str, Any], _options: Any = None) -> dict[str, Any]:
        seen.append(request)
        return {"status": status, "body": body(len(seen)) if callable(body) else body}

    return fetch, seen


class TestTheOpenAIShape:
    def test_the_whole_batch_goes_in_one_request(self) -> None:
        fetch, seen = recorder(openai_body(count=3))
        adapter = OpenAIEmbeddingAdapter("k")
        result = adapter.embed("text-embedding-3-small", ["a", "b", "c"], fetch)
        assert len(seen) == 1
        assert len(result.embeddings) == 3

    def test_a_single_string_is_still_an_array_on_the_wire(self) -> None:
        # The rule the spec carries rather than a ternary nobody re-reads.
        fetch, seen = recorder(openai_body())
        OpenAIEmbeddingAdapter("k").embed("m", ["only one"], fetch)
        assert seen[0]["body"]["input"] == ["only one"]

    def test_it_addresses_the_documented_endpoint(self) -> None:
        request = OpenAIEmbeddingAdapter("k").build_request("m", ["x"])
        assert request["url"] == "https://api.openai.com/v1/embeddings"
        assert request["headers"]["authorization"] == "Bearer k"
        assert request["method"] == "POST"

    def test_a_base_url_is_honoured(self) -> None:
        request = OpenAIEmbeddingAdapter("k", "https://proxy.internal").build_request("m", ["x"])
        assert request["url"] == "https://proxy.internal/v1/embeddings"

    def test_dimensions_come_from_the_vector_that_arrived(self) -> None:
        # Not from a config: the number that matters is the one the provider
        # actually returned, which is how a caller catches a model swap.
        fetch, _ = recorder(openai_body(dims=8))
        result = OpenAIEmbeddingAdapter("k").embed("m", ["x"], fetch)
        assert result.dimensions == 8

    def test_usage_is_carried_when_the_provider_reports_it(self) -> None:
        fetch, _ = recorder(openai_body(tokens=42))
        result = OpenAIEmbeddingAdapter("k").embed("m", ["x"], fetch)
        assert result.usage is not None
        assert result.usage.input_tokens == 42

    def test_the_request_can_be_inspected_without_being_sent(self) -> None:
        # `build_request` is public for the same reason the chat adapters expose
        # theirs: a caller debugging a 400 needs to see what we built.
        request = OpenAIEmbeddingAdapter("k").build_request("m", ["a", "b"])
        assert json.loads(json.dumps(request["body"]))["input"] == ["a", "b"]


class TestTheGoogleShape:
    def test_a_batch_is_a_loop_of_single_calls(self) -> None:
        # `embedContent` takes one text, so three inputs are three requests.
        # Worth asserting rather than assuming: it is the cost difference.
        fetch, seen = recorder(google_body())
        result = GoogleEmbeddingAdapter("k").embed("gemini-embedding-001", ["a", "b", "c"], fetch)
        assert len(seen) == 3
        assert len(result.embeddings) == 3

    def test_each_request_carries_only_its_own_text(self) -> None:
        fetch, seen = recorder(google_body())
        GoogleEmbeddingAdapter("k").embed("m", ["first", "second"], fetch)
        assert seen[0]["body"]["content"]["parts"][0]["text"] == "first"
        assert seen[1]["body"]["content"]["parts"][0]["text"] == "second"

    def test_it_addresses_the_documented_endpoint(self) -> None:
        request = GoogleEmbeddingAdapter("k").build_request("gemini-embedding-001", "x")
        assert request["url"].endswith("/v1beta/models/gemini-embedding-001:embedContent")
        assert request["headers"]["x-goog-api-key"] == "k"

    def test_no_usage_is_reported_rather_than_a_false_zero(self) -> None:
        # `embedContent` reports none. A zero would put a line in the ledger
        # saying the call was free, which is not what the provider said.
        fetch, _ = recorder(google_body())
        result = GoogleEmbeddingAdapter("k").embed("m", ["x"], fetch)
        assert result.usage is None


class TestOpenRouterIsTheSameWireElsewhere:
    def test_only_the_url_differs(self) -> None:
        openai = OpenAIEmbeddingAdapter("k").build_request("m", ["x"])
        router = OpenRouterEmbeddingAdapter("k").build_request("m", ["x"])
        assert router["url"] == "https://openrouter.ai/api/v1/embeddings"
        assert router["body"] == openai["body"]
        assert router["provider"] == "openrouter"


class TestPickingAnAdapter:
    def test_each_supported_provider_has_one(self) -> None:
        for provider in ("openai", "google", "openrouter"):
            assert embedding_adapter(provider, "k").name == provider

    @pytest.mark.parametrize("provider", ["anthropic", "xai"])
    def test_a_provider_with_no_embeddings_api_is_refused_by_name(self, provider: str) -> None:
        # Anthropic and xAI serve none at all. Falling back to a provider the
        # caller did not name would spend their money somewhere unexpected.
        with pytest.raises(ValueError, match="serves no embeddings API"):
            embedding_adapter(provider, "k")


class TestTheEntryPoint:
    def test_a_namespaced_model_routes_to_its_own_provider(self) -> None:
        sent: list[Any] = []

        def stub(request: Any) -> TransportResponse:
            sent.append(request)
            return TransportResponse(status=200, body=json.dumps(openai_body(dims=4)))

        result = embed(
            model="openai/text-embedding-3-small",
            input="hello",
            api_key="k",
            transport=stub,
        )
        assert sent[0].url == "https://api.openai.com/v1/embeddings"
        assert result.dimensions == 4
        assert result.model == "text-embedding-3-small"

    def test_google_routes_to_google(self) -> None:
        def stub(request: Any) -> TransportResponse:
            assert "generativelanguage" in request.url
            return TransportResponse(status=200, body=json.dumps(google_body()))

        result = embed(
            model="google/gemini-embedding-001", input=["a", "b"], api_key="k", transport=stub
        )
        assert len(result.embeddings) == 2

    def test_a_bare_model_needs_a_provider(self) -> None:
        with pytest.raises(ValueError, match="name the provider"):
            embed(model="text-embedding-3-small", input="x", api_key="k")

    def test_a_missing_key_is_refused_before_any_request(self) -> None:
        with pytest.raises(ValueError, match="no API key"):
            embed(model="openai/text-embedding-3-small", input="x")

    def test_an_empty_batch_sends_nothing(self) -> None:
        # A request with an empty array is a 400 on OpenAI and a wasted round
        # trip everywhere else.
        result = embed(model="openai/text-embedding-3-small", input=[], api_key="k")
        assert result == EmbedResult(model="text-embedding-3-small")

    def test_a_single_string_still_returns_a_list_of_vectors(self) -> None:
        # Returning a bare vector for one input and a list for two would make
        # every caller branch on what they passed.
        fetch, _ = recorder(openai_body())
        result = OpenAIEmbeddingAdapter("k").embed("m", ["one"], fetch)
        assert len(result.embeddings) == 1
        assert isinstance(result.embeddings[0], tuple)


class TestFailures:
    def test_a_4xx_names_the_provider(self) -> None:
        fetch, _ = recorder({"error": {"message": "bad model"}}, status=400)
        with pytest.raises(RuntimeError, match="'openai' failed \\(400\\)"):
            OpenAIEmbeddingAdapter("k").embed("m", ["x"], fetch)

    def test_a_malformed_body_yields_no_vectors_rather_than_raising(self) -> None:
        # A provider that answered 200 with a shape we do not recognise has not
        # failed the REQUEST; the caller sees an empty result and can act.
        fetch, _ = recorder({"data": "not a list"})
        result = OpenAIEmbeddingAdapter("k").embed("m", ["x"], fetch)
        assert result.embeddings == ()
        assert result.dimensions == 0
