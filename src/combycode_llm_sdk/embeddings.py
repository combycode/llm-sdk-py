"""Text in, vector out.

The thinnest of the provider subsystems, and the shape the media and files ones
follow: a per-provider adapter that builds its request FROM THE WIRE SPEC and
sends it through the injected fetch, so an embedding call rides the same queue,
retry policy and hooks as everything else this library sends. Never a
side-fetch.

Two provider shapes, and the difference is not cosmetic:

- **OpenAI (and OpenRouter, which is the same wire at a different URL)** embeds a
  whole BATCH in one request. `input` is always an array on the wire even when
  the caller passed one string.
- **Google** embeds ONE text per request, so a batch is a loop. Which means the
  cost of embedding 500 chunks differs by two orders of magnitude between them,
  and a caller choosing a provider should know that from the docs rather than
  from a bill.

Transposed from `unified-library-ts/src/plugins/embeddings/` and the three
`llm/providers/*/embeddings.ts` adapters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .llm.wire_transforms import make_registry
from .wire.interpreter import build_from_spec
from .wire.service_specs import service_spec

#: Where each provider serves embeddings, when the caller names no base URL.
DEFAULT_BASE_URLS: Mapping[str, str] = {
    "openai": "https://api.openai.com",
    "google": "https://generativelanguage.googleapis.com",
    "openrouter": "https://openrouter.ai",
}


@dataclass(frozen=True)
class EmbedUsage:
    """What the call cost, when the provider says.

    Google's `embedContent` reports no usage at all, so this is optional rather
    than defaulted to zero -- "the provider did not say" and "it cost nothing"
    are different facts and a ledger must not confuse them.
    """

    input_tokens: int


@dataclass(frozen=True)
class EmbedResult:
    """One vector per input, in the order the inputs were given."""

    embeddings: Sequence[Sequence[float]] = field(default_factory=tuple)
    model: str = ""
    #: Vector length, or 0 when nothing came back. Read from the FIRST vector
    #: rather than from a config: the number that matters is the one the
    #: provider actually returned.
    dimensions: int = 0
    usage: EmbedUsage | None = None


class EmbeddingAdapter(Protocol):
    """One provider's embeddings endpoint."""

    name: str

    def embed(self, model: str, inputs: Sequence[str], fetch: Any) -> EmbedResult: ...


def _rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [v for v in value if isinstance(v, Mapping)]


def _floats(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [float(v) for v in value if isinstance(v, (int, float)) and not isinstance(v, bool)]


def _send(fetch: Any, request: Mapping[str, Any]) -> Mapping[str, Any]:
    """One request, with a failure that names the provider rather than the shape.

    The body is decoded here when it arrives as text. This library's own
    transport parses JSON already, but a caller may bring one that does not --
    and a raw string reaching the parse below would silently produce zero
    vectors rather than an error anyone could act on.
    """
    response = fetch(dict(request))
    status = response.get("status") if isinstance(response, Mapping) else 0
    body = (response.get("body") if isinstance(response, Mapping) else None) or {}
    if isinstance(body, (str, bytes)):
        import json

        try:
            body = json.loads(body)
        except ValueError:
            # Not JSON at all -- a 500 served as an HTML error page, say. Left
            # as text so the failure below can quote it.
            pass
    if not isinstance(status, int) or status >= 400:
        raise RuntimeError(
            f"embeddings request to {request.get('provider')!r} failed ({status}): {body!r}"
        )
    return body if isinstance(body, Mapping) else {}


class OpenAIEmbeddingAdapter:
    """`POST /v1/embeddings` -- the whole batch in one request."""

    #: The spec that builds the request. OpenRouter overrides just this, because
    #: the whole difference between them is the URL.
    spec_id = "openai/embeddings"

    def __init__(self, api_key: str, base_url: str | None = None, name: str = "openai") -> None:
        self.name = name
        self._api_key = api_key
        self._base_url = base_url or DEFAULT_BASE_URLS.get(name) or DEFAULT_BASE_URLS["openai"]
        self._registry = make_registry({})

    def build_request(self, model: str, inputs: Sequence[str]) -> dict[str, Any]:
        """The request, built and inspectable without performing it."""
        built = build_from_spec(
            service_spec(self.spec_id),
            {"model": model, "input": list(inputs)},
            self._registry,
            self.name,
            None,
            {"baseURL": self._base_url, "apiKey": self._api_key},
        )
        return {
            "url": built.url,
            "method": built.method or "POST",
            "headers": dict(built.headers or {}),
            "body": built.body,
            "provider": self.name,
            "model": model,
            "responseType": "json",
        }

    def embed(self, model: str, inputs: Sequence[str], fetch: Any) -> EmbedResult:
        body = _send(fetch, self.build_request(model, inputs))
        vectors = [_floats(row.get("embedding")) for row in _rows(body.get("data"))]
        usage = body.get("usage")
        return EmbedResult(
            embeddings=tuple(tuple(v) for v in vectors),
            model=model,
            dimensions=len(vectors[0]) if vectors else 0,
            usage=EmbedUsage(int(usage.get("prompt_tokens") or 0))
            if isinstance(usage, Mapping)
            else None,
        )


class OpenRouterEmbeddingAdapter(OpenAIEmbeddingAdapter):
    """The same wire, a different host and path."""

    spec_id = "openrouter/embeddings"

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        super().__init__(api_key, base_url or DEFAULT_BASE_URLS["openrouter"], "openrouter")


class GoogleEmbeddingAdapter:
    """`:embedContent` -- ONE text per request, so a batch is a loop."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self.name = "google"
        self._api_key = api_key
        self._base_url = base_url or DEFAULT_BASE_URLS["google"]
        self._registry = make_registry({})

    def build_request(self, model: str, text: str) -> dict[str, Any]:
        """One request, for ONE input text."""
        built = build_from_spec(
            service_spec("google/embeddings"),
            {"model": model, "text": text},
            self._registry,
            "google",
            None,
            {"baseURL": self._base_url, "apiKey": self._api_key},
        )
        return {
            "url": built.url,
            "method": built.method or "POST",
            "headers": dict(built.headers or {}),
            "body": built.body,
            "provider": "google",
            "model": model,
            "responseType": "json",
        }

    def embed(self, model: str, inputs: Sequence[str], fetch: Any) -> EmbedResult:
        vectors: list[tuple[float, ...]] = []
        for text in inputs:
            body = _send(fetch, self.build_request(model, text))
            embedding = body.get("embedding")
            values = embedding.get("values") if isinstance(embedding, Mapping) else None
            vectors.append(tuple(_floats(values)))
        # No usage: `embedContent` reports none, and inventing a zero would put a
        # false line in the ledger.
        return EmbedResult(
            embeddings=tuple(vectors),
            model=model,
            dimensions=len(vectors[0]) if vectors else 0,
        )


#: Which adapter serves which provider.
ADAPTERS: Mapping[str, type[Any]] = {
    "openai": OpenAIEmbeddingAdapter,
    "google": GoogleEmbeddingAdapter,
    "openrouter": OpenRouterEmbeddingAdapter,
}


def embedding_adapter(
    provider: str, api_key: str, base_url: str | None = None
) -> EmbeddingAdapter:
    """The adapter for one provider, or a refusal naming the ones that exist.

    Anthropic and xAI serve no embeddings API at all, so there is nothing to
    fall back to -- and picking a different provider silently would spend the
    caller's money somewhere they did not ask for.
    """
    factory = ADAPTERS.get(provider)
    if factory is None:
        raise ValueError(
            f"embed: {provider!r} serves no embeddings API. "
            f"Available: {', '.join(sorted(ADAPTERS))}."
        )
    adapter: EmbeddingAdapter = factory(api_key, base_url)
    return adapter


def embed(
    *,
    model: str,
    input: str | Sequence[str],
    # word every provider uses on the wire.
    api_key: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    engine: Any = None,
    transport: Any = None,
) -> EmbedResult:
    """Embed one string or a batch.

        vectors = embed(model="openai/text-embedding-3-small", input="hello")
        vectors.embeddings[0]        # the vector
        vectors.dimensions           # how long it is

    A single string still comes back as a LIST of one vector. Returning a bare
    vector for one input and a list for two would make every caller branch on
    what they passed.
    """
    from .helpers.client_resolver import is_namespaced_model_id, parse_model_id

    provider_name, model_name = (
        parse_model_id(model) if is_namespaced_model_id(model) else (provider, model)
    )
    if not provider_name:
        raise ValueError(
            'embed: name the provider, either as `provider=` or as "provider/model".'
        )

    key = api_key or (engine.api_keys.get(provider_name) if engine is not None else None)
    if not key:
        raise ValueError(
            f'embed: no API key for provider "{provider_name}". Pass api_key= or engine.api_keys.'
        )

    inputs = [input] if isinstance(input, str) else list(input)
    if not inputs:
        # Nothing to send, so nothing is sent. A request with an empty array is
        # a 400 on OpenAI and a wasted round trip everywhere else.
        return EmbedResult(model=model_name)

    catalog_model = model_name
    if engine is not None:
        catalog_model = engine.catalog.resolve_model_id(provider_name, model_name)

    adapter = embedding_adapter(provider_name, key, base_url)
    result = adapter.embed(catalog_model, inputs, _fetch(engine, transport))
    _record_cost(engine, provider_name, catalog_model, result)
    return result


def _fetch(engine: Any, transport: Any) -> Any:
    """The engine's fetch, or a one-off built on the same executor.

    Identical to `transcription._fetch` on purpose: a subsystem that reached the
    network its own way would be invisible to the rate limiter that exists to
    keep the whole fleet under a provider's ceiling.
    """
    from .bus.hook_bus import HookBus
    from .network.executor import RequestExecutor
    from .network.retry import DEFAULT_RETRY
    from .transport import as_fetch, http_transport

    if engine is not None and transport is None:
        return engine.fetch

    send = as_fetch(transport or http_transport())
    executor = RequestExecutor(engine.hooks if engine is not None else HookBus())

    def fetch(req: Any, options: Any = None) -> Any:
        return executor.execute(req, lambda r: send(r), DEFAULT_RETRY)

    return fetch


def _record_cost(engine: Any, provider: str, model: str, result: EmbedResult) -> None:
    """Embeddings are billed on INPUT tokens only; there is no output to price."""
    if engine is None or result.usage is None:
        return
    recorder = getattr(engine, "cost", None)
    if recorder is None:
        return
    record = getattr(recorder, "record", None)
    if record is None:
        return
    record(
        {
            "provider": provider,
            "model": model,
            "inputTokens": result.usage.input_tokens,
            "outputTokens": 0,
        }
    )


__all__ = [
    "ADAPTERS",
    "DEFAULT_BASE_URLS",
    "EmbedResult",
    "EmbedUsage",
    "EmbeddingAdapter",
    "GoogleEmbeddingAdapter",
    "OpenAIEmbeddingAdapter",
    "OpenRouterEmbeddingAdapter",
    "embed",
    "embedding_adapter",
]
