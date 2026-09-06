"""How many tokens, and how sure we are.

Transposed from `unified-library-ts/src/plugins/context-measurer/counter/`.

Three strategies, chosen per model by the catalog, and they are not
interchangeable:

    "tiktoken"  a local BPE tokenizer -- exact, needs the optional extra
    "api"       ask the provider -- exact, costs a round trip
    "estimate"  characters/rate -- free, approximate, and honest about it

Returning a bare integer would make an estimate indistinguishable from a
measurement, which is how a context-window check ends up wrong at exactly the
moment it matters. So every count comes back saying which of the three produced
it.

`tiktoken` is an OPTIONAL extra. Without it the local strategy degrades to
`estimate` and SAYS so, rather than failing or silently lying.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from .catalog.catalog import ModelCatalog
from .wire.interpreter import js_json

#: Characters per token when the catalog names no rate. Deliberately not 4:
#: measured across the corpus the average sits nearer 3.8, and 4 under-counts
#: on exactly the dense technical text most likely to fill a window.
DEFAULT_CHARS_PER_TOKEN = 3.8

#: What one image, document or audio part costs when it cannot be measured.
#: A placeholder, and a generous one: under-counting a window is the failure
#: that matters, because it is the one discovered by a 400 at the far end.
NON_TEXT_PART_TOKENS = 250

#: Tokens added per tool call when the tokenizer counts the name and arguments
#: exactly: the provider still wraps them in an envelope the tokenizer never
#: sees. The heuristic uses a character figure for the same thing.
TOOL_CALL_ENVELOPE_TOKENS = 4


@dataclass(frozen=True)
class TokenCount:
    """A number, and what it is worth."""

    tokens: int
    #: `tiktoken` | `api` | `estimate`.
    strategy: str
    #: True only for a real measurement. An estimate must never claim to be one.
    exact: bool
    provider: str = ""
    model: str = ""

    def __int__(self) -> int:
        return self.tokens


def message_chars(message: Mapping[str, Any] | str) -> int:
    """Characters in one message, counting the parts a tokenizer cannot see."""
    if isinstance(message, str):
        return len(message)
    content = message.get("content")
    if isinstance(content, str):
        return len(content)
    if not isinstance(content, Sequence):
        return len(js_json(content))

    chars = 0
    for part in content:
        if not isinstance(part, Mapping):
            chars += len(str(part))
            continue
        kind = part.get("type")
        if kind == "text":
            chars += len(str(part.get("text") or ""))
        elif kind == "tool_call":
            # The name, the arguments, and a little for the envelope the
            # provider wraps them in.
            chars += len(str(part.get("name") or "")) + len(js_json(part.get("arguments") or {})) + 10
        elif kind == "tool_result":
            inner = part.get("content")
            chars += len(inner) if isinstance(inner, str) else len(js_json(inner))
        else:
            chars += NON_TEXT_PART_TOKENS * 4
    return chars


class HeuristicCounter:
    """Characters divided by a per-model rate. Free, and never exact."""

    strategy = "estimate"
    exact = False

    def __init__(self, catalog: ModelCatalog | None = None) -> None:
        self.catalog = catalog

    def rate(self, provider: str, model: str) -> float:
        if self.catalog is None:
            return DEFAULT_CHARS_PER_TOKEN
        info = self.catalog.get(provider, model)
        tokenizer = dict(info).get("tokenizer") if info else None
        rate = (tokenizer or {}).get("charsPerTokenDefault")
        return float(rate) if isinstance(rate, (int, float)) and rate > 0 else DEFAULT_CHARS_PER_TOKEN

    def count(self, text: str, provider: str, model: str) -> int:
        return math.ceil(len(text) / self.rate(provider, model))

    def estimate_message(
        self, message: Mapping[str, Any] | str, ctx: Mapping[str, Any] | None = None
    ) -> int:
        """A whole message, parts included, at this model's character rate."""
        ctx = ctx or {}
        rate = self.rate(str(ctx.get("provider") or ""), str(ctx.get("model") or ""))
        return math.ceil(message_chars(message) / rate)


class TiktokenCounter:
    """A local BPE tokenizer. Exact, and optional.

    The encoding comes from the catalog rather than from a provider guess: two
    models from one provider can use different encodings, and counting
    `o200k_base` text with `cl100k_base` is wrong quietly.
    """

    strategy = "tiktoken"
    exact = True

    def __init__(self, catalog: ModelCatalog | None = None) -> None:
        self.catalog = catalog

    def encoding_for(self, provider: str, model: str) -> str | None:
        if self.catalog is None:
            return None
        info = self.catalog.get(provider, model)
        tokenizer = dict(info).get("tokenizer") if info else None
        name = (tokenizer or {}).get("tiktokenEncoding")
        return name if isinstance(name, str) else None

    def available(self) -> bool:
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            return False
        return True

    def count(self, text: str, provider: str, model: str) -> int:
        import tiktoken

        name = self.encoding_for(provider, model) or "o200k_base"
        return len(tiktoken.get_encoding(name).encode(text))

    def estimate_message(
        self, message: Mapping[str, Any] | str, ctx: Mapping[str, Any] | None = None
    ) -> int:
        """A whole message, tokenising each part that HAS text.

        Per part rather than over the concatenation, because the parts are not
        concatenated on the wire either -- and because the parts that carry no
        text cannot be tokenised at all, so they are charged a flat figure
        instead of being silently counted as nothing.
        """
        ctx = ctx or {}
        provider = str(ctx.get("provider") or "")
        model = str(ctx.get("model") or "")
        if isinstance(message, str):
            return self.count(message, provider, model)

        content = message.get("content")
        if isinstance(content, str):
            return self.count(content, provider, model)
        if not isinstance(content, Sequence):
            return 0

        tokens = 0
        for part in content:
            if not isinstance(part, Mapping):
                continue
            kind = part.get("type")
            if kind == "text":
                tokens += self.count(str(part.get("text") or ""), provider, model)
            elif kind == "tool_call":
                name = str(part.get("name") or "")
                tokens += self.count(name + js_json(part.get("arguments")), provider, model)
                tokens += TOOL_CALL_ENVELOPE_TOKENS
            elif kind == "tool_result":
                body = part.get("content")
                text = body if isinstance(body, str) else js_json(body)
                tokens += self.count(text, provider, model)
            else:
                tokens += NON_TEXT_PART_TOKENS
        return tokens


class CountApiCounter:
    """Ask the provider. Exact, and costs a round trip.

    Every request goes through the injected fetch, like everything else. The
    TypeScript's own note on this is worth keeping: these two endpoints once
    called global fetch directly, so every exact count went around the engine --
    no queue, no rate limit, no retry, no telemetry -- while every other file
    said all HTTP goes through the injected one. The fetch is REQUIRED here
    rather than defaulted, because a default that silently bypasses the engine is
    the trap that produced that.
    """

    strategy = "api"
    exact = True

    #: Which spec builds the request, and where the answer is in the response.
    ENDPOINTS: ClassVar[dict[str, tuple[str, str]]] = {
        "anthropic": ("anthropic/count.messages", "input_tokens"),
        "google": ("google/count.tokens", "totalTokens"),
        "xai": ("xai/count.tokenize", "token_ids"),
    }

    def __init__(self, fetch: Any, api_key: str, catalog: ModelCatalog | None = None) -> None:
        self.fetch = fetch
        self.api_key = api_key
        self.catalog = catalog

    def supports(self, provider: str) -> bool:
        return provider in self.ENDPOINTS

    def count(self, text: str, provider: str, model: str) -> int:
        from .llm.providers.anthropic.constants import ANTHROPIC_API_VERSION
        from .llm.wire_transforms import make_registry
        from .wire.interpreter import build_from_spec
        from .wire.utility_specs import utility_spec

        spec_id, field = self.ENDPOINTS[provider]
        model_id = self.catalog.resolve_model_id(provider, model) if self.catalog else model
        payload: dict[str, Any] = (
            {"model": model_id, "messages": [{"role": "user", "content": text}]}
            if provider == "anthropic"
            else {"model": model_id, "text": text}
        )
        config: dict[str, Any] = {"apiKey": self.api_key, "baseURL": _base_url(provider)}
        if provider == "anthropic":
            config["apiVersion"] = ANTHROPIC_API_VERSION

        built = build_from_spec(
            utility_spec(spec_id), payload, make_registry({}), provider, None, config
        )
        # `build_from_spec` answers a `BuiltRequest`, not a mapping. Spreading it
        # as one produced a request with no `url`, which the transport reported
        # as a bare KeyError three layers down.
        request: dict[str, Any] = {
            "url": built.url,
            "method": built.method or "POST",
            "headers": dict(built.headers or {}),
            "body": built.body,
            # `model` is the QUEUE key, not the model being counted: every count
            # shares one queue rather than opening one per model.
            "provider": provider,
            "model": "count_tokens",
            "responseType": "json",
        }
        response = self.fetch(request)
        status = response.get("status") if isinstance(response, Mapping) else 0
        body = response.get("body") if isinstance(response, Mapping) else {}
        if not isinstance(status, int) or status >= 400:
            raise RuntimeError(f"{provider} count endpoint failed ({status}): {body}")

        value = (body or {}).get(field)
        # xAI answers with the token LIST rather than a count -- the length is
        # the count, and reading it as a number would silently give zero.
        if isinstance(value, list):
            return len(value)
        return int(value) if isinstance(value, (int, float)) else 0


def _base_url(provider: str) -> str:
    return {
        "anthropic": "https://api.anthropic.com",
        "google": "https://generativelanguage.googleapis.com",
        "xai": "https://api.x.ai",
    }[provider]


class HybridCounter:
    """Route to the strategy the catalog names for this model.

    Falls back to the estimate when -- and only when -- the chosen strategy
    cannot run: no `tiktoken` installed, no key for the count endpoint, no
    catalog entry. Every fallback is REPORTED through the returned strategy
    name, so a caller is never told an estimate is a measurement.
    """

    def __init__(
        self,
        catalog: ModelCatalog | None = None,
        *,
        fetch: Any = None,
        api_key: str | None = None,
    ) -> None:
        self.catalog = catalog
        self.heuristic = HeuristicCounter(catalog)
        self.tiktoken = TiktokenCounter(catalog)
        self.count_api = (
            CountApiCounter(fetch, api_key, catalog) if fetch is not None and api_key else None
        )

    def strategy_for(self, provider: str, model: str) -> str:
        """Which strategy this model asks for, before trying to run it."""
        if self.catalog is None:
            return "estimate"
        info = self.catalog.get(provider, model)
        tokenizer = dict(info).get("tokenizer") if info else None
        named = (tokenizer or {}).get("strategy")
        if named == "tiktoken":
            return "tiktoken"
        if named == "count_api":
            return "api"
        return "estimate"

    def estimate_message(
        self, message: Mapping[str, Any] | str, ctx: Mapping[str, Any] | None = None
    ) -> int:
        """Whichever counter this model asks for, and can actually run."""
        ctx = ctx or {}
        provider = str(ctx.get("provider") or "")
        model = str(ctx.get("model") or "")
        if self.strategy_for(provider, model) == "tiktoken" and self.tiktoken.available():
            return self.tiktoken.estimate_message(message, ctx)
        return self.heuristic.estimate_message(message, ctx)

    def count(
        self, text: str, provider: str, model: str, *, exact: bool = True
    ) -> TokenCount:
        wanted = "estimate" if not exact else self.strategy_for(provider, model)

        if wanted == "tiktoken" and self.tiktoken.available():
            return TokenCount(
                tokens=self.tiktoken.count(text, provider, model),
                strategy="tiktoken",
                exact=True,
                provider=provider,
                model=model,
            )
        if wanted == "api" and self.count_api is not None and self.count_api.supports(provider):
            return TokenCount(
                tokens=self.count_api.count(text, provider, model),
                strategy="api",
                exact=True,
                provider=provider,
                model=model,
            )
        return TokenCount(
            tokens=self.heuristic.count(text, provider, model),
            strategy="estimate",
            exact=False,
            provider=provider,
            model=model,
        )


__all__ = [
    "DEFAULT_CHARS_PER_TOKEN",
    "NON_TEXT_PART_TOKENS",
    "TOOL_CALL_ENVELOPE_TOKENS",
    "CountApiCounter",
    "HeuristicCounter",
    "HybridCounter",
    "TiktokenCounter",
    "TokenCount",
    "message_chars",
]
