"""What models exist, from the catalog and from the provider.

Two answers to two different questions, and conflating them is the mistake this
module exists to prevent:

- **`list_models()`** is the bundled catalog -- instant, offline, and carrying
  pricing and capabilities. The right answer almost always.
- **`list_models_live()`** asks the PROVIDER what your key can actually reach
  today. Slower, costs a round trip, and it is the only way to see a model that
  shipped after this library did.

The live answer is ENRICHED by default: each live id is matched against the
bundled catalog so pricing survives, and an id we have never heard of gets a
minimal entry rather than being dropped. OpenRouter is the exception -- it is
not bundled at all, so its entries are built from the live response, prices
included.

Cached in memory for 24 hours, because a model list does not change between two
calls in the same program and a provider's `/models` endpoint is rate-limited
like everything else.

Transposed from `unified-library-ts/src/helpers/models.ts`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

#: How long a fetched list is reused. A day: long enough that a program never
#: asks twice, short enough that a new model appears the next time it runs.
TTL_SECONDS = 24 * 60 * 60

#: Where each provider publishes its list, and how to read the response. The
#: URL lives in the wire spec; this is the part a spec cannot express -- which
#: key holds the array, and which field is the callable id.
LIVE_SHAPES: Mapping[str, tuple[str, str]] = {
    # provider -> (list key, id field)
    "openai": ("data", "id"),
    "openrouter": ("data", "id"),
    "xai": ("data", "id"),
    "anthropic": ("data", "id"),
    # Google answers `models: [{name: "models/gemini-..."}]`, and the prefix is
    # not part of the callable id.
    "google": ("models", "name"),
}

_cache: dict[str, tuple[float, Mapping[str, Any]]] = {}
_lock = threading.Lock()


def clear_live_models_cache() -> None:
    """Forget every cached list. For a test, or a caller who knows better."""
    with _lock:
        _cache.clear()


def list_models(provider: str | None = None, engine: Any = None) -> list[Any]:
    """The bundled catalog: instant, offline, priced."""
    from .engine import default_engine

    handle = engine if engine is not None else default_engine()
    if handle is None:
        raise ValueError(
            "list_models: no engine. Pass engine=, or create one so the catalog "
            "has somewhere to live."
        )
    return list(handle.catalog.list(provider))


def _model_id(provider: str, row: Mapping[str, Any]) -> str:
    field = LIVE_SHAPES[provider][1]
    value = str(row.get(field) or "")
    # `models/gemini-3-flash` is not a name anything accepts as a model.
    return value[len("models/") :] if provider == "google" and value.startswith("models/") else value


def _minimal(provider: str, model_id: str, engine: Any) -> dict[str, Any]:
    """An entry for a live id the bundled catalog has never heard of.

    Minimal rather than absent: a model that shipped this morning is real, and
    dropping it would make the live list less useful than the frozen one.
    """
    preferred = None
    getter = getattr(getattr(engine, "catalog", None), "get_preferred_api", None)
    if getter is not None:
        preferred = getter(provider, model_id)
    return {
        "provider": provider,
        "model": model_id,
        "providerModelName": model_id,
        "pricing": {},
        "preferredApi": preferred or "completions",
        "supportedApis": ["completions"],
        "capabilities": {
            "toolUse": True,
            "streaming": True,
            "structuredOutput": False,
            "vision": False,
            "audio": False,
            "video": False,
            "imageGeneration": False,
            "audioGeneration": False,
            "videoGeneration": False,
        },
        "reasoning": {"supported": False, "automatic": False, "effortControl": False},
        "active": True,
    }


def _openrouter_entry(row: Mapping[str, Any]) -> dict[str, Any]:
    """OpenRouter is not bundled, so its entry is built from the live response."""
    pricing_raw = row.get("pricing") or {}
    params = row.get("supported_parameters") or []
    architecture = row.get("architecture") or {}
    modalities = (
        architecture.get("input_modalities") or [] if isinstance(architecture, Mapping) else []
    )

    def per_mtok(value: Any) -> float | None:
        # OpenRouter prices per token; the catalog speaks per million.
        try:
            number = float(value) * 1e6
        except (TypeError, ValueError):
            return None
        return round(number, 6) if number >= 0 else None

    pricing: dict[str, float] = {}
    if isinstance(pricing_raw, Mapping):
        for source, target in (("prompt", "inputPerMTok"), ("completion", "outputPerMTok")):
            value = per_mtok(pricing_raw.get(source))
            if value is not None:
                pricing[target] = value

    model_id = str(row.get("id") or "")
    return {
        "provider": "openrouter",
        "model": model_id,
        "providerModelName": model_id,
        "pricing": pricing,
        "preferredApi": "completions",
        "supportedApis": ["completions"],
        "contextWindow": row.get("context_length"),
        "capabilities": {
            "toolUse": "tools" in params,
            "streaming": True,
            "structuredOutput": "structured_outputs" in params,
            "vision": "image" in modalities,
            "audio": "audio" in modalities,
            "video": False,
            "imageGeneration": False,
            "audioGeneration": False,
            "videoGeneration": False,
        },
        "reasoning": {
            "supported": "reasoning" in params,
            "automatic": False,
            "effortControl": False,
        },
        "active": True,
    }


def _fetch_body(
    provider: str, api_key: str | None, engine: Any, refresh: bool
) -> Mapping[str, Any]:
    from ..llm.providers.anthropic.constants import ANTHROPIC_API_VERSION
    from ..llm.wire_transforms import make_registry
    from ..runtime import is_browser
    from ..wire.interpreter import build_from_spec
    from ..wire.utility_specs import utility_spec

    if not refresh:
        with _lock:
            hit = _cache.get(provider)
        if hit and time.time() - hit[0] < TTL_SECONDS:
            return hit[1]

    key = api_key or (engine.api_keys.get(provider) if engine is not None else None)
    if not key:
        raise ValueError(f'list_models_live: no API key for provider "{provider}".')

    built = build_from_spec(
        utility_spec(f"{provider}/models.list"),
        # Anthropic refuses a browser request without an explicit opt-in, and
        # the spec decides whether to send that header from this flag.
        {"browser": is_browser()},
        make_registry({}),
        provider,
        None,
        {"apiKey": key, "apiVersion": ANTHROPIC_API_VERSION},
    )
    request: dict[str, Any] = {
        "url": built.url,
        "method": built.method or "GET",
        "headers": dict(built.headers or {}),
        "provider": provider,
        # The endpoint is model-agnostic, so the queue is named explicitly
        # rather than derived from an empty model id.
        "model": "models",
        "responseType": "json",
    }
    response = _fetch(engine)(request)
    body = response.get("body") if isinstance(response, Mapping) else None
    if isinstance(body, (str, bytes)):
        import json

        try:
            body = json.loads(body)
        except ValueError:
            body = {}
    body = body if isinstance(body, Mapping) else {}
    with _lock:
        _cache[provider] = (time.time(), body)
    return body


def _fetch(engine: Any) -> Any:
    from ..bus.hook_bus import HookBus
    from ..network.executor import RequestExecutor
    from ..network.retry import DEFAULT_RETRY
    from ..transport import as_fetch, http_transport

    if engine is not None:
        return engine.fetch
    send = as_fetch(http_transport())
    executor = RequestExecutor(HookBus())

    def fetch(req: Any, options: Any = None) -> Any:
        return executor.execute(req, lambda r: send(r), DEFAULT_RETRY)

    return fetch


def list_models_live(
    *,
    provider: str,
    api_key: str | None = None,
    engine: Any = None,
    raw: bool = False,
    refresh: bool = False,
) -> list[Any]:
    """What this key can reach right now, from the provider's own endpoint.

        ids = list_models_live(provider="openai", api_key=key, raw=True)

    `raw=True` gives bare id strings; the default enriches each against the
    bundled catalog so pricing and capabilities survive.
    """
    if provider not in LIVE_SHAPES:
        raise ValueError(
            f'list_models_live: no live models endpoint for provider "{provider}". '
            f"Available: {', '.join(sorted(LIVE_SHAPES))}."
        )
    body = _fetch_body(provider, api_key, engine, refresh)
    key = LIVE_SHAPES[provider][0]
    rows_raw = body.get(key)
    rows = (
        [r for r in rows_raw if isinstance(r, Mapping)]
        if isinstance(rows_raw, Sequence) and not isinstance(rows_raw, (str, bytes))
        else []
    )
    if raw:
        return [_model_id(provider, row) for row in rows]
    if provider == "openrouter":
        return [_openrouter_entry(row) for row in rows]

    catalog = getattr(engine, "catalog", None)
    out: list[Any] = []
    for row in rows:
        model_id = _model_id(provider, row)
        known = catalog.get(provider, model_id) if catalog is not None else None
        out.append(known if known is not None else _minimal(provider, model_id, engine))
    return out


__all__ = [
    "LIVE_SHAPES",
    "TTL_SECONDS",
    "clear_live_models_cache",
    "list_models",
    "list_models_live",
]
