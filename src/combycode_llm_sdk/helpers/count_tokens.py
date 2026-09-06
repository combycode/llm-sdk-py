"""`count_tokens()` -- the number AND how it was arrived at.

Transposed from `unified-library-ts/src/helpers/count-tokens.ts`, with the one
divergence the reviewed examples ask for: the TypeScript returns a number, this
returns a `TokenCount`. A bare integer makes an estimate indistinguishable from
a measurement, and a context-window check built on one is wrong at exactly the
moment it matters.

    count = count_tokens(model="openai/gpt-4.1", input="hello", engine=engine)
    count.tokens      # 2
    count.strategy    # "tiktoken"
    count.exact       # True

It implements no counting itself: the catalog routes per model to a local
tokenizer, the provider's own endpoint, or the calibrated estimate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from ..tokens import NON_TEXT_PART_TOKENS, HybridCounter, TokenCount
from ..wire.interpreter import js_json

#: The count endpoints are free, and a free call is still a call. A zero entry
#: keeps the ledger's record of it rather than leaving silent absence.
COUNT_API_NOTE = "free: provider does not bill the count endpoint"


def count_tokens(
    *,
    model: str,
    input: Any,
    provider: str | None = None,
    api_key: str | None = None,
    exact: bool = True,
    engine: Any = None,
    transport: Any = None,
) -> TokenCount:
    """How many tokens `input` is, for this model.

    `exact=True` (the default) uses the precise path where one exists -- a local
    tokenizer for OpenAI, the count endpoint for Anthropic, Google and xAI --
    and falls back to the estimate otherwise, saying so. `exact=False` is always
    the estimate and never spends a request.
    """
    from ..catalog.catalog import resolve_catalog
    from .client_resolver import is_namespaced_model_id, parse_model_id
    from .engine import default_engine

    provider_name, model_name = (
        parse_model_id(model) if is_namespaced_model_id(model) else (provider, model)
    )
    if not provider_name:
        raise ValueError(
            "count_tokens: name the provider, either as `provider=` or as "
            '`"provider/model"`.'
        )

    engine = engine if engine is not None else default_engine()
    catalog = resolve_catalog(engine.catalog if engine is not None else None)
    key = api_key or (engine.api_keys.get(provider_name) if engine is not None else None)

    fetch = _fetch_for(engine, transport)
    counter = HybridCounter(catalog, fetch=fetch, api_key=key)
    text, allowance = _flatten(input)
    result = counter.count(text, provider_name, model_name, exact=exact)
    if allowance:
        # The non-text parts, added after: they were never characters, so they
        # cannot go through a character rate or a tokenizer.
        result = replace(result, tokens=result.tokens + allowance)

    if result.strategy == "api" and engine is not None:
        _record_free_call(engine, provider_name, model_name)
    return result


def _flatten(value: Any) -> tuple[str, int]:
    """The real TEXT to count, and a token allowance for what is not text.

    Two returns rather than one string, because a tokenizer must see the actual
    characters: padding an image out to its character-equivalent in spaces
    would make the estimate right and `tiktoken` wrong, since a run of spaces
    does not tokenize like the prose it stood in for.
    """
    if isinstance(value, str):
        return value, 0
    if isinstance(value, Mapping):
        return _from_message(value)
    if isinstance(value, Sequence):
        texts: list[str] = []
        extra = 0
        for item in value:
            text, allowance = _flatten(item)
            if text:
                texts.append(text)
            extra += allowance
        return "\n".join(texts), extra
    return str(value), 0


def _from_message(message: Mapping[str, Any]) -> tuple[str, int]:
    """One message, or one content part."""
    content = message.get("content")
    if content is None:
        # A bare content part rather than a message.
        return _from_part(message)
    if isinstance(content, str):
        return content, 0
    if isinstance(content, Sequence):
        texts: list[str] = []
        extra = 0
        for part in content:
            text, allowance = _from_part(part)
            if text:
                texts.append(text)
            extra += allowance
        return "\n".join(texts), extra
    return js_json(content), 0


def _from_part(part: Any) -> tuple[str, int]:
    if not isinstance(part, Mapping):
        return str(part), 0
    kind = part.get("type")
    if kind == "text":
        return str(part.get("text") or ""), 0
    if kind == "tool_call":
        return f"{part.get('name') or ''} {js_json(part.get('arguments') or {})}", 0
    if kind == "tool_result":
        inner = part.get("content")
        return (inner if isinstance(inner, str) else js_json(inner)), 0
    # An image, a document, an audio clip: not text, and not free either.
    return "", NON_TEXT_PART_TOKENS


def _fetch_for(engine: Any, transport: Any) -> Any:
    """The synchronous fetch a count request goes through.

    There is always one. An earlier version returned None when no engine was
    passed, which made `count_tokens(model=..., api_key=...)` silently take the
    ESTIMATE path on every provider whose strategy is the count endpoint -- it
    answered a plausible number, reported `exact=False`, and never called
    anything. Honest, and not what was asked for.
    """
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


def _record_free_call(engine: Any, provider: str, model: str) -> None:
    """Put an honest zero in the ledger for a call that really was free."""
    import time
    import uuid

    from ..cost import Cost
    from ..cost_collector import CostEntry

    entry = CostEntry(
        id=f"cost_{uuid.uuid4().hex[:12]}",
        timestamp=time.time() * 1000,
        provider=provider,
        model=model,
        tokens={"input": 0, "output": 0, "cached": 0, "cache_write": 0, "reasoning": 0},
        cost=Cost(total=0.0, source="calculated"),
        tags={"provider": provider, "model": model, "type": "count_tokens", "note": COUNT_API_NOTE},
    )
    engine.hooks.emit_sync("onCostEntry", entry)


__all__ = ["COUNT_API_NOTE", "count_tokens"]
