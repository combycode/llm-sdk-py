"""Classifying content before a model reads it.

`moderate()` is the standalone endpoint, not the `moderation=` option on a
completion: it classifies text that no model has seen and none has to see, which
is the only way to check a user's message before paying to send it.

`moderation_guardrail()` is that same call wired into an `Agent`. Left to each
call site, the check is something every one of them has to remember, and the one
that forgets is the one in production. As a guardrail it runs on every
completion, and a flagged input raises BEFORE the request is built -- so the
provider never sees the text and nothing is billed for it.

The endpoint is free. The call is still REPORTED: a zero-priced entry reaches
the engine's ledger, because free is not the same as invisible and a run that
made ten thousand moderation calls should be able to say so.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .llm.moderation.types import MODERATION_DEFAULT_MODEL

#: The moderations endpoint is not billed. The entry exists so the ledger can
#: say the call happened, which is a different fact from it being free.
FREE_NOTE = "free: provider does not bill the moderations endpoint"

_BASE_URL = "https://api.openai.com"


@dataclass(frozen=True)
class ModerationResult:
    """One verdict. `flagged` is the answer; the maps are why."""

    flagged: bool
    #: Category -> whether it tripped. A plain map rather than a field per
    #: category: OpenAI keeps adding them, and a fixed set would drop each new
    #: one on the floor rather than report it.
    categories: Mapping[str, bool] = field(default_factory=dict)
    category_scores: Mapping[str, float] = field(default_factory=dict)
    #: Which input kind triggered each category, on the models that report it.
    #: None -- not `{}` -- when the model does not: an empty map would say
    #: "nothing triggered", which is a claim this model never made.
    category_applied_input_types: Mapping[str, Any] | None = None
    model: str = ""

    @property
    def worst(self) -> str | None:
        """The highest-scoring category, or None when nothing scored."""
        if not self.category_scores:
            return None
        return max(self.category_scores, key=lambda k: self.category_scores[k])

    @staticmethod
    def of(raw: Mapping[str, Any], model: str = "") -> ModerationResult:
        return ModerationResult(
            flagged=bool(raw.get("flagged")),
            categories=dict(raw.get("categories") or {}),
            category_scores=dict(raw.get("categoryScores") or {}),
            category_applied_input_types=raw.get("categoryAppliedInputTypes"),
            model=model,
        )


class ModerationFlagged(RuntimeError):
    """Raised by the guardrail when content was flagged."""

    def __init__(self, result: ModerationResult, *, kind: str = "input") -> None:
        worst = result.worst or "content"
        super().__init__(f"{kind} flagged by moderation ({worst})")
        self.result = result
        self.kind = kind


def moderate(
    *,
    input: Any,
    api_key: str | None = None,
    model: str = MODERATION_DEFAULT_MODEL,
    engine: Any = None,
    transport: Any = None,
    base_url: str | None = None,
) -> ModerationResult:
    """Classify text, without any model reading it."""
    from .llm.moderation.native import _to_result
    from .llm.wire_transforms import make_registry
    from .wire.interpreter import build_from_spec
    from .wire.registry import get_wire_spec

    key = api_key or (engine.api_keys.get("openai") if engine is not None else None)
    if not key:
        raise ValueError(
            "moderate: no API key for openai. Pass api_key= or engine.api_keys."
        )

    built = build_from_spec(
        get_wire_spec("openai/moderations"),
        {"model": model, "input": _as_text(input)},
        make_registry({}),
        "openai",
        None,
        {"apiKey": key, "baseURL": base_url or _BASE_URL},
    )
    request = {
        "url": built.url,
        "method": built.method or "POST",
        "headers": dict(built.headers or {}),
        "body": built.body,
        "provider": "openai",
        # One queue for every moderation call rather than one per model: they
        # share a rate limit, so keying by model would let them starve.
        "model": "moderation",
        "responseType": "json",
    }

    response = _fetch(engine, transport)(request)
    status = response.get("status") if isinstance(response, Mapping) else 0
    body = (response.get("body") if isinstance(response, Mapping) else {}) or {}
    if not isinstance(status, int) or status >= 400:
        raise RuntimeError(f"moderation endpoint failed ({status}): {body}")

    # The standalone endpoint answers `{results: [...]}`. The completion-time
    # `moderation=` option answers `{input, output}`, which is what
    # `parse_native_moderation` reads -- a different envelope around the same
    # entry. `_to_result` is that shared entry, used here rather than parsed a
    # second way, so one verdict cannot come back in two shapes.
    results = body.get("results") if isinstance(body, Mapping) else None
    entry = results[0] if isinstance(results, Sequence) and results else None
    parsed = _to_result(entry) if isinstance(entry, Mapping) else {}
    if engine is not None:
        _record_free_call(engine, model)
    return ModerationResult.of(parsed, model=str(body.get("model") or model))


def moderation_guardrail(
    *,
    kind: str = "input",
    api_key: str | None = None,
    model: str = MODERATION_DEFAULT_MODEL,
    engine: Any = None,
    transport: Any = None,
    on_flagged: Callable[[ModerationResult], Any] | None = None,
) -> Callable[[Any], None]:
    """A guard for `Agent(before=[...])` or `after=[...]`.

    Raises on a flag rather than returning a decision, so a caller who forgot to
    read the return value still cannot proceed -- which is the failure mode a
    guard exists to remove.
    """

    def guard(ctx: Any) -> None:
        text = _text_of_context(ctx, kind)
        if not text:
            return
        result = moderate(
            input=text, api_key=api_key, model=model, engine=engine, transport=transport
        )
        if not result.flagged:
            return
        if on_flagged is not None:
            on_flagged(result)
        from .agent import GuardrailError

        raise GuardrailError(str(ModerationFlagged(result, kind=kind)), result=result)

    guard.__name__ = f"moderation_{kind}"
    return guard


def _text_of_context(ctx: Any, kind: str) -> str:
    """What this phase should classify.

    The INPUT phase reads the messages about to be sent; the OUTPUT phase reads
    what came back. Reading the wrong one is silent -- the call still happens
    and the verdict is simply about the wrong text.
    """
    if kind == "output":
        response = getattr(ctx, "response", None)
        return str(getattr(response, "text", "") or "")
    messages = getattr(ctx, "messages", None) or []
    return "\n".join(_as_text(m) for m in messages)


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        content = value.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, Sequence):
            return "".join(
                str(p.get("text") or "") for p in content if isinstance(p, Mapping)
            )
        return str(content or "")
    if isinstance(value, Sequence):
        return "\n".join(_as_text(v) for v in value)
    return str(value)


def _fetch(engine: Any, transport: Any) -> Any:
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


def _record_free_call(engine: Any, model: str) -> None:
    """An honest zero, so the ledger records a call that really was free."""
    from .cost import Cost
    from .cost_collector import CostEntry

    entry = CostEntry(
        id=f"cost_{uuid.uuid4().hex[:12]}",
        timestamp=time.time() * 1000,
        provider="openai",
        model=model,
        tokens={"input": 0, "output": 0, "cached": 0, "cache_write": 0, "reasoning": 0},
        cost=Cost(total=0.0, source="calculated"),
        tags={"provider": "openai", "model": model, "type": "moderation", "note": FREE_NOTE},
    )
    engine.hooks.emit_sync("onCostEntry", entry)


__all__ = [
    "FREE_NOTE",
    "ModerationFlagged",
    "ModerationResult",
    "moderate",
    "moderation_guardrail",
]
