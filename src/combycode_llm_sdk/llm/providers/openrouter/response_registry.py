"""OpenRouter's response registry: OpenAI's, plus the one thing it adds.

Transposed from
`unified-library-ts/src/llm/providers/openrouter/response-registry.ts`.

Composed rather than redefined, mirroring the adapter -- OpenRouter's adapter
subclasses OpenAI's and its parseResponse calls super first.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ....wire.interpreter import Ctx, Registry
from ..openai.response_registry import OPENAI_RESPONSE_REGISTRY


def _has_url_citation(annotations: Any) -> bool:
    """`:online` search leaves no tool-call item, only `url_citation` annotations."""
    return isinstance(annotations, list) and any(
        isinstance(a, Mapping) and a.get("type") == "url_citation" for a in annotations
    )


def _web_search_call(_arg: Any, ctx: Ctx) -> dict[str, Any] | None:
    """None -- so the emit is skipped -- when the search did not run."""
    raw = ctx.req.get("raw")
    raw = raw if isinstance(raw, Mapping) else {}
    choices = raw.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, Mapping) else None
    annotations = message.get("annotations") if isinstance(message, Mapping) else None
    return {"tool": "web_search"} if _has_url_citation(annotations) else None


OPENROUTER_RESPONSE_REGISTRY = Registry(
    transforms={
        **OPENAI_RESPONSE_REGISTRY.transforms,
        "openrouterWebSearchCall": _web_search_call,
    },
    builders=dict(OPENAI_RESPONSE_REGISTRY.builders),
    predicates=dict(OPENAI_RESPONSE_REGISTRY.predicates),
    effects=dict(OPENAI_RESPONSE_REGISTRY.effects),
)

__all__ = ["OPENROUTER_RESPONSE_REGISTRY"]
