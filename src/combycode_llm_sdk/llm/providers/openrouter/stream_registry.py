"""OpenRouter's stream registry: OpenAI's, plus the one thing it adds.

Transposed from
`unified-library-ts/src/llm/providers/openrouter/stream-registry.ts`.

`:online` web search leaves no tool-call item in the stream; `url_citation`
annotations are the only signal it ran, so their first appearance IS the builtin
call, emitted once per stream.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ....wire.interpreter import Ctx, Registry
from ..openai.stream_registry import OPENAI_STREAM_REGISTRY


def _has_url_citation(annotations: Any) -> bool:
    return isinstance(annotations, list) and any(
        isinstance(a, Mapping) and a.get("type") == "url_citation" for a in annotations
    )


def _web_search(ctx: Ctx) -> None:
    out: dict[str, Any] = ctx.req["out"]
    if out["webSearchEmitted"]:
        return
    raw = ctx.req.get("raw")
    raw = raw if isinstance(raw, Mapping) else {}
    choices = raw.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    choice = choice if isinstance(choice, Mapping) else {}
    delta = choice.get("delta")
    message = choice.get("message")
    annotations = (delta or {}).get("annotations") if isinstance(delta, Mapping) else None
    if annotations is None and isinstance(message, Mapping):
        annotations = message.get("annotations")
    if not _has_url_citation(annotations):
        return
    out["webSearchEmitted"] = True
    out["events"].append({"type": "builtin_tool_start", "tool": "web_search"})
    out["events"].append({"type": "builtin_tool_end", "tool": "web_search"})


OPENROUTER_STREAM_REGISTRY = Registry(
    transforms=dict(OPENAI_STREAM_REGISTRY.transforms),
    builders=dict(OPENAI_STREAM_REGISTRY.builders),
    predicates=dict(OPENAI_STREAM_REGISTRY.predicates),
    effects={**OPENAI_STREAM_REGISTRY.effects, "openrouterStreamWebSearch": _web_search},
)

__all__ = ["OPENROUTER_STREAM_REGISTRY"]
