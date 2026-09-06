"""Google's token-usage readers.

Transposed from the `parseUsage` methods in
`unified-library-ts/src/llm/providers/google/generate.ts` and
`.../google/interactions.ts`, lifted out of the adapters so both the buffered
and streaming registries can call them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..anthropic.parse_helpers import empty_usage
from .tiers import google_billed_tier


def google_usage(u: Mapping[str, Any] | None) -> dict[str, Any]:
    """generateContent usage. The billed tier is merged in HERE rather than by
    the registry, which is where the TypeScript puts it too."""
    if not u:
        return empty_usage()
    inp = u.get("promptTokenCount") or 0
    out = u.get("candidatesTokenCount") or 0
    total = u.get("totalTokenCount")
    return {
        "inputTokens": inp,
        "outputTokens": out,
        "totalTokens": total if total is not None else inp + out,
        "cachedTokens": u.get("cachedContentTokenCount") or 0,
        "cacheWriteTokens": 0,
        "reasoningTokens": u.get("thoughtsTokenCount") or 0,
        # Billed service tier (output-only `usageMetadata.serviceTier`).
        **google_billed_tier(u.get("serviceTier")),
    }


def google_interactions_usage(u: Mapping[str, Any] | None) -> dict[str, Any]:
    """Interactions usage, which names every field differently again."""
    if not u:
        return empty_usage()
    inp = u.get("total_input_tokens")
    if inp is None:
        inp = u.get("prompt_tokens")
    inp = inp or 0
    out = u.get("total_output_tokens")
    if out is None:
        out = u.get("candidates_tokens")
    out = out or 0
    total = u.get("total_tokens")
    return {
        "inputTokens": inp,
        "outputTokens": out,
        "totalTokens": total if total is not None else inp + out,
        "cachedTokens": u.get("total_cached_tokens") or 0,
        "cacheWriteTokens": 0,
        "reasoningTokens": u.get("total_thought_tokens") or 0,
    }


__all__ = ["google_interactions_usage", "google_usage"]
