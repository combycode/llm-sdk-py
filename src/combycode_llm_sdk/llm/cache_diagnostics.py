"""Prompt-cache diagnostics, unified across the two providers that report them.

Transposed from `unified-library-ts/src/llm/cache-diagnostics.ts`.

Both providers answer the same question -- "why did the cache not reuse the
prefix of the request I named?" -- and neither answers it in the same shape.
Every mapping below was MEASURED on 2026-09-29 (claude-haiku-4-5 on the GA
`/v1/messages` with no beta header, gpt-5.6-luna on `/v1/responses`) with a
prompt large enough to actually be cached, because a short one produces nothing
to diagnose on either side and reads as a broken feature.

====================  ==========================================  =====================================
                      Anthropic                                   OpenAI
====================  ==========================================  =====================================
hit                   ``diagnostics: null``                       ``{"type": "cache_hit"}``
miss                  ``cache_miss_reason.type`` + tokens         ``type: cache_miss`` + reason + tokens
unknown id            ``previous_message_not_found``, HTTP 200    ``comparison_response_not_found``, 200
nothing to say        ``diagnostics: null``                       ``{"type": "unavailable"}``
====================  ==========================================  =====================================

**Anthropic has no hit signal, and that is the load-bearing fact.** A request
whose prefix WAS reused returns exactly the body an undiagnosed request returns.
Its SDK calls that null "diagnosis still pending"; the measurement says it is
also what a hit looks like. So nothing here turns an Anthropic null into
``status="hit"`` -- that would publish our inference as the provider's answer.
``usage.cachedTokens`` is what says whether the cache was used; this says why it
was not, when the provider chose to say.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: `interface CacheDiagnostics` (response.ts). `{status, reason?, missedTokens?,
#: reusableTokens?, raw}`; `status` and `reason` are OPEN unions (R1).
CacheDiagnostics = dict[str, Any]


def _num(value: Any, key: str) -> dict[str, Any]:
    return {key: value} if isinstance(value, (int, float)) and not isinstance(value, bool) else {}


def anthropic_cache_diagnostics(value: Any) -> CacheDiagnostics | None:
    """`Message.diagnostics` on Anthropic messages.

    None when the provider said nothing -- which covers both "not requested" and
    "the prefix matched".
    """
    if not isinstance(value, Mapping):
        return None
    miss = value.get("cache_miss_reason")
    if not isinstance(miss, Mapping):
        return None
    kind = miss.get("type")
    if kind == "unavailable":
        return {"status": "unavailable", "raw": value}
    if kind == "previous_message_not_found":
        return {"status": "comparison_not_found", "raw": value}
    return {
        "status": "miss",
        # The block that diverged rides through under its own name. Anthropic's
        # vocabulary is NOT translated into OpenAI's: `system_changed` and
        # `input_changed` are not the same claim, and picking one for the other
        # would be a guess dressed as a unification.
        **({"reason": kind} if isinstance(kind, str) else {}),
        **_num(miss.get("cache_missed_input_tokens"), "missedTokens"),
        "raw": value,
    }


def openai_cache_diagnostics(value: Any) -> CacheDiagnostics | None:
    """`response.prompt_cache_diagnostics` on OpenAI Responses."""
    if not isinstance(value, Mapping):
        return None
    kind = value.get("type")
    if not isinstance(kind, str):
        return None
    # Unmapped values pass through unchanged rather than collapsing to a
    # neighbour: `status` is an open union (R1) and OpenAI has grown this enum
    # twice already.
    status = {
        "cache_hit": "hit",
        "cache_miss": "miss",
        "comparison_response_not_found": "comparison_not_found",
    }.get(kind, kind)
    reason = value.get("reason")
    return {
        "status": status,
        **({"reason": reason} if isinstance(reason, str) else {}),
        **_num(value.get("cache_missed_tokens"), "missedTokens"),
        **_num(value.get("comparison_reusable_tokens"), "reusableTokens"),
        "raw": value,
    }


__all__ = ["CacheDiagnostics", "anthropic_cache_diagnostics", "openai_cache_diagnostics"]
