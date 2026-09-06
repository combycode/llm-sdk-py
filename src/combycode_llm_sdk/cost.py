"""What a call cost, or nothing when we cannot say.

Transposed from the token half of
`unified-library-ts/src/plugins/cost-collector/cost-collector-internal.ts`
(`calculateCost` / `computeCost`). The media-unit half and the CostCollector
plugin that accumulates entries land with the plugins layer; what is here is the
one calculation a `Completion` needs to answer `result.cost`.

**The unpriced case is the whole point.** The TypeScript returns a cost record
with `source: 'unknown'` and every field 0; Python's API contract is explicit
that this must surface as `None`::

    `cost` is `None` when the model is not priced, never `0.0`. A silent zero is
    exactly the bug that shipped in the TypeScript library -- a client reported
    72k tokens billed at $0.00.

So `compute_cost` returns `None` there, and a caller who sees a number knows a
rate was actually found.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .catalog.catalog import ModelCatalog

_PER_M = 1_000_000


@dataclass(frozen=True)
class Cost:
    """USD for one call, broken down. `total` is the sum of the parts."""

    total: float
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    reasoning: float = 0.0
    #: `provider` when the provider reported a total itself, `calculated` when
    #: derived from catalog rates. Never `unknown` -- that case is `None`.
    source: str = "calculated"


def provider_total(provider: str, evidence: Mapping[str, Any]) -> float | None:
    """The total the provider reported, when it reports one.

    OpenRouter and xAI bill in their own units and tell you the number; trusting
    ours over theirs would produce a figure that disagrees with the invoice.
    """
    if provider == "openrouter" and isinstance(evidence.get("cost"), (int, float)):
        return float(evidence["cost"])
    if provider == "xai" and isinstance(evidence.get("cost_usd"), (int, float)):
        return float(evidence["cost_usd"])
    return None


def compute_cost(
    catalog: ModelCatalog,
    provider: str,
    model: str,
    usage: Mapping[str, Any],
    *,
    provider_evidence: Mapping[str, Any] | None = None,
    tier: str | None = None,
) -> Cost | None:
    """The cost of one completion, or None when no rate applies.

    Order, as in the TypeScript: a provider-reported total wins outright; else
    catalog rates at the billed service tier; else None.
    """
    reported = provider_total(provider, provider_evidence or {})
    if reported is not None:
        return Cost(total=reported, source="provider")

    base = catalog.get_pricing(provider, model)
    # A model with no per-token rates is not priced by tokens -- an image model
    # priced per image would otherwise come back as a confident 0.00.
    if not base or (base.get("inputPerMTok") is None and base.get("outputPerMTok") is None):
        return None

    # Service-tier rates overlay the flat (standard) rates field by field; any
    # field a tier does not change falls back to standard.
    tier_rates = (base.get("tiers") or {}).get(tier) if tier and tier != "standard" else None
    pricing = {**base, **(tier_rates or {})}

    input_rate = pricing.get("inputPerMTok") or 0
    output_rate = pricing.get("outputPerMTok") or 0
    # The two derived defaults are the industry shape, not a guess we invented:
    # a cache read is a tenth of an input token and a cache write a quarter more
    # than one, for providers that publish no separate rate.
    cache_read_rate = _rate(pricing, "cacheReadPerMTok", input_rate * 0.1)
    cache_write_rate = _rate(pricing, "cacheWritePerMTok", input_rate * 1.25)
    # Audio tokens price at their own rate where the model has one, else at the
    # text rate.
    audio_in_rate = _rate(pricing, "audioInputPerMTok", input_rate)
    audio_out_rate = _rate(pricing, "audioOutputPerMTok", output_rate)

    input_cost = _num(usage.get("inputTokens")) / _PER_M * input_rate + _num(
        usage.get("audioInputTokens")
    ) / _PER_M * audio_in_rate
    output_cost = _num(usage.get("outputTokens")) / _PER_M * output_rate + _num(
        usage.get("audioOutputTokens")
    ) / _PER_M * audio_out_rate
    cache_read = _num(usage.get("cachedTokens")) / _PER_M * cache_read_rate
    cache_write = _num(usage.get("cacheWriteTokens")) / _PER_M * cache_write_rate
    # Reasoning tokens are billed as OUTPUT: they are generated, and every
    # provider that separates them still charges the output rate.
    reasoning = _num(usage.get("reasoningTokens")) / _PER_M * output_rate

    return Cost(
        input=input_cost,
        output=output_cost,
        cache_read=cache_read,
        cache_write=cache_write,
        reasoning=reasoning,
        total=input_cost + output_cost + cache_read + cache_write + reasoning,
        source="calculated",
    )


def _rate(pricing: Mapping[str, Any], key: str, fallback: float) -> float:
    """`pricing[key] ?? fallback` -- nullish, so a published rate of 0 (a free
    tier) stays 0 rather than falling back to a derived number."""
    value = pricing.get(key)
    return float(value) if value is not None else float(fallback)


def _num(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


__all__ = ["Cost", "compute_cost", "provider_total"]
