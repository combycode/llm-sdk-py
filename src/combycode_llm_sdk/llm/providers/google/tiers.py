"""Google service-tier mapping -- provider-specific (shared by the generate adapter),
never leaked into the SDK core. Google accepts flex|standard|priority on the request
(top-level `serviceTier`) and reports the billed tier on `usageMetadata.serviceTier`.

Transposed from `unified-library-ts/src/llm/providers/google/tiers.ts`.
"""

from __future__ import annotations

from typing import Any

#: Google's accepted request tiers (no 'auto'/'scale').
_REQUEST = frozenset({"flex", "standard", "priority"})


def google_request_tier(t: str | None = None) -> str | None:
    """unified ServiceTier -> Google request `serviceTier`.

    Unsupported values ('auto', 'scale', or anything Google doesn't take) are
    omitted so Google applies its default (standard).
    """
    if not t:
        return None
    return t if t in _REQUEST else None


def google_billed_tier(raw: Any) -> dict[str, str]:
    """Google billed tier (response `usageMetadata.serviceTier`, e.g. 'FLEX') ->
    {raw, normalized catalog key}.

    `pricing_tier` is lower-cased to key `pricing.tiers`; `service_tier` preserves
    the provider's raw value. TypeScript returns an anonymous object literal, and
    an EMPTY one when there is nothing to report -- a mapping keeps that
    distinction, where a dataclass full of `None`s would not.

    The keys are camelCase because this merges into the WIRE-level unified
    `usage`, which is what the shared specs name and what the recorded corpus
    froze. The REQUEST-side `service_tier` above stays snake_case because that is
    OpenAI's own field name on the wire. The two look alike and are not the same
    thing; the port had both as snake_case, and every OpenAI cell in the
    differential disagreed with TypeScript on `usage` alone.
    """
    if not isinstance(raw, str) or not raw:
        return {}
    return {"serviceTier": raw, "pricingTier": raw.lower()}


__all__ = ["google_billed_tier", "google_request_tier"]
