"""OpenAI service-tier mapping -- provider-specific, kept here (shared by the
responses + completions adapters), never leaked into the SDK core.
Request param + response value share the enum: auto|default|flex|scale|priority.

Transposed from `unified-library-ts/src/llm/providers/openai/tiers.ts`.
"""

from __future__ import annotations

from typing import Any

#: unified -> OpenAI request `service_tier`.
_REQUEST: dict[str, str] = {
    "auto": "auto",
    "standard": "default",
    "priority": "priority",
    "flex": "flex",
    "scale": "scale",
    "fast": "fast",
}

_SHARED = ("auto", "default", "flex", "scale", "priority", "fast")

#: What each SURFACE accepts, read off the openai-ts clone rather than the diff.
#: 7.23 carries three vocabularies and the diff's +/- lines alone suggest changes
#: that did not happen (`scale` looks removed; it is simply absent from Live):
#:
#:   Responses, GA and beta   auto default flex scale priority fast ultrafast
#:   Chat Completions         auto default flex scale priority fast
#:   Live                     auto default flex priority fast_tier_temp_pilot ultrafast
#:
#: Live is not a surface this map serves, so `fast_tier_temp_pilot` is absent on
#: purpose: listing a value the adapter can never send would be a claim we do not
#: honour.
_ACCEPTED: dict[str, frozenset[str]] = {
    "responses": frozenset((*_SHARED, "ultrafast")),
    "chat-completions": frozenset(_SHARED),
}

_DEFAULT_SURFACE = "responses"


def openai_tier_decision(t: str | None = None, api: str | None = None) -> tuple[str | None, str | None]:
    """`(value, note)` for one surface. `note` is set ONLY on a downgrade.

    A tier the surface will not take falls back to `auto` -- but never silently.
    That fallback has now cost two releases: `fast` arrived in 2026-08 and was
    downgraded to the project default for a month, and `ultrafast` was on course
    to repeat it. Both are billing and latency decisions taken on the caller's
    behalf, and the caller could not see either happen. The note reaches them as
    an `onWarning` with code `request_adjusted`.

    `ultrafast` is access-controlled and served only by `gpt-5.6-sol`; an account
    without access gets a 400 naming `service_tier`, which is the honest answer
    and strictly better than being quietly billed at another tier.
    """
    if not t:
        return None, None
    accepted = _ACCEPTED.get(api or _DEFAULT_SURFACE, _ACCEPTED[_DEFAULT_SURFACE])
    mapped = _REQUEST.get(t, t)
    if mapped in accepted:
        return mapped, None
    return "auto", (
        f'serviceTier "{t}" is not accepted on OpenAI {api or _DEFAULT_SURFACE}; '
        f"sent \"auto\" instead. Accepted here: {', '.join(sorted(accepted))}."
    )


def openai_request_tier(t: str | None = None, api: str | None = None) -> str | None:
    """The wire value alone, for callers that do not surface notes."""
    return openai_tier_decision(t, api)[0]


def openai_billed_tier(raw: Any) -> dict[str, str]:
    """OpenAI billed `service_tier` (response) -> {raw, normalized catalog key}.

    `default` is OpenAI's word for the standard tier. As in the Google file, an
    anonymous object literal transposes to a mapping so "nothing to report" stays
    an empty result.

    The keys are camelCase because this merges into the WIRE-level unified
    `usage`, which is what the shared specs name and what the recorded corpus
    froze. The REQUEST-side `service_tier` above stays snake_case because that is
    OpenAI's own field name on the wire. The two look alike and are not the same
    thing; the port had both as snake_case, and every OpenAI cell in the
    differential disagreed with TypeScript on `usage` alone.
    """
    if not isinstance(raw, str) or not raw:
        return {}
    return {"serviceTier": raw, "pricingTier": "standard" if raw == "default" else raw}


__all__ = ["openai_billed_tier", "openai_request_tier", "openai_tier_decision"]
