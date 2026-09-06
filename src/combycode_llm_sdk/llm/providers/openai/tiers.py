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

#: `fast` arrived 2026-08 (openai-ts 7.x) on Responses (GA + beta),
#: chat-completions and `responses.compact`. Probe-verified on `gpt-5.5`:
#: accepted, and `"hyperfast"` rejected -- so the value is genuinely validated,
#: not tolerated. Until it was listed here `openai_request_tier('fast')` fell
#: through to `'auto'`, so a caller asking for Fast mode silently got the project
#: default. Response side needs no change: OpenAI echoes
#: `service_tier: 'priority'` for both `fast` and `priority`, which
#: `openai_billed_tier` already passes through.
_KNOWN = frozenset({"auto", "default", "flex", "scale", "priority", "fast"})


def openai_request_tier(t: str | None = None) -> str | None:
    """Map a unified tier to OpenAI's `service_tier`.

    Unknown values pass through if OpenAI accepts them, otherwise fall back to
    `auto`. None -> omit.
    """
    if not t:
        return None
    mapped = _REQUEST.get(t, t)
    return mapped if mapped in _KNOWN else "auto"


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


__all__ = ["openai_billed_tier", "openai_request_tier"]
