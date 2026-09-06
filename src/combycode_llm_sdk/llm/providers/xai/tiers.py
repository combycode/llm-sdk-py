"""xAI service-tier mapping -- provider-specific, kept out of the SDK core.

xAI's `ServiceTier` enum accepts only `SERVICE_TIER_DEFAULT` /
`SERVICE_TIER_PRIORITY` (verified against the xai-sdk proto). The
OpenAI-inherited map emits `auto` / `flex` / `scale`, which xAI rejects -- so the
xAI adapter remaps here instead of inheriting. `standard` -> xAI's `default`;
anything xAI can't honor is omitted (the server then uses its own default tier).
The billed response value (`default` / `priority`) parses correctly through the
shared `openai_billed_tier` (`default` -> `standard`), so only the request
direction needs an xAI-specific map.

Transposed from `unified-library-ts/src/llm/providers/xai/tiers.ts`.
"""

from __future__ import annotations


def xai_request_tier(t: str | None = None) -> str | None:
    """unified -> xAI request `service_tier` (`default` | `priority`), or None to omit."""
    if not t:
        return None
    if t == "priority":
        return "priority"
    if t in ("standard", "default"):
        return "default"
    # auto / flex / scale / unknown -> omit: xAI has no equivalent and 400s on them.
    return None


__all__ = ["xai_request_tier"]
