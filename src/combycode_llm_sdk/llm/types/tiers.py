"""Unified service tier for a request.

Transposed from `unified-library-ts/src/llm/types/tiers.ts`.
"""

from __future__ import annotations

#: `type ServiceTier = 'auto' | 'standard' | 'priority' | 'flex' | 'fast' | (string & {})`.
#:
#: The named values are the cross-provider core; the open `(string & {})` lets
#: callers pass any tier (e.g. `'scale'`, or a future internal-optimization
#: label) -- each adapter decides whether it can honor it (pass through if the
#: provider allows it, else fall back to that provider's `auto`). Open by design,
#: per CONSTITUTION.md R1: a provider adding a tier must never break a consumer,
#: so listing a value here only adds autocomplete. Hence `str`, not a `Literal`.
#:
#: `batch` is intentionally NOT a value here -- it's a separate API (the Batch
#: endpoint), not a per-request flag on a synchronous call.
#:
#: Tier mapping is provider-specific and lives ENTIRELY in the adapters.
ServiceTier = str

#: The documented set, for callers that want to offer a choice. Not a validation
#: list: anything outside it is still legal and reaches the adapter.
KNOWN_SERVICE_TIERS = ("auto", "standard", "priority", "flex", "fast")

__all__ = ["KNOWN_SERVICE_TIERS", "ServiceTier"]
