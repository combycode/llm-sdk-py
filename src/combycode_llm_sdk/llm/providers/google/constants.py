"""Google provider constants.

Transposed from `unified-library-ts/src/llm/providers/google/constants.ts`.

The thinking-control tables and the 2.5-vs-3.x band test have moved into data:
the effort maps are `$table`s in the chain specs, and the band is
`wire/pins/google.generate.json`. Only the Interactions map remains here, because
its lowercase enum is read by code the specs do not own.
"""

from __future__ import annotations

#: Effort -> Interactions `thinking_level`.
#:
#: The Interactions API uses **lowercase** values (`minimal`/`low`/`medium`/
#: `high`) -- distinct from generateContent's uppercase `thinkingLevel`, and it
#: 400s on the uppercase form (live 2026-07-16). `max` maps to `high`, which is
#: the ceiling this API offers.
GOOGLE_INTERACTION_THINKING_LEVELS: dict[str, str] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "high",
}

__all__ = ["GOOGLE_INTERACTION_THINKING_LEVELS"]
