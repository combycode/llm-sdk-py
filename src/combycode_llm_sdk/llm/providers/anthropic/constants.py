"""Anthropic provider constants.

Transposed from `unified-library-ts/src/llm/providers/anthropic/constants.ts`.

The thinking-shape and top_k band helpers that used to live here are gone: that
knowledge is `wire/pins/anthropic.messages.json`, and the token budgets are a
`$table` inside the chain specs. Both were version arithmetic in TypeScript,
which this port would have had to re-implement and keep in step -- and one copy
drifting is exactly how 2.2.1 shipped the wrong thinking shape.
"""

from __future__ import annotations

#: The Anthropic API version header sent on every request.
ANTHROPIC_API_VERSION = "2023-06-01"

__all__ = ["ANTHROPIC_API_VERSION"]
