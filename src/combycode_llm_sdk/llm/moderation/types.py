"""Inline-moderation option + result shapes.

PARTIAL transposition of `unified-library-ts/src/llm/moderation/types.ts`, and
deliberately so: that file is 81 lines of which all but the last six are TYPE
declarations (`ModerationRequest`, `ModerationReport`, `ModerationEntry`,
`ModerationStreamOptions`, `EmulationConfig`). Those land with the moderation
area, together with the runner that consumes them.

What is here is the runtime half -- the three exported CONSTANTS --
because `native.py` imports `MODERATION_DEFAULT_MODEL` and duplicating the
literal in the importer is how two sources of truth start.
"""

from __future__ import annotations

#: Default moderation model when none is specified.
MODERATION_DEFAULT_MODEL = "omni-moderation-latest"
#: Default streaming strategy.
MODERATION_DEFAULT_STRATEGY = "buffer"
#: Default characters between streaming moderation checks.
MODERATION_DEFAULT_INTERVAL = 400

__all__ = [
    "MODERATION_DEFAULT_INTERVAL",
    "MODERATION_DEFAULT_MODEL",
    "MODERATION_DEFAULT_STRATEGY",
]
