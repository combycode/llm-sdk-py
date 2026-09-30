"""Which xAI models take a reasoning effort -- provider-specific, kept beside
`tiers.py` for the same reason and never leaked into the SDK core.

Transposed from `unified-library-ts/src/llm/providers/xai/reasoning.ts`.

The wire used to delete `reasoning` for every xAI model whose id did not contain
`multi-agent`, on the belief that only that model used the field. The catalog
meanwhile advertised `effortControl: True` with `effortValues` including `xhigh`
for grok-4.5 and 4.6 -- so the catalog promised a control the request never
carried, and a caller asking for `xhigh` silently got the default.

MEASURED 2026-09-30 against `/v1/responses`, reasoning-token counts on a hard
prompt (a trivial prompt cannot separate the efforts, which is how "accepted and
inert" hides -- the same trap `top_k` fell into):

===========================  =====  ==========================================
model                        code   reasoning tokens
===========================  =====  ==========================================
grok-4.6                     200    low 449 -> xhigh 3066   x6.8, disjoint
grok-4.5                     200    low  95 -> xhigh 3475   x36.6, disjoint
grok-4.3                     200    low 1307 -> xhigh 9729  x7.4, disjoint
grok-4.20                    400    "does not support parameter reasoningEffort"
grok-4.20-non-reasoning      400    same
grok-4.20-0309-reasoning     400    same
grok-4.20-multi-agent        200    low 882 -> xhigh 4611, but see below
===========================  =====  ==========================================

Two things rule out a version comparison, which is what a regex over the number
would amount to:

1. `grok-4.20` REFUSES the field outright while the numerically LOWER 4.3, 4.5
   and 4.6 honour it. 4.20 is a separate line, not a later 4.2 -- exactly the
   trap `wire/pins.py` records for version arithmetic, and a 400 is not a failure
   mode worth risking on a guess.
2. On `grok-4.20-multi-agent` the field is accepted but means something else: an
   agent COUNT rather than a thinking budget. Sending a caller's `xhigh` there
   would buy them a different thing than they asked for, so it is deliberately
   treated as not taking an effort.

So this is an explicit ordered table, first match wins, with the measurement
beside each rule. A model released after this build falls to False -- the
conservative direction: an unsent field costs a caller the control they asked
for, while a rejected one costs them the whole request.
"""

from __future__ import annotations

import re

#: Ordered rules, first match wins. The third element is why, for humans.
_RULES: tuple[tuple[re.Pattern[str], bool, str], ...] = (
    (
        # The whole 4.20 line refuses the parameter by name, under every spelling
        # its listing shows. Checked BEFORE the general grok-4 rule below, which
        # the id would otherwise match.
        re.compile(r"^grok-4\.20\b"),
        False,
        '400 "does not support parameter reasoningEffort" (2026-09-30, all three spellings)',
    ),
    (
        # 4.3 and up honour it. The catalog said `effortControl: False` for 4.3
        # and 4.7 while saying True for 4.5/4.6 -- an ordering that was wrong on
        # its face and wrong in measurement.
        re.compile(r"^grok-4\.(?:[3-9]|\d\d?\.)"),
        True,
        "measured honoured on 4.3 (x7.4), 4.5 (x36.6), 4.6 (x6.8); 4.7 accepts it",
    ),
)


def xai_takes_reasoning_effort(model: str) -> bool:
    """Does this xAI model accept `reasoning.effort` AS a thinking budget?

    The id may arrive with or without the `xai/` prefix, and a caller may pass
    either -- a capability must not depend on which.
    """
    ident = str(model).lower()
    ident = ident.removeprefix("xai/")
    for pattern, takes, _why in _RULES:
        if pattern.search(ident):
            return takes
    return False


def xai_uses_effort_as_agent_count(model: str) -> bool:
    """The multi-agent grok reads `reasoning.effort` as an agent COUNT, so it is
    the one model that wants the field for a reason of its own.

    Kept separate from `xai_takes_reasoning_effort` precisely because the two are
    different questions that happen to touch the same wire field.
    """
    return "multi-agent" in str(model).lower()


__all__ = ["xai_takes_reasoning_effort", "xai_uses_effort_as_agent_count"]
