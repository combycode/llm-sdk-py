"""Our effort vocabulary, in OpenAI's.

Transposed from
`unified-library-ts/src/llm/providers/openai/reasoning-effort.ts`.

This map exists in two places: here, for the `configuration_update` input item
the adapter builds in Python, and as the `reasoningEffort` table in
`wire/specs/openai-responses.json`, for the top-level `reasoning.effort` field
the spec interpreter builds. They MUST agree -- an effort meaning one thing on a
request and another on a stored configuration update is a bug nobody would look
for -- and `tests/unit/llm/test_configuration_update.py` asserts the two are
identical rather than trusting a later edit to touch both.

`max` is MAPPED, not sent. It means "the most this model will do", and OpenAI's
ladder tops out at `xhigh`; sending the word itself was measured as a 400 on
`gpt-5.4-nano` ("Unsupported value: 'max' is not supported"). The
`configuration_update` validator on `gpt-5.6-*` does list `max` among its
accepted values, so passing it through would work there -- but then `max` would
mean the top rung on a request and whatever OpenAI defines on an update, in the
same conversation. One meaning is worth more than one fewer line of mapping.

`none` and `minimal` pass through: they exist in OpenAI's vocabulary and not in
ours, and are nameable only because `ConfigurationEffort` admits them.
"""

from __future__ import annotations

#: Our name -> theirs. Anything not listed passes through unchanged.
OPENAI_REASONING_EFFORT: dict[str, str] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "xhigh",
    "xhigh": "xhigh",
}


def to_openai_reasoning_effort(effort: object) -> object:
    """The provider's name for `effort`, or `effort` itself when it has none."""
    if isinstance(effort, str):
        return OPENAI_REASONING_EFFORT.get(effort, effort)
    return effort
