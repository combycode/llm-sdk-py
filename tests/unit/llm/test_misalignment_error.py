"""A safety block that explains itself is worth more than one that does not.

Transposed from `unified-library-ts/tests/unit/llm/misalignment-error.test.ts`.

A Responses call can fail INSIDE a 200: ``status: "failed"`` with a
``response.error``. There is no exception to catch, so that object is the only
thing the caller ever learns about the failure -- and two fields of it were
being kept.

**A numeric code was dropped entirely.** The field was read only when it was
already a string, so ``code: 429`` became an error with NO code -- not a wrong
one, an absent one, which reads as "the provider did not say why". OpenAI sends
both forms; openai-py 3.14 began coercing it with ``str(code)`` for the same
reason.

**``misalignment`` was never read.** Added 2026-09 beside the new
``misalignment_policy_violation`` code, it carries the explanation for the block
and, sometimes, ``steer.message`` -- a continuation the caller can actually
send. Without it an agent learns only that it was stopped.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")


from combycode_llm_sdk.llm.providers.openai.responses_registry import (
    OPENAI_RESPONSES_REGISTRY,
    openai_misalignment,
)


def error_of(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Run the registry's `oaiRespError` against a raw `response` body."""
    fn = OPENAI_RESPONSES_REGISTRY.transforms["oaiRespError"]
    ctx = type("Ctx", (), {"req": {"raw": raw, "out": {"content": [], "toolCalls": []}}})()
    return fn(None, ctx)  # type: ignore[no-any-return]


class TestTheErrorObjectInsideA200:
    def test_it_is_absent_when_no_failure_was_reported(self) -> None:
        assert error_of({}) is None
        assert error_of({"error": None}) is None
        assert error_of({"error": {}}) is None

    def test_it_keeps_a_string_code_and_message_as_before(self) -> None:
        assert error_of({"error": {"code": "data_residency_mismatch", "message": "nope"}}) == {
            "code": "data_residency_mismatch",
            "message": "nope",
        }

    def test_it_reads_a_numeric_code_as_its_decimal_string(self) -> None:
        # Previously dropped: the caller saw a failure with no code at all.
        assert error_of({"error": {"code": 429, "message": "slow down"}}) == {
            "code": "429",
            "message": "slow down",
        }

    def test_it_reads_a_numeric_code_even_when_it_is_zero(self) -> None:
        # `0` is falsy, which is how this kind of fix usually still loses one.
        assert error_of({"error": {"code": 0, "message": "x"}})["code"] == "0"  # type: ignore[index]

    def test_a_boolean_code_is_not_a_code(self) -> None:
        # `bool` is an `int` in Python, so the obvious isinstance check would
        # turn `True` into the code `"True"`. The TypeScript cannot hit this --
        # `typeof true` is not `'number'` -- so only this side needs the guard.
        assert error_of({"error": {"code": True, "message": "x"}}) == {"message": "x"}

    def test_it_ignores_a_code_that_is_neither_string_nor_number(self) -> None:
        assert error_of({"error": {"code": {"nested": True}, "message": "x"}}) == {"message": "x"}


class TestMisalignment:
    def test_it_carries_the_explanation_the_type_and_the_steer(self) -> None:
        assert error_of(
            {
                "error": {
                    "code": "misalignment_policy_violation",
                    "message": "Blocked by safety systems.",
                    "misalignment": {
                        "detailed_explanation": "The tool call would have emailed a private file.",
                        "error_type": "potentially_unintended_data_transfer",
                        "steer": {"message": "Ask the user to confirm the recipient."},
                    },
                }
            }
        ) == {
            "code": "misalignment_policy_violation",
            "message": "Blocked by safety systems.",
            "misalignment": {
                "detailedExplanation": "The tool call would have emailed a private file.",
                "errorType": "potentially_unintended_data_transfer",
                "steer": {"message": "Ask the user to confirm the recipient."},
            },
        }

    def test_an_unknown_error_type_passes_straight_through(self) -> None:
        # The provider documents four values and says clients must accept more,
        # so validating against the four would drop precisely the new ones.
        result = openai_misalignment({"error_type": "potentially_unintended_credential_use"})
        assert result == {"errorType": "potentially_unintended_credential_use"}

    def test_it_keeps_whichever_parts_arrived(self) -> None:
        assert openai_misalignment({"detailed_explanation": "why"}) == {
            "detailedExplanation": "why"
        }
        assert openai_misalignment({"steer": {"message": "try this"}}) == {
            "steer": {"message": "try this"}
        }

    def test_it_is_none_rather_than_empty_when_nothing_usable_arrived(self) -> None:
        # `misalignment: {}` would read as "a safety system explained itself"
        # when none did.
        assert openai_misalignment(None) is None
        assert openai_misalignment({}) is None
        assert openai_misalignment("blocked") is None
        assert openai_misalignment({"steer": {}}) is None
        assert openai_misalignment({"steer": None}) is None

    def test_it_does_not_appear_on_an_error_that_carries_none(self) -> None:
        assert "misalignment" not in (error_of({"error": {"code": "server_error"}}) or {})
