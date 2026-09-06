"""Classifying before the model reads, and checking which model answered.

Both answer with EVIDENCE rather than a boolean, and both are about a failure
that leaves no other trace: text the provider should never have seen, and a
substitution recorded only in a field nothing reads.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from combycode_llm_sdk import (
    Agent,
    Engine,
    LLMError,
    ModerationResult,
    OpenAIProvenanceAdapter,
    TransportResponse,
    moderate,
    moderation_guardrail,
)
from combycode_llm_sdk.agent import GuardrailError
from combycode_llm_sdk.provenance import AnthropicProvenanceAdapter, normalise_model

CATEGORIES = {"harassment": False, "violence": True}
SCORES = {"harassment": 0.01, "violence": 0.98}

COMPLETION = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude",
    "content": [{"type": "text", "text": "done"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 3},
}


def moderations(flagged: bool = False, applied: Any = None) -> Any:
    def stub(request: Any) -> TransportResponse:
        result: dict[str, Any] = {
            "flagged": flagged,
            "categories": CATEGORIES,
            "category_scores": SCORES,
        }
        if applied is not None:
            result["category_applied_input_types"] = applied
        return TransportResponse(
            body={"id": "modr-1", "model": "omni-moderation-2024-09-26", "results": [result]}
        )

    return stub


class Provider:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: Any) -> TransportResponse:
        self.calls += 1
        return TransportResponse(body=COMPLETION)


class TestModerate:
    def test_one_input_in_one_verdict_out(self) -> None:
        verdict = moderate(input="I will find you.", api_key="k", transport=moderations(True))
        assert isinstance(verdict, ModerationResult)
        assert verdict.flagged is True

    def test_the_scores_say_what_was_flagged(self) -> None:
        # A plain map rather than a field per category: OpenAI keeps adding
        # them, and a fixed set would drop each new one on the floor.
        verdict = moderate(input="x", api_key="k", transport=moderations(True))
        assert verdict.category_scores["violence"] == 0.98
        assert verdict.worst == "violence"

    def test_a_model_that_does_not_report_applied_types_says_none(self) -> None:
        # Not `{}`, which would say "nothing triggered" -- a claim this model
        # never made.
        verdict = moderate(input="x", api_key="k", transport=moderations(True))
        assert verdict.category_applied_input_types is None

    def test_a_model_that_does_report_them_keeps_them(self) -> None:
        verdict = moderate(
            input="x", api_key="k", transport=moderations(True, {"violence": ["text"]})
        )
        assert verdict.category_applied_input_types == {"violence": ["text"]}

    def test_a_free_call_is_still_on_the_books(self) -> None:
        # Free is not the same as invisible: a run that made ten thousand
        # moderation calls should be able to say so.
        engine = Engine(register_as_default=False)
        entries: list[Any] = []
        engine.on("on_cost_entry", entries.append)
        moderate(input="hello", api_key="k", engine=engine, transport=moderations())
        assert len(entries) == 1
        assert entries[0].cost.input == 0
        assert entries[0].cost.output == 0

    def test_no_key_is_refused_before_any_request(self) -> None:
        with pytest.raises(ValueError, match="no API key"):
            moderate(input="x", transport=moderations())

    def test_a_failing_endpoint_raises_rather_than_passing_content(self) -> None:
        # Answering "not flagged" on an error would let the text straight
        # through, which is the one direction this must never fail in.
        def failing(request: Any) -> TransportResponse:
            return TransportResponse(status=500, body={"error": "down"})

        # A typed `LLMError` from the retry layer, which classifies the status
        # before this code sees it -- the point being that it RAISES rather
        # than answering "not flagged" and letting the text through.
        with pytest.raises(LLMError):
            moderate(input="x", api_key="k", transport=failing)


class TestTheGuardrail:
    def test_a_flagged_input_stops_the_run_before_the_completion(self) -> None:
        provider = Provider()
        agent = Agent(
            model="anthropic/claude-haiku-4.5",
            api_key="k",
            transport=provider,
            before=[moderation_guardrail(api_key="k", transport=moderations(True))],
        )
        with pytest.raises(GuardrailError):
            agent.complete("I will find you.")
        # Stopped before it was paid for.
        assert provider.calls == 0

    def test_a_clean_run_passes_through_both_phases(self) -> None:
        agent = Agent(
            model="anthropic/claude-haiku-4.5",
            api_key="k",
            transport=Provider(),
            before=[moderation_guardrail(api_key="k", transport=moderations(False))],
            after=[
                moderation_guardrail(
                    kind="output", api_key="k", transport=moderations(False)
                )
            ],
        )
        assert agent.complete("hello").text == "done"

    def test_the_refusal_carries_the_verdict(self) -> None:
        guard = moderation_guardrail(api_key="k", transport=moderations(True))

        class Ctx:
            messages: ClassVar[list[Any]] = [{"role": "user", "content": "bad"}]

        try:
            guard(Ctx())
        except GuardrailError as exc:
            assert isinstance(exc.result, ModerationResult)
            assert exc.result.flagged
        else:
            pytest.fail("the guard did not refuse")

    def test_nothing_to_classify_is_not_a_call(self) -> None:
        called: list[Any] = []

        def counting(request: Any) -> TransportResponse:
            called.append(request)
            return TransportResponse(body={"results": []})

        class Empty:
            messages: ClassVar[list[Any]] = []

        moderation_guardrail(api_key="k", transport=counting)(Empty())
        assert called == []


class TestProvenance:
    ADAPTER = OpenAIProvenanceAdapter()

    def test_a_dated_snapshot_is_a_match(self) -> None:
        # A provider pinning a version is normal; reading it as a substitution
        # would make the check cry wolf on every well-behaved request.
        report = self.ADAPTER.check(
            requested_model="gpt-5.4-nano",
            response={"id": "r", "model": "gpt-5.4-nano-2026-01-15"},
        )
        assert report.verdict == "match"
        assert report.matched is True

    def test_a_different_family_is_a_mismatch(self) -> None:
        report = self.ADAPTER.check(
            requested_model="gpt-5.4-nano", response={"id": "r", "model": "gpt-3.5-turbo"}
        )
        assert report.verdict == "mismatch"

    def test_a_response_naming_no_model_is_unknown_not_a_mismatch(self) -> None:
        # A provider that reports less has not substituted anything, and saying
        # it did would train the reader to ignore the check.
        report = self.ADAPTER.check(requested_model="gpt-5.4-nano", response={"id": "r"})
        assert report.verdict == "unknown"

    def test_the_report_says_which_checks_ran(self) -> None:
        # A bare True/False hides the reason, and the reason is the point.
        report = self.ADAPTER.check(
            requested_model="gpt-5.4-nano", response={"id": "r", "model": "gpt-5.4-nano"}
        )
        assert len(report.checks) > 0
        assert {c.name for c in report.checks} >= {"model_family", "response_names_a_model"}

    def test_the_serving_stack_is_evidence_not_a_verdict(self) -> None:
        # A fingerprint changing means the backend changed, which is not the
        # same as being served another model.
        report = self.ADAPTER.check(
            requested_model="gpt-5.4-nano",
            response={"id": "r", "model": "gpt-5.4-nano", "system_fingerprint": "fp_abc"},
        )
        assert report.evidence["system_fingerprint"] == "fp_abc"
        assert report.verdict == "match"

    def test_normalising_strips_only_a_snapshot_suffix(self) -> None:
        assert normalise_model("gpt-5.4-nano-2026-01-15") == "gpt-5.4-nano"
        assert normalise_model("gpt-5.4-nano") == "gpt-5.4-nano"
        # `-turbo` is part of the name, not a date.
        assert normalise_model("gpt-3.5-turbo") == "gpt-3.5-turbo"

    def test_the_anthropic_adapter_reads_its_own_evidence(self) -> None:
        report = AnthropicProvenanceAdapter().check(
            requested_model="claude-haiku-4.5",
            response={"id": "msg_1", "model": "claude-haiku-4.5", "stop_reason": "end_turn"},
        )
        assert report.verdict == "match"
        assert report.evidence["stop_reason"] == "end_turn"
