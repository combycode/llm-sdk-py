"""Pricing before the call, and one step list run two ways.

The failures here are both about a number or a shape that looks right: an
estimate believed as a measurement, a budget that warns instead of refusing, and
a fan-out that silently prompts each branch with another branch's answer.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from combycode_llm_sdk import (
    Budget,
    BudgetExceededError,
    ChainStepInfo,
    Estimator,
    OutputCalibrationStore,
    Step,
    TransportResponse,
    UnknownModelError,
    aparallel,
    chain,
    estimate_cost,
    parallel,
)
from combycode_llm_sdk.calibration import (
    P90_HISTOGRAM_BIN_WIDTH,
    calibration_key,
    input_bucket_label,
)

MODEL = "anthropic/claude-haiku-4.5"
PROMPT = "Summarise the attached incident report and list every action taken."


class TestTheEstimate:
    def test_the_three_bounds_stay_ordered(self) -> None:
        e = estimate_cost(model=MODEL, prompt=PROMPT)
        assert e.low <= e.expected <= e.high

    def test_low_is_the_input_and_nothing_back(self) -> None:
        # The floor: spent the moment the request is accepted, whatever the
        # model decides to say.
        e = estimate_cost(model=MODEL, prompt=PROMPT)
        assert e.low == e.breakdown.input_usd

    def test_it_admits_the_count_is_an_estimate(self) -> None:
        # An estimate that cannot say what it assumed gets believed like a
        # measurement.
        assumed = " ".join(estimate_cost(model=MODEL, prompt=PROMPT).assumptions)
        assert "estimated from characters" in assumed

    def test_the_assumed_reply_length_is_written_down(self) -> None:
        e = estimate_cost(model=MODEL, prompt=PROMPT)
        assert str(e.est_output_tokens) in " ".join(e.assumptions)

    def test_a_caller_supplied_length_is_used_and_declared(self) -> None:
        e = estimate_cost(model=MODEL, prompt=PROMPT, output_tokens=64)
        assert e.est_output_tokens == 64
        assert "caller supplied" in " ".join(e.assumptions)

    def test_an_unpriced_model_is_refused_rather_than_priced_at_zero(self) -> None:
        # An estimate of $0.00 is indistinguishable from a genuinely free call:
        # a budget built on it passes every check, and the first sign that the
        # pricing was missing is the invoice.
        with pytest.raises(UnknownModelError, match="not in the catalog"):
            estimate_cost(model="anthropic/nothing-like-this", prompt=PROMPT)

    def test_the_refusal_names_the_model_it_could_not_price(self) -> None:
        with pytest.raises(UnknownModelError) as caught:
            estimate_cost(model="anthropic/nothing-like-this", prompt=PROMPT)
        assert caught.value.provider == "anthropic"
        assert caught.value.model == "nothing-like-this"

    def test_the_calibrated_estimator_refuses_it_too(self) -> None:
        # It prices through `estimate_cost`, so it cannot quietly answer a
        # question the plain call refuses.
        with pytest.raises(UnknownModelError):
            Estimator(OutputCalibrationStore()).estimate(
                model="anthropic/nothing-like-this", prompt=PROMPT
            )

    def test_a_bare_model_without_a_provider_is_refused(self) -> None:
        with pytest.raises(ValueError, match="name the provider"):
            estimate_cost(model="claude-haiku-4.5", prompt=PROMPT)


class TestCalibration:
    def test_recorded_replies_replace_the_guess(self) -> None:
        static = estimate_cost(model=MODEL, prompt=PROMPT)
        estimator = Estimator(OutputCalibrationStore())
        for _ in range(20):
            estimator.record("anthropic", "claude-haiku-4.5", static.input_tokens, 40)

        calibrated = estimator.estimate(model=MODEL, prompt=PROMPT)
        assert calibrated.est_output_tokens == 40
        assert calibrated.expected < static.expected
        # The input is measured the same way either way.
        assert calibrated.low == static.low

    def test_a_learned_number_says_how_many_samples(self) -> None:
        estimator = Estimator(OutputCalibrationStore())
        for _ in range(3):
            estimator.record("anthropic", "claude-haiku-4.5", 20, 40)
        assumed = " ".join(estimator.estimate(model=MODEL, prompt=PROMPT).assumptions)
        assert "calibrated: expected from 3 samples" in assumed

    def test_nothing_observed_falls_back_to_the_static_guess(self) -> None:
        # Which already says it is a guess, so the honest answer is available
        # from the first call rather than after the first twenty.
        fresh = Estimator(OutputCalibrationStore()).estimate(model=MODEL, prompt=PROMPT)
        assert "a static default" in " ".join(fresh.assumptions)

    def test_learning_is_keyed_by_input_size_bucket(self) -> None:
        # A 2,001-token prompt and a 2,002-token one are the same question;
        # keying on the exact number means every prompt is a fresh key.
        assert input_bucket_label(2_001) == input_bucket_label(2_002)
        assert input_bucket_label(10) != input_bucket_label(50_000)
        assert calibration_key("p", "m", "0-500").endswith("p/m#0-500")

    def test_a_different_size_does_not_read_anothers_learning(self) -> None:
        store = OutputCalibrationStore()
        store.record("p", "m", 10, 40)
        assert store.get("p", "m", 10) is not None
        assert store.get("p", "m", 50_000) is None

    def test_the_ewma_starts_at_the_first_observation(self) -> None:
        # Starting from zero spends the first dozen samples climbing out of a
        # number nobody observed.
        store = OutputCalibrationStore()
        store.record("p", "m", 10, 400)
        entry = store.get("p", "m", 10)
        assert entry is not None
        assert entry.ewma_mean == 400

    def test_the_high_bound_comes_from_the_histogram(self) -> None:
        store = OutputCalibrationStore()
        for _ in range(10):
            store.record("p", "m", 10, 100)
        entry = store.get("p", "m", 10)
        assert entry is not None
        assert entry.quantile() == P90_HISTOGRAM_BIN_WIDTH

    def test_a_key_with_no_observations_has_no_quantile(self) -> None:
        from combycode_llm_sdk.calibration import OutputCalibrationEntry

        assert OutputCalibrationEntry(key="k").quantile() == 0


class TestTheBudget:
    def estimate(self, output_tokens: int) -> Any:
        return estimate_cost(model=MODEL, prompt=PROMPT, output_tokens=output_tokens)

    def test_a_call_that_fits_returns_what_it_will_cost(self) -> None:
        cheap = self.estimate(10)
        assert Budget(1.0).check(cheap) == cheap.expected

    def test_a_limit_that_lets_the_call_through_has_not_limited_anything(self) -> None:
        expensive = self.estimate(100_000)
        with pytest.raises(BudgetExceededError):
            Budget(0.0001).check(expensive)

    def test_a_refused_call_is_not_charged_for(self) -> None:
        budget = Budget(0.0001)
        with pytest.raises(BudgetExceededError):
            budget.check(self.estimate(100_000))
        assert budget.spent == 0.0

    def test_the_refusal_carries_what_it_judged(self) -> None:
        # "Over budget" without these is an error nobody can act on.
        expensive = self.estimate(100_000)
        budget = Budget(0.0001)
        try:
            budget.check(expensive)
        except BudgetExceededError as exc:
            assert exc.estimate is expensive
            assert exc.cost_usd == expensive.expected
            assert exc.limit_usd == 0.0001
            assert exc.bound == "expected"
            assert exc.spent_usd == 0.0
        else:
            pytest.fail("the budget did not refuse")

    def test_only_the_call_that_would_cross_is_refused(self) -> None:
        # A budget that refused the first affordable call because the total
        # might one day be reached is not usable.
        one = self.estimate(10)
        budget = Budget(one.expected * 2.5)
        made = 0
        while True:
            try:
                budget.record(budget.check(one))
            except BudgetExceededError:
                break
            made += 1
        assert made == 2
        assert budget.spent <= budget.limit_usd

    def test_remaining_is_the_limit_minus_the_ledger(self) -> None:
        budget = Budget(1.0)
        budget.record(0.25)
        assert budget.remaining == 0.75

    def test_the_bound_can_be_the_high_one(self) -> None:
        # For a caller who cannot afford a surprise.
        one = self.estimate(10)
        assert Budget(1.0, bound="high").cost_of(one) == one.high


class TestSteps:
    MODEL = "anthropic/claude-haiku-4.5"

    def build(self) -> tuple[list[Step], list[str]]:
        sent: list[str] = []

        def stub(request: Any) -> TransportResponse:
            prompt = " ".join(p["text"] for p in request.body["messages"][-1]["content"])
            sent.append(prompt)
            verb, _, subject = prompt.partition(": ")
            return TransportResponse(
                body={
                    "content": [{"type": "text", "text": f"{verb.lower()}({subject})"}],
                    "usage": {"input_tokens": 3, "output_tokens": 5},
                    "stop_reason": "end_turn",
                }
            )

        options = {"api_key": "k", "transport": stub}
        return [
            Step(model=self.MODEL, prompt=lambda t: f"Summarise: {t}", name="summarise",
                 options=options),
            Step(model=self.MODEL, prompt=lambda t: f"Translate: {t}", name="translate",
                 options=options),
        ], sent

    def test_a_chain_prompts_each_step_with_the_last_ones_answer(self) -> None:
        # Threading the input through twice would look identical from the
        # result alone, so this asserts on what went on the wire.
        steps, sent = self.build()
        assert chain(steps)("the article") == "translate(summarise(the article))"
        assert sent == ["Summarise: the article", "Translate: summarise(the article)"]

    def test_a_fan_out_gives_every_branch_the_input(self) -> None:
        steps, sent = self.build()
        assert parallel(steps)("the article") == [
            "summarise(the article)",
            "translate(the article)",
        ]
        assert sorted(sent) == ["Summarise: the article", "Translate: the article"]

    def test_the_result_list_keeps_step_order(self) -> None:
        # Whatever finished first: a caller reading position 1 gets step 1.
        steps, _sent = self.build()
        assert parallel(steps)("x")[1].startswith("translate")

    def test_on_step_reports_the_index_and_the_output(self) -> None:
        steps, _sent = self.build()
        seen: list[ChainStepInfo] = []
        chain(steps, on_step=seen.append)("the article")
        assert [(i.index, i.name, i.output) for i in seen] == [
            (0, "summarise", "summarise(the article)"),
            (1, "translate", "translate(summarise(the article))"),
        ]

    def test_the_same_callback_works_for_a_fan_out(self) -> None:
        steps, _sent = self.build()
        seen: list[ChainStepInfo] = []
        parallel(steps, on_step=seen.append)("the article")
        assert sorted((i.index, i.name) for i in seen) == [(0, "summarise"), (1, "translate")]

    def test_the_async_fan_out_agrees_with_the_sync_one(self) -> None:
        # The only difference a caller should observe is that one costs threads.
        sync_steps, _a = self.build()
        async_steps, _b = self.build()
        assert asyncio.run(aparallel(async_steps)("x")) == parallel(sync_steps)("x")

    def test_an_empty_pipeline_is_not_an_error(self) -> None:
        assert parallel([])("x") == []
        assert asyncio.run(aparallel([])("x")) == []

    def test_a_step_without_a_name_still_reports_one(self) -> None:
        seen: list[ChainStepInfo] = []
        steps, _sent = self.build()
        unnamed = Step(model=self.MODEL, prompt=steps[0].prompt, options=steps[0].options)
        chain([unnamed], on_step=seen.append)("x")
        assert seen[0].name == "step-0"
