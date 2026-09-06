"""Two helpers whose value is entirely in one edge case each.

`build_models_list` stamps one timestamp across the list rather than one per
row. `observation_from_completion` refuses to believe a reported input count of
zero, which is the difference between calibration learning something and
calibration learning a lie.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")

from combycode_llm_sdk.calibration import (
    CalibrationObservation,
    OutputCalibrationStore,
    input_bucket_label,
    observation_from_completion,
)
from combycode_llm_sdk.server.oai import build_models_list


class TestBuildModelsList:
    def test_each_id_becomes_an_openai_model_row(self) -> None:
        rows = build_models_list(["a", "b"])
        assert [r["id"] for r in rows] == ["a", "b"]
        assert {r["object"] for r in rows} == {"model"}
        assert {r["owned_by"] for r in rows} == {"orxa"}

    def test_every_row_shares_one_timestamp(self) -> None:
        # They were not created at measurably different times, and a differing
        # timestamp reads as meaning something.
        rows = build_models_list([str(i) for i in range(50)])
        assert len({r["created"] for r in rows}) == 1

    def test_an_empty_list_is_an_empty_list(self) -> None:
        assert build_models_list([]) == []

    def test_the_timestamp_is_whole_seconds(self) -> None:
        # OpenAI clients parse `created` as a unix second, not a float.
        (row,) = build_models_list(["a"])
        assert isinstance(row["created"], int)


def completion(**over: Any) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
        "usage": {"inputTokens": 1200, "outputTokens": 300},
        "estimatedInputTokens": 1100,
    }
    ctx.update(over)
    return ctx


class TestObservationFromCompletion:
    def test_it_carries_both_counts_straight_across(self) -> None:
        assert observation_from_completion(completion()) == CalibrationObservation(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=1200,
            output_tokens=300,
        )

    def test_a_reported_zero_falls_back_to_the_estimate(self) -> None:
        # Several providers omit input tokens. A 0 is not merely missing: it
        # files the call under the smallest bucket and drags that bucket's
        # learned output length toward a value no real call produced.
        ctx = completion(usage={"inputTokens": 0, "outputTokens": 300}, estimatedInputTokens=777)
        assert observation_from_completion(ctx).input_tokens == 777

    def test_the_fallback_changes_which_bucket_it_lands_in(self) -> None:
        # The reason the fallback matters, stated as the consequence.
        zero = observation_from_completion(
            completion(usage={"inputTokens": 0, "outputTokens": 5}, estimatedInputTokens=9000)
        )
        assert input_bucket_label(zero.input_tokens) != input_bucket_label(0)

    def test_snake_case_payloads_are_read_too(self) -> None:
        # The bus delivers both spellings; a helper that knew only one would
        # silently return zeros for half its callers.
        ctx = {
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        got = observation_from_completion(ctx)
        assert (got.input_tokens, got.output_tokens) == (10, 20)

    def test_a_payload_missing_everything_is_zeros_not_an_error(self) -> None:
        # It is fed from a hook, and raising there would take down the run over
        # a statistic nobody asked for.
        got = observation_from_completion({})
        assert (got.provider, got.input_tokens, got.output_tokens) == ("", 0, 0)

    def test_what_it_produces_is_what_record_accepts(self) -> None:
        # The point of the helper: the four fields line up with `record`.
        store = OutputCalibrationStore()
        obs = observation_from_completion(completion())
        store.record(obs.provider, obs.model, obs.input_tokens, obs.output_tokens)
        entry = store.get(obs.provider, obs.model, obs.input_tokens)
        assert entry is not None
        assert entry.ewma_mean == 300.0
