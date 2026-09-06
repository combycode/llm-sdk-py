"""Trying models in order, and knowing when not to.

The whole value is in the second half: falling over on a rate limit is useful,
falling over on a malformed request just pays a second provider to say the same
thing. So most of these are about which failures are NOT worth another model.
"""

from __future__ import annotations

from typing import Any

import pytest

from combycode_llm_sdk.helpers.route import DEFAULT_FALLBACK_KINDS, RouteResult, route
from combycode_llm_sdk.network.errors import LLMError


class Answer:
    def __init__(self, text: str = "ok", model: str = "m") -> None:
        self.text = text
        self.model = model
        self.parsed = None


def failing(kind: str | None) -> LLMError:
    """An LLMError classified the way the executor would classify it."""
    error = LLMError("boom")
    # `kind` is what the router branches on; setting it here is the whole
    # point of the double.
    object.__setattr__(error, "kind", kind)
    return error


def patch_complete(monkeypatch: pytest.MonkeyPatch, behaviour: Any) -> list[dict[str, Any]]:
    """Replace the one-shot `complete` the router calls, recording each attempt."""
    calls: list[dict[str, Any]] = []

    def fake(**options: Any) -> Any:
        calls.append(options)
        result = behaviour(options["model"], len(calls))
        if isinstance(result, BaseException):
            raise result
        return result

    from combycode_llm_sdk.helpers import one_shot

    monkeypatch.setattr(one_shot, "complete", fake)
    return calls


class TestFallingOver:
    def test_the_first_model_that_answers_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = patch_complete(monkeypatch, lambda model, _n: Answer(model=model))
        result = route(models=["openai/a", "anthropic/b"], prompt="x")
        assert result.served_by == "openai/a"
        assert len(calls) == 1, "the second model is never asked"

    def test_a_retryable_failure_moves_to_the_next(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_complete(
            monkeypatch,
            lambda model, _n: failing("rate_limit") if model == "openai/a" else Answer(model=model),
        )
        result = route(models=["openai/a", "anthropic/b"], prompt="x")
        assert result.served_by == "anthropic/b"
        # The failure is on the record, not swallowed.
        assert result.attempts[0].kind == "rate_limit"
        assert len(result.attempts) == 2

    @pytest.mark.parametrize("kind", ["auth", "invalid_request", "content_filter"])
    def test_a_failure_another_model_cannot_fix_is_raised_at_once(
        self, monkeypatch: pytest.MonkeyPatch, kind: str
    ) -> None:
        # The point of the denylist: a bad key or a malformed request is bad
        # everywhere, and trying the next provider only costs more.
        calls = patch_complete(monkeypatch, lambda _m, _n: failing(kind))
        with pytest.raises(LLMError):
            route(models=["openai/a", "anthropic/b"], prompt="x")
        assert len(calls) == 1

    def test_an_unclassified_failure_is_raised_rather_than_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No kind means nothing is known about it, and guessing "retryable"
        # would turn one unexplained failure into several.
        calls = patch_complete(monkeypatch, lambda _m, _n: failing(None))
        with pytest.raises(LLMError):
            route(models=["openai/a", "anthropic/b"], prompt="x")
        assert len(calls) == 1

    def test_the_caller_can_widen_the_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_complete(
            monkeypatch,
            lambda model, _n: failing("auth") if model == "openai/a" else Answer(model=model),
        )
        result = route(models=["openai/a", "anthropic/b"], prompt="x", fallback_on=["auth"])
        assert result.served_by == "anthropic/b"

    def test_exhausting_every_candidate_reports_all_of_them(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_complete(monkeypatch, lambda _m, _n: failing("server_error"))
        with pytest.raises(RuntimeError, match="all 2 model\\(s\\) failed") as caught:
            route(models=["openai/a", "anthropic/b"], prompt="x")
        # Both named: knowing only the last one failed hides whether the first
        # was even tried.
        assert "openai/a" in str(caught.value)
        assert "anthropic/b" in str(caught.value)

    def test_an_empty_candidate_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            route(models=[], prompt="x")

    def test_the_default_list_excludes_what_a_swap_cannot_fix(self) -> None:
        assert "rate_limit" in DEFAULT_FALLBACK_KINDS
        assert "auth" not in DEFAULT_FALLBACK_KINDS
        assert "invalid_request" not in DEFAULT_FALLBACK_KINDS
        assert "content_filter" not in DEFAULT_FALLBACK_KINDS


class TestOpenRouterRoutesForUs:
    def test_all_openrouter_candidates_become_one_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = patch_complete(monkeypatch, lambda model, _n: Answer(model=model))
        route(models=["openrouter/a", "openrouter/b"], prompt="x")
        assert len(calls) == 1, "one round trip, not two"
        # BARE ids: OpenRouter does not want our namespace prefix.
        assert calls[0]["provider_options"]["openrouter"]["models"] == ["a", "b"]

    def test_a_mixed_list_falls_back_client_side(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = patch_complete(monkeypatch, lambda model, _n: Answer(model=model))
        route(models=["openrouter/a", "openai/b"], prompt="x")
        assert "provider_options" not in calls[0]

    def test_a_single_openrouter_model_is_an_ordinary_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Nothing to route between.
        calls = patch_complete(monkeypatch, lambda model, _n: Answer(model=model))
        route(models=["openrouter/a"], prompt="x")
        assert "provider_options" not in calls[0]

    def test_the_caller_s_own_provider_options_survive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = patch_complete(monkeypatch, lambda model, _n: Answer(model=model))
        route(
            models=["openrouter/a", "openrouter/b"],
            prompt="x",
            provider_options={"openrouter": {"transforms": ["middle-out"]}},
        )
        openrouter = calls[0]["provider_options"]["openrouter"]
        assert openrouter["transforms"] == ["middle-out"]
        assert openrouter["models"] == ["a", "b"]


class TestTheResultReadsLikeACompletion:
    def test_model_is_who_actually_served_it(self) -> None:
        # The same value as `served_by`, under the name a completion uses, so
        # swapping complete() for route() does not rewrite the caller's line.
        result = RouteResult(completion=Answer(text="hi"), served_by="anthropic/b")
        assert result.model == "anthropic/b"
        assert result.text == "hi"

    def test_it_does_not_invent_a_text_that_is_not_there(self) -> None:
        assert RouteResult(completion=object(), served_by="m").text == ""
