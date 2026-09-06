"""Switching reasoning off, on the provider that reasons by default.

`thinking={"mode": "off"}` used to type-check, emit nothing at all, and leave
Google applying its own default -- so the model reasoned, the caller was told
nothing, and only the bill disagreed. Measured before the fix: 387 thought
tokens on gemini-2.5-flash and 684 on 2.5-pro for requests that asked for none.

The wire form differs by family, and both halves are measured against the live
API rather than assumed:

  2.5  -> thinkingBudget: 0        (2.5 rejects thinkingLevel outright)
  3.x  -> thinkingLevel: MINIMAL   (3.5-flash-lite, 3.6-flash and gemma-4
                                    reject a budget; MINIMAL zeroes the rest)

Some models cannot do it at all. Those are marked in the catalog and the client
drops the request with a warning, because sending the field earns a 400:
`Budget 0 is invalid. This model only works in thinking mode.`
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")

from combycode_llm_sdk.catalog.catalog import resolve_catalog
from combycode_llm_sdk.llm.providers.google.generate import GoogleAdapter

KEY = "test-key"


def _config(model: str, req: dict[str, Any]) -> dict[str, Any]:
    built = GoogleAdapter({"apiKey": KEY}).build_request({"model": model, "messages": [], **req})
    body = built.body
    assert isinstance(body, dict)
    config = body.get("generationConfig")
    assert isinstance(config, dict)
    return config


class TestTheWireForm:
    def test_two_five_disables_with_a_zero_budget(self) -> None:
        cfg = _config("gemini-2.5-flash", {"thinking": {"mode": "off"}})
        assert cfg["thinkingConfig"] == {"thinkingBudget": 0}

    def test_three_x_disables_with_the_minimal_level(self) -> None:
        cfg = _config("gemini-3.5-flash", {"thinking": {"mode": "off"}})
        assert cfg["thinkingConfig"] == {"thinkingLevel": "MINIMAL"}

    def test_off_is_not_silently_nothing(self) -> None:
        # The whole defect in one assertion: an absent thinkingConfig means
        # "use your default", and Google's default is to think.
        for model in ("gemini-2.5-flash", "gemini-3.5-flash"):
            assert "thinkingConfig" in _config(model, {"thinking": {"mode": "off"}})

    def test_off_does_not_also_send_an_effort(self) -> None:
        # A body carrying both a disable and an effort is refused, and the
        # effort block is gated separately -- so this is worth pinning.
        for model in ("gemini-2.5-flash", "gemini-3.5-flash"):
            cfg = _config(model, {"thinking": {"mode": "off", "effort": "high"}})
            assert len(cfg["thinkingConfig"]) == 1

    def test_on_still_sends_the_effort_it_always_did(self) -> None:
        assert _config("gemini-2.5-flash", {"thinking": {"mode": "auto", "effort": "low"}})[
            "thinkingConfig"
        ] == {"thinkingBudget": 2048, "includeThoughts": True}


class TestTheCatalogRecordsWhatIsPossible:
    def test_the_models_measured_as_unable_are_marked(self) -> None:
        catalog = resolve_catalog("defaults")
        for model in (
            "gemini-2.5-pro",
            "gemini-3.1-pro",
            "gemini-3.7-flash",
        ):
            entry = catalog.get("google", model)
            assert entry is not None, model
            assert dict(entry)["reasoning"]["canDisable"] is False, model

    def test_the_models_measured_as_able_are_marked_too(self) -> None:
        catalog = resolve_catalog("defaults")
        for model in ("gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.5-flash"):
            entry = catalog.get("google", model)
            assert entry is not None, model
            assert dict(entry)["reasoning"]["canDisable"] is True, model

    def test_an_unmeasured_model_says_nothing_rather_than_guessing(self) -> None:
        # Absent means nobody established it, which is NOT the same as `True`.
        # Only an explicit False makes the client act, so a provider nobody
        # probed keeps behaving exactly as it did.
        catalog = resolve_catalog("defaults")
        entry = catalog.get("anthropic", "claude-opus-4-6")
        if entry is not None:
            assert "canDisable" not in dict(entry).get("reasoning", {})
