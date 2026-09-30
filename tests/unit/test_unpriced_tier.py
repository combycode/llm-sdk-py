"""A billed tier the catalog does not price is reported, not quietly assumed.

Transposed from the `unpriced_tier` half of
`unified-library-ts/tests/unit/plugins/cost-collector/unpriced.test.ts`.

An unpriced MODEL reports unknown -- visibly wrong. An unpriced TIER falls back
to the flat rate and hands back a confident number computed at the wrong one.
A latency tier is bought BECAUSE it costs more, so the error always runs the
same way: too low, on exactly the requests someone chose to pay extra for.

The live case is OpenAI's `ultrafast`: the Responses API accepts it, the catalog
prices `fast` for the same model and not `ultrafast`, and that bill read as
standard.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")


from combycode_llm_sdk.bus.hook_bus import HookBus
from combycode_llm_sdk.catalog.catalog import ModelCatalog
from combycode_llm_sdk.cost_collector import CostCollector


def catalog() -> ModelCatalog:
    cat = ModelCatalog()
    cat.set(
        "openai",
        "tiered-model",
        {
            "pricing": {
                "inputPerMTok": 1,
                "outputPerMTok": 5,
                "tiers": {"fast": {"inputPerMTok": 2, "outputPerMTok": 10}},
            }
        },
    )
    cat.set("openai", "flat-model", {"pricing": {"inputPerMTok": 1, "outputPerMTok": 5}})
    return cat


def billed_at(tier: str, model: str = "tiered-model") -> dict[str, Any]:
    return {
        "provider": "openai",
        "model": model,
        "response": {
            "usage": {
                "inputTokens": 1000,
                "outputTokens": 100,
                "pricingTier": tier,
                "serviceTier": tier,
            },
            "raw": {},
        },
        "request": {"estimatedInputTokens": 1000},
        "ctx": {},
    }


def warnings_for(*calls: dict[str, Any]) -> list[dict[str, Any]]:
    hooks = HookBus()
    seen: list[dict[str, Any]] = []
    hooks.on("onWarning", lambda c: seen.append(dict(c)))
    CostCollector(catalog=catalog(), hooks=hooks)
    for call in calls:
        hooks.emit_sync("onCompletion", call)
    return [w for w in seen if w.get("code") == "unpriced_tier"]


class TestAnUnpricedTierIsSaidOutLoud:
    def test_it_names_the_tier_and_the_ones_that_are_priced(self) -> None:
        found = warnings_for(billed_at("ultrafast"))
        assert len(found) == 1
        assert "ultrafast" in found[0]["message"]
        assert "too low" in found[0]["message"]
        assert found[0]["details"]["priced"] == ["fast"]

    def test_it_is_silent_for_a_tier_it_does_price(self) -> None:
        assert warnings_for(billed_at("fast")) == []

    def test_it_is_silent_for_standard_which_is_the_flat_rate(self) -> None:
        assert warnings_for(billed_at("standard")) == []

    def test_it_is_silent_for_a_model_with_no_tier_pricing(self) -> None:
        # Nothing to be missing from: that model is flat-priced and the flat
        # rate is the right answer. Warning here would fire on every priority
        # call to every model without tier rates.
        assert warnings_for(billed_at("priority", "flat-model")) == []

    def test_it_says_so_once_per_model_and_tier(self) -> None:
        # A missing price is a catalog fact, not a per-request event.
        assert len(warnings_for(billed_at("ultrafast"), billed_at("ultrafast"))) == 1

    def test_but_separately_for_a_second_unpriced_tier(self) -> None:
        assert len(warnings_for(billed_at("ultrafast"), billed_at("scale"))) == 2
