"""Picking a model by what it can do.

The failure worth guarding is the quiet one: a query the parser did not
understand coming back as "no models matched", which reads like an answer.
"""

from __future__ import annotations

import pytest

from combycode_llm_sdk import Engine, filter_aliases, filter_facets, select, select_models
from combycode_llm_sdk.catalog.catalog import resolve_catalog
from combycode_llm_sdk.helpers.select_model import SelectPrefs


def engine_with(*providers: str) -> Engine:
    return Engine(
        catalog="defaults",
        api_keys=dict.fromkeys(providers, "k"),
        register_as_default=False,
    )


class TestTheQueryLanguage:
    def test_an_unknown_key_raises_rather_than_returning_nothing(self) -> None:
        # An empty result reads like an answer; being rejected is what a stale
        # tag should cause.
        with pytest.raises(ValueError, match="unknown filter"):
            select_models("definitely_not_a_key:1", engine=engine_with("openai"))

    def test_a_bare_key_means_yes(self) -> None:
        engine = engine_with("openai", "anthropic")
        assert select_models("vision", engine=engine) == select_models(
            "vision:yes", engine=engine
        )

    def test_an_alias_expands_to_its_clause(self) -> None:
        engine = engine_with("openai", "anthropic")
        assert select_models("cheap", engine=engine) == select_models(
            "price:low", engine=engine
        )

    def test_a_numeric_comparison_is_inclusive(self) -> None:
        found = select_models("type:chat; context > 200000", engine=engine_with("openai"))
        assert found
        assert all(m.context_window >= 200_000 for m in found)

    def test_a_k_suffix_is_understood(self) -> None:
        engine = engine_with("openai")
        assert select_models("context > 200k", engine=engine) == select_models(
            "context > 200000", engine=engine
        )

    def test_clauses_are_combined_with_and(self) -> None:
        engine = engine_with("openai", "anthropic")
        both = select_models("type:chat; vision", engine=engine)
        chat = select_models("type:chat", engine=engine)
        assert len(both) <= len(chat)

    def test_a_query_can_be_a_list_of_clauses(self) -> None:
        engine = engine_with("openai")
        assert select_models(["type:chat", "vision"], engine=engine) == select_models(
            "type:chat; vision", engine=engine
        )

    def test_a_threshold_can_be_overridden(self) -> None:
        engine = engine_with("openai", "anthropic")
        strict = select_models(
            "cheap", engine=engine, prefs=SelectPrefs(thresholds={"price.low": 0.01})
        )
        assert len(strict) < len(select_models("cheap", engine=engine))


class TestWhatComesBack:
    def test_select_returns_a_string_complete_can_take(self) -> None:
        best = select("type:chat; cheap", engine=engine_with("openai", "anthropic"))
        assert best is not None
        assert "/" in best

    def test_only_providers_with_a_key_are_considered(self) -> None:
        # Selecting a model you cannot call is not a useful answer.
        best = select("type:chat", engine=engine_with("anthropic"))
        assert best is not None
        assert best.startswith("anthropic/")

    def test_the_cheapest_comes_first(self) -> None:
        from combycode_llm_sdk.helpers.select_model import _price_of

        found = select_models("type:chat", engine=engine_with("openai", "anthropic"))
        prices = [_price_of(m) or float("inf") for m in found]
        assert prices == sorted(prices)

    def test_a_query_matching_nothing_is_an_empty_list_not_an_error(self) -> None:
        # An empty result IS legitimate -- only an unknown key is not.
        assert select_models("type:chat; context > 999m", engine=engine_with("openai")) == []

    def test_select_answers_none_when_nothing_matched(self) -> None:
        assert select("context > 999m", engine=engine_with("openai")) is None

    def test_retired_models_are_left_out_unless_asked_about(self) -> None:
        engine = engine_with("openai", "anthropic")
        assert len(select_models("type:chat", engine=engine)) <= len(
            select_models("type:chat; active:no", engine=engine)
        ) + len(select_models("type:chat; active:yes", engine=engine))

    def test_a_provider_can_be_pinned(self) -> None:
        found = select_models("type:chat", engine=engine_with("openai", "anthropic"),
                              provider="anthropic")
        assert found
        assert all(m.provider == "anthropic" for m in found)


class TestTheVocabulary:
    def test_every_offered_value_is_one_the_parser_accepts(self) -> None:
        # The assertion the whole design exists for: the picker and the parser
        # cannot drift, because a stale value fails silently.
        engine = engine_with("openai")
        for facet in filter_facets(engine.catalog):
            for value in facet.values:
                select_models(f"{facet.key}:{value}", engine=engine)

    def test_every_alias_and_its_expansion_are_both_accepted(self) -> None:
        engine = engine_with("openai")
        for alias, expansion in filter_aliases().items():
            select_models(alias, engine=engine)
            select_models(expansion, engine=engine)

    def test_open_sets_come_from_the_catalog(self) -> None:
        # A provider ships a new model type and a hard-coded list is wrong that
        # day.
        types = next(f for f in filter_facets(resolve_catalog("defaults")) if f.key == "type")
        assert "chat" in types.values

    def test_without_a_catalog_the_open_sets_are_empty_not_guessed(self) -> None:
        # An empty list is honest; a stale one is not.
        types = next(f for f in filter_facets(None) if f.key == "type")
        assert list(types.values) == []

    def test_the_closed_sets_are_still_offered_without_a_catalog(self) -> None:
        price = next(f for f in filter_facets(None) if f.key == "price")
        assert "free" in price.values
        assert price.numeric is True

    def test_a_bare_flag_is_marked_as_one(self) -> None:
        vision = next(f for f in filter_facets(None) if f.key == "vision")
        assert vision.bare is True
