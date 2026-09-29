"""A model nobody can call must not be recommended.

Two different facts, deliberately kept apart:

`unavailable` is MEASURED -- somebody called the endpoint and it was gone. It
forces `active: False`, so it drops out of select() and can be refused without
spending a round trip to be told again.

A past `shutdownDate` is ANNOUNCED, and is checked at query time because a
catalog exported yesterday cannot know a date passed overnight.

What is NOT grounds for hiding a model: `deprecation.date` alone. That says a
source announced end-of-life, and the model stays callable until the shutdown --
hiding it would take away something that still works.
"""

from __future__ import annotations

from typing import Any

from combycode_llm_sdk.catalog.catalog import ModelCatalog
from combycode_llm_sdk.helpers.select_model import select_models

PAST = "2020-01-01"
FUTURE = "2999-01-01"
PRICED: dict[str, Any] = {"inputPerMTok": 0, "outputPerMTok": 0}


def _catalog(**extra: dict[str, Any]) -> ModelCatalog:
    data: dict[str, Any] = {
        "openai/fine": {"type": "chat", "active": True, "pricing": {"inputPerMTok": 1, "outputPerMTok": 1}}
    }
    data.update({k.replace("__", "/"): v for k, v in extra.items()})
    catalog = ModelCatalog()
    catalog.load(data)
    return catalog


class _Engine:
    def __init__(self, catalog: ModelCatalog) -> None:
        self.catalog = catalog
        self.api_keys = {"openai": "k", "google": "k"}


def _ids(catalog: ModelCatalog) -> list[str]:
    return [str(dict(m).get("model")) for m in select_models("type:chat", engine=_Engine(catalog))]


class TestSelectSkipsWhatCannotBeCalled:
    def test_drops_a_measured_unavailable_model(self) -> None:
        catalog = _catalog(
            openai__dead={
                "type": "chat",
                "active": False,
                "unavailable": {"since": "2026-09-29", "reason": "endpoint answers 404"},
                "pricing": PRICED,
            }
        )
        ids = _ids(catalog)
        assert "fine" in ids
        assert "dead" not in ids

    def test_drops_a_model_whose_shutdown_has_passed(self) -> None:
        catalog = _catalog(
            openai__retired={
                "type": "chat",
                "active": True,
                "deprecation": {"shutdownDate": PAST, "source": "litellm"},
                "pricing": PRICED,
            }
        )
        assert "retired" not in _ids(catalog)

    def test_keeps_one_whose_shutdown_is_ahead(self) -> None:
        catalog = _catalog(
            openai__soon={
                "type": "chat",
                "active": True,
                "deprecation": {"shutdownDate": FUTURE, "source": "litellm"},
                "pricing": PRICED,
            }
        )
        assert "soon" in _ids(catalog)

    def test_keeps_one_that_is_merely_deprecated(self) -> None:
        # It still works until its shutdown; hiding it would remove a usable model.
        catalog = _catalog(
            openai__olden={
                "type": "chat",
                "active": True,
                "deprecation": {"date": PAST, "source": "litellm"},
                "pricing": PRICED,
            }
        )
        assert "olden" in _ids(catalog)

    def test_an_explicit_active_filter_still_reaches_them(self) -> None:
        catalog = _catalog(
            openai__dead={
                "type": "chat",
                "active": False,
                "unavailable": {"since": "2026-09-29", "reason": "gone"},
                "pricing": PRICED,
            }
        )
        ids = [
            str(dict(m).get("model"))
            for m in select_models("type:chat; active:no", engine=_Engine(catalog))
        ]
        assert "dead" in ids


class TestUnavailableReason:
    def test_explains_a_measured_refusal_and_when(self) -> None:
        catalog = _catalog(
            openai__dead={
                "type": "chat",
                "active": False,
                "unavailable": {"since": "2026-09-29", "reason": "/v1/videos answers 404"},
                "pricing": PRICED,
            }
        )
        why = catalog.unavailable_reason("openai", "dead")
        assert why is not None
        assert "404" in why
        assert "2026-09-29" in why

    def test_explains_a_passed_shutdown_date(self) -> None:
        catalog = _catalog(
            openai__retired={
                "type": "chat",
                "deprecation": {"shutdownDate": PAST, "source": "litellm"},
                "pricing": PRICED,
            }
        )
        why = catalog.unavailable_reason("openai", "retired")
        assert why is not None and PAST in why

    def test_says_nothing_about_a_model_that_works(self) -> None:
        assert _catalog().unavailable_reason("openai", "fine") is None

    def test_says_nothing_about_a_merely_deprecated_model(self) -> None:
        catalog = _catalog(
            openai__olden={
                "type": "chat",
                "deprecation": {"date": PAST, "source": "litellm"},
                "pricing": PRICED,
            }
        )
        assert catalog.unavailable_reason("openai", "olden") is None

    def test_says_nothing_about_an_unknown_model(self) -> None:
        assert _catalog().unavailable_reason("openai", "who") is None


class TestTheShippedCatalogCarriesTheMeasurements:
    def test_imagen_and_sora_are_marked(self) -> None:
        catalog = ModelCatalog()
        catalog.load_provider_defaults()
        for provider, model in (("google", "imagen-4"), ("openai", "sora-2")):
            info = catalog.get(provider, model)
            if not info:
                continue
            entry = dict(info)
            assert entry.get("active") is False
            assert entry.get("unavailable") is not None
