"""Sampling parameters Anthropic REFUSES, which we were sending anyway.

Transposed from `unified-library-ts/tests/unit/llm/anthropic-sampling.test.ts`.

The row behind this said to keep our sampling options because "the wire still
accepts them on older models". Measured 2026-10-01, that is half true and the
other half was a shipped defect:

    claude-opus-5.5    400  `temperature` is deprecated for this model.
    claude-sonnet-4.6  200  (each one alone)

So a caller who set `temperature` -- an ordinary thing to set -- and used any
model from the `claude-opus-4.8` generation onward got a FAILED REQUEST. Our own
public option was a guaranteed 400 on the majority of a provider's line.

The boundary is exactly the wire-spec era: every model on
`anthropic/messages@4.7` refuses all three, every model on `@4.6` accepts them.
`top_k` had already been removed there for this reason; `temperature` and `top_p`
had not.

The last class below reads the FROZEN corpus (`tests/fixtures/wire-golden.json`)
rather than a hand-written expectation. That fixture ships with this package and,
until now, nothing in the Python suite read it -- so the 2.3.0 baseline that
guards the TypeScript side guarded nothing here. The TS corpus test caught this
very change and required an explicit waiver; this is the Python half of the same
question, for the one shape the change touches.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, "src")

from combycode_llm_sdk.bus.hook_bus import HookBus
from combycode_llm_sdk.catalog.catalog import ModelCatalog
from combycode_llm_sdk.llm.client import LLMClient
from combycode_llm_sdk.llm.providers.anthropic.messages import AnthropicAdapter

ADAPTER = AnthropicAdapter({"apiKey": "k"})
CATALOG = ModelCatalog.with_provider_defaults()

#: The `sampling` shape the frozen corpus pins, as the freeze script builds it.
SAMPLING: dict[str, Any] = {
    "messages": [{"role": "user", "content": "hi"}],
    "maxTokens": 256,
    "temperature": 0.3,
    "topP": 0.8,
    "topK": 5,
    "seed": 7,
    "presencePenalty": 0.1,
    "frequencyPenalty": 0.2,
    "stop": ["x"],
}


def body_for(model: str, **over: Any) -> dict[str, Any]:
    built = ADAPTER.build_request({"model": model, **SAMPLING, **over})
    return dict(built.body)


FOURSEVEN = "claude-opus-5-5"
FOURSIX = "claude-sonnet-4-6"


class TestAFourSevenEraModelRefusesAllThree:
    def test_it_sends_no_sampling_parameters(self) -> None:
        body = body_for(FOURSEVEN)
        assert "temperature" not in body
        assert "top_p" not in body
        assert "top_k" not in body

    def test_it_still_sends_everything_else(self) -> None:
        # The removal is three fields, not the shape. `stop_sequences` and
        # `max_tokens` travelling proves the era delta did not overreach.
        body = body_for(FOURSEVEN)
        assert body["max_tokens"] == 256
        assert body["stop_sequences"] == ["x"]


class TestAFourSixEraModelAcceptsThem:
    def test_it_still_sends_temperature(self) -> None:
        # The assertion that keeps this from being a blanket deletion: without it,
        # a fix that dropped sampling everywhere would look identical from the 4.7
        # side.
        assert body_for(FOURSIX, topP=None)["temperature"] == 0.3

    def test_it_sends_top_p_and_top_k(self) -> None:
        assert body_for(FOURSIX, temperature=None)["top_p"] == 0.8
        assert body_for(FOURSIX)["top_k"] == 5


class TestTheEraBoundaryMatchesTheCatalog:
    def test_every_four_seven_model_omits_them_and_no_other_does(self) -> None:
        """The boundary is a property of the spec era, so assert it across the
        whole catalogue rather than on two examples -- a model added later gets
        the right behaviour from its `wireSpec` or this fails.
        """
        checked = 0
        for info in CATALOG.list():
            if info.provider != "anthropic" or info.type != "chat":
                continue
            spec = info.wire_spec or ""
            body = body_for(info.provider_model_name or info.model)
            carries = any(k in body for k in ("temperature", "top_p", "top_k"))
            if spec == "anthropic/messages@4.7":
                assert not carries, f"{info.model} is 4.7-era and must send no sampling"
            elif spec.startswith("anthropic/messages@4.") and spec != "anthropic/messages@4.0":
                # 4.1 and 4.6 accept them; 4.0 predates top_k entirely.
                assert carries, f"{info.model} ({spec}) should still send sampling"
            checked += 1
        # A catalogue that silently emptied would make every case above vacuous.
        assert checked >= 10


class TestAgainstTheFrozenCorpus:
    """The 2.3.0 baseline, which nothing in this suite was reading."""

    CORPUS = json.loads(
        (Path(__file__).resolve().parents[2] / "fixtures" / "wire-golden.json").read_text(
            encoding="utf-8"
        )
    )

    def test_the_corpus_pins_a_sampling_shape_for_anthropic(self) -> None:
        # Guard against the guard: if the fixture stopped carrying these cells, the
        # comparison below would pass by having nothing to compare.
        rows = [k for k in self.CORPUS["index"] if k.startswith("anthropic/")]
        assert rows, "the corpus should carry anthropic models"
        assert all("sampling" in self.CORPUS["index"][k] for k in rows)

    def test_four_seven_models_now_differ_from_the_baseline_and_that_is_the_fix(self) -> None:
        """What 2.3.0 sent, and why it is right that we no longer send it.

        This is the deliberate drift the TypeScript corpus test required a waiver
        for. Asserted here from the other direction: the baseline DID carry
        `temperature`, and a 4.7-era model must no longer.
        """
        index = self.CORPUS["index"]
        bodies = self.CORPUS["bodies"]
        era_models = [
            info
            for info in CATALOG.list()
            if info.provider == "anthropic" and info.wire_spec == "anthropic/messages@4.7"
        ]
        compared = 0
        for info in era_models:
            key = next(
                (k for k in index if k.endswith(f"/{info.provider_model_name}")
                 or k == f"anthropic/{info.provider_model_name}"),
                None,
            )
            if key is None:
                continue
            digest = index[key].get("sampling")
            was = bodies.get(digest) if digest else None
            if not isinstance(was, dict):
                continue
            frozen_body = was.get("body") if isinstance(was.get("body"), dict) else was
            if not isinstance(frozen_body, dict) or "temperature" not in frozen_body:
                continue
            now = body_for(info.provider_model_name or info.model)
            assert "temperature" not in now, f"{info.model} must no longer send temperature"
            compared += 1
        # If no 4.7-era model matched a frozen row, this test proves nothing.
        assert compared >= 1, "expected at least one 4.7-era model in the frozen corpus"


def through_client(model: str, provider: str = "anthropic", **options: Any) -> tuple[dict[str, Any], list[str]]:
    """One request built by the real client, with its `request_adjusted` warnings.

    The adapter tests above cannot see the pair rule or the drop note: both live in
    the client, above the spec. Nothing is sent -- the transport is a stub.
    """
    sent: list[dict[str, Any]] = []
    warnings: list[str] = []
    hooks = HookBus()

    def on_warning(w: Any) -> None:
        if dict(w).get("code") == "request_adjusted":
            warnings.append(str(dict(w).get("message") or ""))

    hooks.on("onWarning", on_warning)

    def fetch(req: dict[str, Any], _opts: Any = None) -> dict[str, Any]:
        sent.append(dict(req.get("body") or {}))
        return {
            "status": 200,
            "headers": {},
            "body": {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        }

    client = LLMClient(
        {
            "provider": provider,
            "model": model,
            "apiKey": "k",
            "adapter": AnthropicAdapter({"apiKey": "k"}),
            "hooks": hooks,
            "fetch": fetch,
        }
    )
    client.complete("hi", {"maxTokens": 16, **options})
    return (sent[0] if sent else {}), warnings


class TestTheTemperatureTopPPairThoseModelsRefuseTogether:
    def test_it_sends_one_of_them_rather_than_failing(self) -> None:
        # `400 'temperature' and 'top_p' cannot both be specified for this model.
        # Please use only one.` -- measured on claude-sonnet-4.6 and claude-haiku-4.5.
        body, _ = through_client(FOURSIX, temperature=0.3, topP=0.8)
        assert body["temperature"] == 0.3
        assert "top_p" not in body

    def test_it_says_which_one_it_kept(self) -> None:
        _, warnings = through_client(FOURSIX, temperature=0.3, topP=0.8)
        assert "topP was dropped" in " ".join(warnings)
        assert "temperature (0.3) was sent" in " ".join(warnings)

    def test_it_does_not_also_claim_the_model_rejects_top_p(self) -> None:
        # It accepts topP perfectly well alone. Saying otherwise about a field we
        # removed ourselves teaches a reader to distrust the other warnings.
        _, warnings = through_client(FOURSIX, temperature=0.3, topP=0.8)
        assert "topP was not sent" not in " ".join(warnings)

    def test_it_leaves_a_request_with_only_one_alone(self) -> None:
        body, warnings = through_client(FOURSIX, temperature=0.3)
        assert body["temperature"] == 0.3
        assert "dropped" not in " ".join(warnings)

    def test_it_is_anthropic_only(self) -> None:
        # The rule is Anthropic's; applying it to another provider would silently
        # discard a parameter that provider honours.
        body, _ = through_client(FOURSIX, provider="openai", temperature=0.3, topP=0.8)
        assert body["temperature"] == 0.3
        assert body["top_p"] == 0.8


class TestTheDropIsReportedNotSilent:
    def test_a_four_seven_model_says_what_it_dropped(self) -> None:
        # `removeFields` deletes a field at inheritance time and says nothing, which
        # is how `top_k` behaved since 4.7 shipped: default sampling, no way to know.
        _, warnings = through_client(FOURSEVEN, temperature=0.3)
        assert "temperature was not sent" in " ".join(warnings)

    def test_it_names_every_option_in_one_note(self) -> None:
        # Three warnings about one decision read as three problems.
        _, warnings = through_client(FOURSEVEN, temperature=0.3, topP=0.8, topK=10)
        assert len(warnings) == 1
        for name in ("temperature", "topP", "topK"):
            assert name in warnings[0]

    def test_it_does_not_claim_a_trade_it_did_not_make(self) -> None:
        # The pair note would say "topP was dropped so temperature could be sent".
        # On this era temperature was not sent either, so that sentence is false.
        _, warnings = through_client(FOURSEVEN, temperature=0.3, topP=0.8)
        assert "so topP was dropped" not in " ".join(warnings)

    def test_it_says_nothing_when_no_sampling_was_asked_for(self) -> None:
        _, warnings = through_client(FOURSEVEN)
        assert not [w for w in warnings if "not sent" in w]
