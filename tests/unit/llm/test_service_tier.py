"""Translated from `unified-library-ts/tests/unit/llm/service-tier.test.ts`.

Service-tier support across the request -> bill -> cost loop.

The REQUEST half is driven through the etalon wire specs, the same way
`test_strict_schema.py` does and for the same reason: TypeScript drives it
through `OpenAIResponsesAdapter.buildRequest()` and the adapters are not in this
port batch, while the spec that TypeScript's own `spec-differential.test.ts`
proves equivalent to each adapter is vendored here byte-for-byte. The
`service_tier` field of every one of those specs routes through the ported
`openaiTier` / `googleTier` transform (Anthropic's routes through the spec's own
`requestTier` table), so the assertion lands on the same code the adapter runs.

The RESPONSE half (`parseResponse`), `parseModelTier` and the cost collector
belong to files this batch does not port. Their describes are carried below as
skips with their assertions intact. Where the skipped describe's expected values
come from a function this batch DOES port -- `openaiBilledTier`,
`googleBilledTier` -- the same assertion is additionally applied to that
function, so the ported code is not left unpinned; that is stated at each one.
"""

from __future__ import annotations

from typing import Any

import pytest

from combycode_llm_sdk.llm.providers.google.tiers import google_billed_tier
from combycode_llm_sdk.llm.providers.openai.tiers import openai_billed_tier
from combycode_llm_sdk.llm.wire_transforms import make_registry
from combycode_llm_sdk.wire.inherit import resolve_spec
from combycode_llm_sdk.wire.interpreter import build_from_spec
from combycode_llm_sdk.wire.registry import WIRE_SPECS

OPENAI_RESPONSES = "openai/responses"
OPENAI_COMPLETIONS = "openai/chat-completions"
ANTHROPIC = "anthropic/messages@4.7"
GOOGLE = "google/generate@3"


def req(**over: Any) -> dict[str, Any]:
    """service-tier.test.ts:15-19."""
    return {"model": "m", "messages": [{"role": "user", "content": "hi"}], **over}


class _MessageBuilderStub:
    """Stands in for the adapter message builders; see test_strict_schema.py.

    No assertion in this file reads the messages it produces.
    """

    def build_input_items(self, msg: Any, tool_names: Any, notes: Any = None) -> list[Any]:
        return [msg]

    def build_messages(self, msg: Any, notes: Any = None) -> list[Any]:
        return [msg]

    def build_message(
        self, msg: Any, request: Any, cache_last: bool, notes: Any = None
    ) -> Any:
        return msg

    def build_content(self, msg: Any, notes: Any = None) -> Any:
        return msg


_HANDLES = {
    "anthropic": _MessageBuilderStub(),
    "google": _MessageBuilderStub(),
    "openai_responses": _MessageBuilderStub(),
    "openai_completions": _MessageBuilderStub(),
}


def body(spec_id: str, request: dict[str, Any]) -> dict[str, Any]:
    return build_from_spec(
        resolve_spec(spec_id, WIRE_SPECS), request, make_registry(_HANDLES)
    ).body


# --- request mapping (per-provider, owned by the adapter) ---


class TestServiceTierToProviderRequestParam:
    """service-tier.test.ts:22."""

    def test_omits_service_tier_when_no_tier_requested(self) -> None:
        # service-tier.test.ts:29-31
        assert "service_tier" not in body(OPENAI_RESPONSES, req())
        assert "service_tier" not in body(ANTHROPIC, req())
        assert "serviceTier" not in body(GOOGLE, req())

    def test_openai_maps_standard_priority_flex_scale(self) -> None:
        # service-tier.test.ts:35-39
        assert body(OPENAI_RESPONSES, req(serviceTier="standard"))["service_tier"] == "default"
        assert body(OPENAI_RESPONSES, req(serviceTier="priority"))["service_tier"] == "priority"
        assert body(OPENAI_RESPONSES, req(serviceTier="flex"))["service_tier"] == "flex"
        assert body(OPENAI_RESPONSES, req(serviceTier="scale"))["service_tier"] == "scale"
        assert body(OPENAI_COMPLETIONS, req(serviceTier="priority"))["service_tier"] == "priority"

    def test_openai_sends_fast_on_both_surfaces(self) -> None:
        # service-tier.test.ts:46-47. Regression: `fast` shipped in openai-ts 7.x
        # and was missing from our KNOWN set, so a caller asking for Fast mode
        # silently received the project default. Probe-verified accepted on
        # gpt-5.5, with an invalid value rejected -- the field is validated, not
        # merely tolerated.
        assert body(OPENAI_RESPONSES, req(serviceTier="fast"))["service_tier"] == "fast"
        assert body(OPENAI_COMPLETIONS, req(serviceTier="fast"))["service_tier"] == "fast"

    def test_openai_passes_an_unknown_but_allowed_tier_through_else_falls_back_to_auto(
        self,
    ) -> None:
        # service-tier.test.ts:52-54
        assert body(OPENAI_RESPONSES, req(serviceTier="auto"))["service_tier"] == "auto"
        assert body(OPENAI_RESPONSES, req(serviceTier="turbo"))["service_tier"] == "auto"

    def test_anthropic_maps_standard_priority_flex_scale_unknown(self) -> None:
        # service-tier.test.ts:58-62
        assert body(ANTHROPIC, req(serviceTier="standard"))["service_tier"] == "standard_only"
        assert body(ANTHROPIC, req(serviceTier="priority"))["service_tier"] == "auto"
        assert body(ANTHROPIC, req(serviceTier="flex"))["service_tier"] == "standard_only"
        assert body(ANTHROPIC, req(serviceTier="scale"))["service_tier"] == "auto"
        assert body(ANTHROPIC, req(serviceTier="unknown"))["service_tier"] == "auto"

    def test_google_maps_flex_standard_priority_and_omits_auto_scale_unknown(self) -> None:
        # service-tier.test.ts:66-71
        assert body(GOOGLE, req(serviceTier="flex"))["serviceTier"] == "flex"
        assert body(GOOGLE, req(serviceTier="standard"))["serviceTier"] == "standard"
        assert body(GOOGLE, req(serviceTier="priority"))["serviceTier"] == "priority"
        assert "serviceTier" not in body(GOOGLE, req(serviceTier="auto"))
        assert "serviceTier" not in body(GOOGLE, req(serviceTier="scale"))
        assert "serviceTier" not in body(GOOGLE, req(serviceTier="turbo"))


# --- billed tier parsed onto Usage (raw + normalized) ---


class TestBilledServiceTierToUsage:
    """service-tier.test.ts:76.

    Unskipped with the adapters. The placeholder that stood here guessed a
    Python-shaped API -- `AnthropicAdapter(api_key=...)` and
    `usage.service_tier` -- which the port does not have and was never going to:
    an adapter takes the config OBJECT the TypeScript passes, and `usage` is a
    wire-level dict whose camelCase keys the response corpus froze. The
    assertions are the TypeScript's, unchanged; only the spelling of the calls
    is corrected against it.
    """
    def test_openai_responses(self) -> None:
        from combycode_llm_sdk.llm.providers.openai.responses import OpenAIResponsesAdapter

        # service-tier.test.ts:78-83
        a = OpenAIResponsesAdapter({"apiKey": "k"})
        flex = a.parse_response(
            {"id": "r", "model": "m", "output": [], "usage": {}, "service_tier": "flex"}, 1
        )
        assert flex["usage"]["serviceTier"] == "flex"
        assert flex["usage"]["pricingTier"] == "flex"
        default = a.parse_response(
            {"id": "r", "model": "m", "output": [], "usage": {}, "service_tier": "default"}, 1
        )
        assert default["usage"]["pricingTier"] == "standard"

    def test_anthropic_identity(self) -> None:
        from combycode_llm_sdk.llm.providers.anthropic.messages import AnthropicAdapter

        # service-tier.test.ts:87-93. Anthropic bills under its OWN tier names,
        # so raw and catalog key are the same string.
        a = AnthropicAdapter({"apiKey": "k"})
        r = a.parse_response(
            {
                "id": "r",
                "model": "m",
                "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 1, "service_tier": "batch"},
            },
            1,
        )
        assert r["usage"]["serviceTier"] == "batch"
        assert r["usage"]["pricingTier"] == "batch"

    def test_google_lower_cases_the_pricing_tier(self) -> None:
        from combycode_llm_sdk.llm.providers.google.generate import GoogleAdapter

        # service-tier.test.ts:97-106. The RAW value is kept as the provider sent
        # it, so a bill can be audited against it; the pricing key is lower-cased
        # because that is how the catalog is keyed.
        a = GoogleAdapter({"apiKey": "k"})
        r = a.parse_response(
            {
                "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "serviceTier": "FLEX",
                },
            },
            1,
        )
        assert r["usage"]["serviceTier"] == "FLEX"
        assert r["usage"]["pricingTier"] == "flex"

    def test_no_provider_tier_leaves_the_fields_unset(self) -> None:
        from combycode_llm_sdk.llm.providers.anthropic.messages import AnthropicAdapter

        # service-tier.test.ts:110-113. ABSENT, not null: a tier that was never
        # reported must not be priced as one that was.
        a = AnthropicAdapter({"apiKey": "k"})
        r = a.parse_response(
            {
                "id": "r",
                "model": "m",
                "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            1,
        )
        assert "serviceTier" not in r["usage"]
        assert "pricingTier" not in r["usage"]


class TestBilledTierMappersTheAdapterDelegatesTo:
    """The same expectations as `TestBilledServiceTierToUsage`, applied one layer
    down.

    TypeScript reads them off `adapter.parseResponse(...).usage`; the adapter's
    whole contribution is to call `openaiBilledTier` / `googleBilledTier` and copy
    the two fields across. Those two functions ARE in this batch, so the skip
    above would otherwise leave them with no test at all. Nothing is asserted here
    that service-tier.test.ts does not assert; the anthropic case has no ported
    function behind it and stays skipped only.
    """

    # These keys are camelCase, and were snake_case until the response
    # differential ran: the billed tier merges into the WIRE-level unified
    # `usage`, which the shared specs name and the recorded corpus froze. The
    # REQUEST-side `service_tier` asserted further up stays snake_case because
    # that is OpenAI's own field name. Both this test and the code it checks were
    # written from the same misreading, so it confirmed the mistake instead of
    # catching it -- every OpenAI cell in the differential disagreed on `usage`.

    def test_openai_default_normalises_to_standard(self) -> None:
        # service-tier.test.ts:80-83
        assert openai_billed_tier("flex") == {"serviceTier": "flex", "pricingTier": "flex"}
        assert openai_billed_tier("default") == {
            "serviceTier": "default",
            "pricingTier": "standard",
        }

    def test_google_lower_cases_the_pricing_tier_and_keeps_the_raw(self) -> None:
        # service-tier.test.ts:105-106
        assert google_billed_tier("FLEX") == {"serviceTier": "FLEX", "pricingTier": "flex"}

    def test_no_provider_tier_yields_nothing_to_copy(self) -> None:
        # service-tier.test.ts:112-113
        assert openai_billed_tier(None) == {}
        assert google_billed_tier(None) == {}
        assert openai_billed_tier("") == {}
        assert google_billed_tier("") == {}
        assert openai_billed_tier(7) == {}
        assert google_billed_tier(7) == {}


# --- model:tier selector sugar ---


@pytest.mark.skip(
    reason="depends-on-unported: src/helpers/client-resolver.ts (parseModelTier) "
    "-- service-tier.test.ts:118"
)
class TestParseModelTier:
    def test_strips_a_recognized_tier_suffix(self) -> None:
        from combycode_llm_sdk.helpers.client_resolver import parse_model_tier

        # service-tier.test.ts:120-123
        assert parse_model_tier("anthropic/claude-opus-4.8:priority") == {
            "model_id": "anthropic/claude-opus-4.8",
            "service_tier": "priority",
        }

    def test_leaves_openrouter_free_and_online_untouched(self) -> None:
        from combycode_llm_sdk.helpers.client_resolver import parse_model_tier

        # service-tier.test.ts:126-131
        assert parse_model_tier("openrouter/qwen/qwen3-coder:free") == {
            "model_id": "openrouter/qwen/qwen3-coder:free"
        }
        assert parse_model_tier("openrouter/perplexity/sonar:online") == {
            "model_id": "openrouter/perplexity/sonar:online"
        }

    def test_leaves_a_plain_model_id_untouched(self) -> None:
        from combycode_llm_sdk.helpers.client_resolver import parse_model_tier

        # service-tier.test.ts:134
        assert parse_model_tier("anthropic/claude-opus-4.8") == {
            "model_id": "anthropic/claude-opus-4.8"
        }


# --- cost prices by billed tier ---


@pytest.mark.skip(
    reason="depends-on-unported: src/plugins/cost-collector/collector.ts, "
    "src/bus/hook-bus.ts, src/catalog/catalog.ts -- service-tier.test.ts:170"
)
class TestCostPricesByServiceTier:
    """service-tier.test.ts:139-215.

    The whole describe, including the `ctx()` builder at :139 and the catalog
    fixture at :171-185, lands with the cost-collector area. Nothing about it is
    reachable from this batch.
    """

    def test_standard_no_tier_is_flat_rates(self) -> None:
        raise AssertionError("unreachable")

    def test_batch_tier_is_half_cost(self) -> None:
        raise AssertionError("unreachable")

    def test_priority_tier_is_premium_rates(self) -> None:
        raise AssertionError("unreachable")

    def test_records_the_billed_service_tier_on_the_entry(self) -> None:
        raise AssertionError("unreachable")

    def test_unknown_tier_falls_back_to_flat_standard(self) -> None:
        raise AssertionError("unreachable")
