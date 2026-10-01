"""OpenAI data residency: a named region instead of a hand-written host.

Transposed from `unified-library-ts/tests/unit/llm/data-residency.test.ts`.

OpenAI serves the same API from four hosts and a project provisioned for one
region must use that region's. Getting it wrong is not a silent fallback --
measured 2026-10-01 against `/v1/responses` from an unrestricted project, the
default host answers 200 while `us.` refuses with "incorrect regional hostname.
Please make your request to api.openai.com" and `eu.` with "only accessible by
projects with geography restrictions enabled". Each reply names the host that was
reached, so it is the routing that is proven, not just a status code. All four
hostnames resolve, so the only question is which one a caller means, and four
spellings checked at construction beat a URL string they have to get exactly
right.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

sys.path.insert(0, "src")

from combycode_llm_sdk.helpers.llm import default_adapter_factory
from combycode_llm_sdk.llm.client import LLMClient
from combycode_llm_sdk.llm.providers.openai.data_residency import resolve_data_residency

REGIONS = ("global", "us", "eu", "ae")


class TestResolveDataResidency:
    def test_it_maps_each_region_to_its_host(self) -> None:
        assert resolve_data_residency("global", None) == "https://api.openai.com"
        assert resolve_data_residency("us", None) == "https://us.api.openai.com"
        assert resolve_data_residency("eu", None) == "https://eu.api.openai.com"
        assert resolve_data_residency("ae", None) == "https://ae.api.openai.com"

    def test_it_carries_no_path_so_the_adapter_does_not_produce_v1_v1(self) -> None:
        # The upstream SDK's table includes `/v1`; ours must not, because
        # `completion_path()` supplies it.
        for region in REGIONS:
            assert "/v1" not in (resolve_data_residency(region, None) or "")

    def test_it_returns_nothing_when_no_region_was_asked_for(self) -> None:
        assert resolve_data_residency(None, None) is None
        assert resolve_data_residency(None, "https://proxy.test") is None

    def test_it_raises_when_combined_with_base_url_instead_of_picking_a_winner(self) -> None:
        # Both name the host. Honouring one would silently discard a configuration
        # the caller wrote, and they cannot tell which.
        with pytest.raises(ValueError, match="mutually exclusive"):
            resolve_data_residency("eu", "https://proxy.test")

    def test_it_refuses_a_region_it_does_not_know(self) -> None:
        # A typo like `"EU"` would otherwise fall through to the default host and
        # send EU-resident data to the global endpoint -- the one failure this
        # prevents.
        with pytest.raises(ValueError, match="Invalid dataResidency"):
            resolve_data_residency("EU", None)


def url_for(**over: Any) -> str:
    """The URL one request goes to, with nothing sent."""
    urls: list[str] = []

    def fetch(req: dict[str, Any], _opts: Any = None) -> dict[str, Any]:
        urls.append(str(req.get("url") or ""))
        return {
            "status": 200,
            "headers": {},
            "body": {
                "id": "resp_1",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        }

    config: dict[str, Any] = {
        "provider": "openai",
        "model": "gpt-5.4-nano",
        "apiKey": "k",
        "adapter": default_adapter_factory(),
        "api": "responses",
        "fetch": fetch,
        **over,
    }
    LLMClient(config).complete("hi", {"maxTokens": 16})
    return urls[0] if urls else ""


class TestAClientBuiltWithARegion:
    def test_it_sends_the_request_to_that_region(self) -> None:
        assert url_for(dataResidency="eu") == "https://eu.api.openai.com/v1/responses"

    def test_it_still_reaches_the_default_host_when_no_region_is_set(self) -> None:
        # The regression that carries every existing caller.
        assert url_for() == "https://api.openai.com/v1/responses"

    def test_it_treats_global_as_the_default_host_not_as_a_fifth_one(self) -> None:
        assert url_for(dataResidency="global") == "https://api.openai.com/v1/responses"

    def test_it_refuses_a_region_on_a_provider_that_has_no_regional_hosts(self) -> None:
        # Silently ignoring it would let someone believe their data was pinned to a
        # region when the option did nothing at all.
        with pytest.raises(ValueError, match="OpenAI option"):
            url_for(provider="anthropic", model="claude-haiku-4.5", api="messages",
                    dataResidency="eu")

    def test_it_refuses_a_region_alongside_a_base_url(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            url_for(dataResidency="eu", baseURL="https://proxy.test")

    def test_it_leaves_a_plain_base_url_working(self) -> None:
        assert url_for(baseURL="https://proxy.test") == "https://proxy.test/v1/responses"
