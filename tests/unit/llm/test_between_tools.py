"""`thinking={"mode": "between_tools"}` -- Anthropic's reason-between-tool-calls
mode, and the model gate it needs.

Transposed from `unified-library-ts/tests/unit/llm/between-tools.test.ts`.

Measured 2026-09-29 against EVERY active Anthropic chat model: exactly one
accepts it -- `claude-sonnet-5.5` -- and the other twelve answer
`400 "thinking.type.between_tools" is not supported for this model`,
`claude-opus-5.5` among them. A deliberately invalid thinking type is refused
everywhere, so the field is READ rather than tolerated, and a 200 means the
model genuinely takes it.

That distribution is why the gate runs the opposite way round to
`reasoning.canDisable`: almost every model can disable reasoning, so only an
explicit `False` stops that one. Almost none takes `between_tools`, so this one
is sent only on an explicit `True` -- one annotation instead of twelve, and no
caller ever buys a surprise 400.
"""

from __future__ import annotations

import contextlib
from typing import Any

import pytest

from combycode_llm_sdk.bus.hook_bus import HookBus
from combycode_llm_sdk.catalog.catalog import ModelCatalog
from combycode_llm_sdk.llm.async_client import AsyncLLMClient
from combycode_llm_sdk.llm.providers.anthropic.messages import AnthropicAdapter

CATALOG = ModelCatalog()
CATALOG.load_provider_defaults()
KEY = {"apiKey": "test-key"}


class TestTheCatalogRecordsWhoTakesIt:
    def test_it_marks_the_one_model_that_does(self) -> None:
        entry = CATALOG.get("anthropic", "claude-sonnet-5.5")
        assert entry is not None
        assert dict(entry)["reasoning"]["betweenTools"] is True

    @pytest.mark.parametrize(
        "model", ["claude-opus-5.5", "claude-sonnet-5", "claude-sonnet-4.6", "claude-haiku-4.5"]
    )
    def test_it_leaves_the_others_unmarked(self, model: str) -> None:
        entry = CATALOG.get("anthropic", model)
        assert entry is not None
        assert "betweenTools" not in dict(entry)["reasoning"]

    def test_the_one_model_does_not_donate_it_to_its_family(self) -> None:
        # `claude-sonnet` is ONE family spanning 4.5, 4.6, 5 and 5.5, and family
        # annotations are inherited. Three of those four were measured refusing
        # it, so donating a measured YES would look exactly like a measured fact.
        marked = [
            m["model"]
            for m in CATALOG.list()
            if m["provider"] == "anthropic"
            and m.get("family") == "claude-sonnet"
            and (m.get("reasoning") or {}).get("betweenTools") is True
        ]
        assert marked == ["claude-sonnet-5.5"]


class TestWhatReachesTheWire:
    @staticmethod
    def _body(model: str, thinking: Any) -> dict[str, Any]:
        entry = CATALOG.get("anthropic", model)
        assert entry is not None
        body = AnthropicAdapter(KEY).build_request(
            {
                "model": model,
                "wireSpec": dict(entry).get("wireSpec"),
                "messages": [{"role": "user", "content": "hi"}],
                "thinking": thinking,
            }
        ).body
        assert isinstance(body, dict)
        return body

    def test_it_sends_the_mode_as_its_own_thinking_type(self) -> None:
        body = self._body("claude-sonnet-5.5", {"mode": "between_tools"})
        assert body["thinking"] == {"type": "between_tools"}

    def test_it_never_sends_it_alongside_the_adaptive_shape(self) -> None:
        # They are siblings in Anthropic's ThinkingConfigParam, not variants:
        # one field, one shape.
        thinking = self._body("claude-sonnet-5.5", {"mode": "between_tools"})["thinking"]
        assert thinking["type"] == "between_tools"
        assert "effort" not in thinking

    def test_the_ordinary_modes_still_go_out_as_adaptive(self) -> None:
        assert self._body("claude-sonnet-5.5", {"mode": "on"})["thinking"]["type"] == "adaptive"


class TestAndTheGateInFrontOfIt:
    @staticmethod
    async def _run(model: str, thinking: Any) -> tuple[list[str], list[dict[str, Any]]]:
        hooks = HookBus()
        warnings: list[str] = []
        hooks.on(
            "onWarning",
            lambda ctx: warnings.append(str(dict(ctx).get("message")))
            if dict(ctx).get("code") == "request_adjusted"
            else None,
        )
        sent: list[dict[str, Any]] = []

        async def fetch(request: Any, _options: Any = None) -> dict[str, Any]:
            body = request.get("body") if isinstance(request, dict) else None
            sent.append(body if isinstance(body, dict) else {})
            return {"status": 200, "headers": {}, "body": {}}

        client = AsyncLLMClient(
            {
                "provider": "anthropic",
                "model": model,
                "apiKey": "sk-test",
                "adapter": AnthropicAdapter(KEY),
                "fetch": fetch,
                "hooks": hooks,
                "catalog": CATALOG,
            }
        )
        # The stub body is not a completion, so the parse may raise. The build
        # notes are emitted BEFORE the request goes out, which is the point.
        with contextlib.suppress(Exception):
            await client.complete("hi", {"maxTokens": 8, "thinking": thinking})
        return warnings, sent

    @pytest.mark.asyncio
    async def test_it_goes_through_silently_on_the_model_that_takes_it(self) -> None:
        warnings, sent = await self._run("claude-sonnet-5.5", {"mode": "between_tools"})
        assert sent[0].get("thinking") == {"type": "between_tools"}
        assert warnings == []

    @pytest.mark.asyncio
    async def test_it_is_dropped_everywhere_else_and_said_out_loud(self) -> None:
        # Downgraded rather than refused -- what Anthropic's own fallback
        # middleware does with this value. The caller gets a request that works
        # plus a warning.
        warnings, sent = await self._run("claude-opus-5.5", {"mode": "between_tools"})
        assert "thinking" not in sent[0]
        assert len(warnings) == 1
        assert "between_tools" in warnings[0]
        assert "claude-sonnet-5.5" in warnings[0]

    @pytest.mark.asyncio
    async def test_it_leaves_the_other_thinking_modes_alone(self) -> None:
        warnings, _ = await self._run("claude-opus-5.5", {"mode": "on"})
        assert warnings == []
