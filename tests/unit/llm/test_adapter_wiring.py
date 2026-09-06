"""Each adapter routes to the spec and registry it claims, proven on the corpus.

`test_response_spec_differential` and `test_stream_spec_differential` already
prove every spec+registry PAIR reproduces what TypeScript emitted. They say
nothing about the wiring: an adapter that names `openai/completions.response`
where it meant `openrouter/completions.response` passes both differentials and
still parses every OpenRouter turn with the wrong spec.

So this replays the same frozen corpus through the ADAPTER -- `parse_response`
and `create_stream_parser`, the two entry points the client actually calls --
and compares against the same frozen `parsed`. The differential proves the
specs; this proves the wiring reaches them.

The request side is covered the same way: an adapter's `build_request` has to
select a spec, and `_spec_id_for` is where a pin is read. The catalog's own
`wireSpec` values are the oracle there -- every spec a catalogued model pins must
be one the runtime can build.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from combycode_llm_sdk.catalog.catalog import ModelCatalog
from combycode_llm_sdk.llm.providers.anthropic.messages import AnthropicAdapter
from combycode_llm_sdk.llm.providers.google.generate import GoogleAdapter
from combycode_llm_sdk.llm.providers.google.interactions import GoogleInteractionsAdapter
from combycode_llm_sdk.llm.providers.openai.completions import OpenAIAdapter
from combycode_llm_sdk.llm.providers.openai.responses import OpenAIResponsesAdapter
from combycode_llm_sdk.llm.providers.openrouter.completions import OpenRouterAdapter
from combycode_llm_sdk.llm.providers.openrouter.responses import OpenRouterResponsesAdapter
from combycode_llm_sdk.llm.providers.xai.completions import XAIAdapter
from combycode_llm_sdk.llm.providers.xai.responses import XAIResponsesAdapter
from combycode_llm_sdk.wire.chat_specs import is_chat_spec

CORPUS: dict[str, Any] = json.loads(
    (Path(__file__).resolve().parents[2] / "fixtures" / "response-golden.json").read_text(
        encoding="utf-8"
    )
)

KEY = {"apiKey": "test-key"}

#: corpus `target` -> the adapter that owns it. The corpus targets are the same
#: strings `response_spec_id` / `stream_spec_id` derive from, so a mis-wired
#: adapter shows up here as a mismatch rather than as a green run.
ADAPTERS: dict[str, Any] = {
    "anthropic/messages": AnthropicAdapter(KEY),
    "openai/completions": OpenAIAdapter(KEY),
    "openai/responses": OpenAIResponsesAdapter(KEY),
    "google/generate": GoogleAdapter(KEY),
    "google/interactions": GoogleInteractionsAdapter(KEY),
    "openrouter/completions": OpenRouterAdapter(KEY),
    "xai/responses": XAIResponsesAdapter(KEY),
}

BUFFERED = [
    (cell_id, cell)
    for cell_id, cell in CORPUS.items()
    if not cell.get("streaming") and cell["target"] in ADAPTERS
]

STREAMED = [
    (cell_id, cell)
    for cell_id, cell in CORPUS.items()
    if cell.get("streaming") and cell["target"] in ADAPTERS
]


def test_it_is_actually_checking_something() -> None:
    # A target filter that silently matched nothing would make every case vacuous.
    assert len(BUFFERED) >= 7
    assert len(STREAMED) >= 7
    unproven = [
        target
        for target in ADAPTERS
        if not any(c["target"] == target for _, c in [*BUFFERED, *STREAMED])
    ]
    assert unproven == []


@pytest.mark.parametrize(("cell_id", "cell"), BUFFERED, ids=[c for c, _ in BUFFERED])
def test_parse_response_reaches_the_right_spec(cell_id: str, cell: dict[str, Any]) -> None:
    adapter = ADAPTERS[cell["target"]]
    built = adapter.parse_response(cell["raw"], 0)
    assert json.loads(json.dumps(built)) == cell["parsed"]


@pytest.mark.parametrize(("cell_id", "cell"), STREAMED, ids=[c for c, _ in STREAMED])
def test_create_stream_parser_reaches_the_right_spec(
    cell_id: str, cell: dict[str, Any]
) -> None:
    # A FRESH parser per cell, exactly as a new conversation gets one.
    parse = ADAPTERS[cell["target"]].create_stream_parser()
    events: list[Any] = []
    for event in cell["raw"]:
        events.extend(parse(event))
    assert json.loads(json.dumps(events)) == cell["parsed"]


class TestRequestSideRouting:
    """Which spec `build_request` selects, which no corpus cell can show."""

    def test_every_catalogued_wire_spec_is_one_the_runtime_can_build(self) -> None:
        # A catalog pin naming a spec the runtime does not carry is the failure
        # `is_chat_spec` exists to make survivable -- but for the chat families
        # that ARE migrated it should never happen, and only this notices.
        catalog = ModelCatalog.with_provider_defaults()
        pinned = {
            spec
            for info in catalog.list()
            if (spec := info.get("wireSpec")) and str(spec).split("@")[0].endswith(
                ("messages", "generate", "interactions", "chat-completions", "responses")
            )
        }
        unbuildable = sorted(s for s in pinned if not is_chat_spec(s))
        assert unbuildable == []

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            # The catalog pin wins when it names a spec the runtime carries.
            ("claude-opus-4-6", "anthropic/messages@4.6"),
            # A model the catalog does not know falls to the pin TABLE.
            ("claude-invented-9-9", "anthropic/messages@4.7"),
        ],
    )
    def test_anthropic_picks_its_chain_node(self, model: str, expected: str) -> None:
        adapter = AnthropicAdapter(KEY)
        assert adapter._spec_id_for({"model": model}) == expected

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            # 2.5 takes a token `thinkingBudget` and 400s on `thinkingLevel`;
            # 3.x takes the level. Sending one model the other's node is a 400 on
            # every thinking request, and no corpus cell shows the selection --
            # the recorded bodies are already built.
            ("gemini-2.5-flash", "google/generate@2.5"),
            ("gemini-3-pro", "google/generate@3"),
            ("gemini-invented-9", "google/generate@3"),
        ],
    )
    def test_google_picks_its_chain_node(self, model: str, expected: str) -> None:
        assert GoogleAdapter(KEY)._spec_id_for({"model": model}) == expected

    def test_googles_pin_is_ignored_when_it_names_another_provider(self) -> None:
        picked = GoogleAdapter(KEY)._spec_id_for(
            {"model": "gemini-3-pro", "wireSpec": "anthropic/messages@4.7"}
        )
        assert picked == "google/generate@3"

    def test_a_wire_spec_from_another_provider_is_ignored(self) -> None:
        # The prefix test is not decoration: a request carrying Google's pin must
        # not make the Anthropic adapter build Google's body.
        adapter = AnthropicAdapter(KEY)
        picked = adapter._spec_id_for(
            {"model": "claude-opus-4-6", "wireSpec": "google/generate@3"}
        )
        assert picked == "anthropic/messages@4.6"

    def test_openrouter_and_xai_keep_their_own_endpoints(self) -> None:
        # The subclasses' whole job is five overrides; a missed one sends
        # OpenRouter's body to api.openai.com with an OpenRouter key.
        assert OpenRouterAdapter(KEY).base_url() == "https://openrouter.ai"
        assert OpenRouterAdapter(KEY).completion_path() == "/api/v1/chat/completions"
        assert OpenRouterResponsesAdapter(KEY).completion_path() == "/api/v1/responses"
        assert XAIAdapter(KEY).base_url() == "https://api.x.ai"
        assert XAIResponsesAdapter(KEY).base_url() == "https://api.x.ai"
        assert XAIAdapter(KEY).wire_flavor == "xai"
        assert OpenRouterAdapter(KEY).wire_flavor == "openrouter"
        assert XAIResponsesAdapter(KEY).response_spec_id() == "xai/responses.response"
        assert OpenRouterAdapter(KEY).stream_spec_id() == "openrouter/completions.stream"
        # xAI's stream spec RESOLVES to OpenAI's -- it inherits the whole thing
        # and overrides only the registry -- so no replayed cell can tell the two
        # ids apart. The id is asserted directly for that reason: it is what
        # `create_stream_parser` pairs the xAI registry with, and pointing it at
        # OpenAI's id would drop xAI's file extraction with every test still green.
        assert XAIResponsesAdapter(KEY).stream_spec_id() == "xai/responses.stream"
        assert OpenRouterResponsesAdapter(KEY).base_url() == "https://openrouter.ai"

    def test_google_streams_from_a_different_url_not_a_body_flag(self) -> None:
        adapter = GoogleAdapter(KEY)
        req = {"model": "gemini-3-pro", "messages": [{"role": "user", "content": "hi"}]}
        built = adapter.build_request(req)
        adapter.enable_streaming(built, req)
        assert built.path == "/v1beta/models/gemini-3-pro:streamGenerateContent?alt=sse"
        assert "stream" not in built.body
