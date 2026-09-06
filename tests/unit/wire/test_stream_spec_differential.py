"""Every stream spec emits what the TypeScript library emits, event for event.

The streaming half of the same check. `raw` is the ordered SSE events a provider
actually sent; `parsed` is the unified event sequence TypeScript made of them,
frozen at the moment of recording. Replay needs no network and no stub.

Sequence is the whole point here. A buffered parse can be wrong in one field; a
stream parse can be wrong in ORDER, or emit the right events for the wrong
event, and only a recorded sequence catches that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from combycode_llm_sdk.llm.providers.anthropic.stream_registry import ANTHROPIC_STREAM_REGISTRY
from combycode_llm_sdk.llm.providers.google.interactions_stream_registry import (
    GOOGLE_INTERACTIONS_STREAM_REGISTRY,
)
from combycode_llm_sdk.llm.providers.google.stream_registry import GOOGLE_STREAM_REGISTRY
from combycode_llm_sdk.llm.providers.openai.responses_stream_registry import (
    OPENAI_RESPONSES_STREAM_REGISTRY,
)
from combycode_llm_sdk.llm.providers.openai.stream_registry import OPENAI_STREAM_REGISTRY
from combycode_llm_sdk.llm.providers.openrouter.stream_registry import OPENROUTER_STREAM_REGISTRY
from combycode_llm_sdk.llm.providers.xai.stream_registry import XAI_STREAM_REGISTRY
from combycode_llm_sdk.wire.interpreter import Registry
from combycode_llm_sdk.wire.stream_interpreter import create_stream_builder
from combycode_llm_sdk.wire.stream_specs import STREAM_SPECS, get_stream_spec, stream_spec_id

CORPUS: dict[str, Any] = json.loads(
    (Path(__file__).resolve().parents[2] / "fixtures" / "response-golden.json").read_text(
        encoding="utf-8"
    )
)

#: spec id -> the registry supplying the names that spec calls.
REGISTRIES: dict[str, Registry] = {
    "anthropic/messages.stream": ANTHROPIC_STREAM_REGISTRY,
    "google/generate.stream": GOOGLE_STREAM_REGISTRY,
    "google/interactions.stream": GOOGLE_INTERACTIONS_STREAM_REGISTRY,
    "openai/completions.stream": OPENAI_STREAM_REGISTRY,
    "openai/responses.stream": OPENAI_RESPONSES_STREAM_REGISTRY,
    "openrouter/completions.stream": OPENROUTER_STREAM_REGISTRY,
    "xai/responses.stream": XAI_STREAM_REGISTRY,
}

STREAMED = [
    (cell_id, cell)
    for cell_id, cell in CORPUS.items()
    if cell.get("streaming") and stream_spec_id(cell["target"]) in REGISTRIES
]


def replay(cell: dict[str, Any]) -> Any:
    spec_id = stream_spec_id(cell["target"])
    # A FRESH parser per cell, exactly as a new conversation gets one. Reusing it
    # would let one cell's pending tool calls answer another cell's result.
    parse = create_stream_builder(get_stream_spec(spec_id), REGISTRIES[spec_id])
    events: list[Any] = []
    for event in cell["raw"]:
        events.extend(parse(event))
    return json.loads(json.dumps(events))


def test_it_is_actually_checking_something() -> None:
    assert len(STREAMED) >= 5
    unproven = [
        spec_id
        for spec_id in REGISTRIES
        if not any(stream_spec_id(c["target"]) == spec_id for _, c in STREAMED)
    ]
    assert unproven == []


def test_every_wired_registry_has_a_spec() -> None:
    assert [spec_id for spec_id in REGISTRIES if spec_id not in STREAM_SPECS] == []


@pytest.mark.parametrize(("cell_id", "cell"), STREAMED, ids=[c for c, _ in STREAMED])
def test_emits_what_typescript_emitted(cell_id: str, cell: dict[str, Any]) -> None:
    assert replay(cell) == cell["parsed"]
