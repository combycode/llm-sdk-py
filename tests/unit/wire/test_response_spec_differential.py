"""Every response spec builds what the TypeScript library builds, exactly.

This is the check the whole port exists to pass. `tests/fixtures/response-golden.json`
is vendored byte-for-byte from the TypeScript tree: `raw` is what a provider
actually sent, and `parsed` is what the TypeScript library made of it, frozen at
the moment of recording. Neither side recomputes `parsed`, so it is the same
oracle for both languages -- and a Python parse that disagrees is a Python bug,
not a difference of opinion.

On field naming: the interpreter's output is camelCase (`toolCalls`,
`finishReason`) because the SPECS are shared data and name their fields that way.
That is the wire-level unified shape. The public Python API is snake_case
attributes (`usage.cached_tokens`); the mapping between them is a separate layer
and is deliberately not exercised here -- comparing at the camelCase layer is
what lets one corpus serve both languages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from combycode_llm_sdk.llm.providers.anthropic.response_registry import (
    ANTHROPIC_RESPONSE_REGISTRY,
)
from combycode_llm_sdk.llm.providers.google.interactions_registry import (
    GOOGLE_INTERACTIONS_REGISTRY,
)
from combycode_llm_sdk.llm.providers.google.response_registry import GOOGLE_RESPONSE_REGISTRY
from combycode_llm_sdk.llm.providers.openai.response_registry import OPENAI_RESPONSE_REGISTRY
from combycode_llm_sdk.llm.providers.openai.responses_registry import OPENAI_RESPONSES_REGISTRY
from combycode_llm_sdk.llm.providers.openrouter.response_registry import (
    OPENROUTER_RESPONSE_REGISTRY,
)
from combycode_llm_sdk.llm.providers.xai.responses_registry import XAI_RESPONSES_REGISTRY
from combycode_llm_sdk.wire.interpreter import Registry
from combycode_llm_sdk.wire.response_interpreter import build_response
from combycode_llm_sdk.wire.response_specs import (
    RESPONSE_SPECS,
    get_response_spec,
    response_spec_id,
)

CORPUS: dict[str, Any] = json.loads(
    (Path(__file__).resolve().parents[2] / "fixtures" / "response-golden.json").read_text(
        encoding="utf-8"
    )
)

#: spec id -> the registry supplying the names that spec calls. Names resolve per
#: PROVIDER, not globally: `anthropicFinish` means nothing to an OpenAI spec.
REGISTRIES: dict[str, Registry] = {
    "anthropic/messages.response": ANTHROPIC_RESPONSE_REGISTRY,
    "openai/completions.response": OPENAI_RESPONSE_REGISTRY,
    "openai/responses.response": OPENAI_RESPONSES_REGISTRY,
    "google/generate.response": GOOGLE_RESPONSE_REGISTRY,
    "google/interactions.response": GOOGLE_INTERACTIONS_REGISTRY,
    "openrouter/completions.response": OPENROUTER_RESPONSE_REGISTRY,
    # xAI subclasses OpenAI's Responses adapter but DOES extend file extraction.
    "xai/responses.response": XAI_RESPONSES_REGISTRY,
}

BUFFERED = [
    (cell_id, cell)
    for cell_id, cell in CORPUS.items()
    if not cell.get("streaming") and response_spec_id(cell["target"]) in REGISTRIES
]


def build(cell: dict[str, Any]) -> Any:
    spec_id = response_spec_id(cell["target"])
    built = build_response(
        get_response_spec(spec_id),
        cell["raw"],
        REGISTRIES[spec_id],
        extra={"latencyMs": 0, "raw": cell["raw"]},
    )
    # The TypeScript output was JSON round-tripped into the corpus, so keys whose
    # value was undefined are already gone. Compare like for like.
    return json.loads(json.dumps(built))


def test_it_is_actually_checking_something() -> None:
    # A filter that silently matched nothing would make every case below vacuous.
    assert len(BUFFERED) >= 7
    # And every registry wired here must have cells behind it, or it is unproven.
    unproven = [
        spec_id
        for spec_id in REGISTRIES
        if not any(response_spec_id(c["target"]) == spec_id for _, c in BUFFERED)
    ]
    assert unproven == []


def test_every_wired_registry_has_a_spec() -> None:
    missing = [spec_id for spec_id in REGISTRIES if spec_id not in RESPONSE_SPECS]
    assert missing == []


@pytest.mark.parametrize(("cell_id", "cell"), BUFFERED, ids=[c for c, _ in BUFFERED])
def test_builds_what_typescript_built(cell_id: str, cell: dict[str, Any]) -> None:
    assert build(cell) == cell["parsed"]


def test_shares_one_object_between_content_and_tool_calls() -> None:
    # Deep equality above cannot see reference identity, and the adapters'
    # behaviour depends on it: a consumer mutating toolCalls[0] must see the
    # change in content too.
    cell = CORPUS["anthropic/messages::tools"]
    spec_id = response_spec_id(cell["target"])
    built = build_response(
        get_response_spec(spec_id),
        cell["raw"],
        REGISTRIES[spec_id],
        extra={"latencyMs": 0, "raw": cell["raw"]},
    )
    calls = built["toolCalls"]
    assert len(calls) > 0
    assert any(part is calls[0] for part in built["content"])


def test_carries_hosted_tool_output_and_files_through_the_default_case() -> None:
    # The branch with no discriminator case of its own: *_tool_result attaches
    # stdout to its call, and code-execution results contribute files.
    built = build(CORPUS["anthropic/messages::builtin.codeexec"])
    calls = built.get("builtinToolCalls") or []
    assert len(calls) > 0
    assert any(isinstance(c.get("output"), str) and c["output"] for c in calls)
