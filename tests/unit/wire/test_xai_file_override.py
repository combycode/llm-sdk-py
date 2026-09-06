"""xAI's file extraction survives the switch to specs -- both halves of it.

This is the one thing xAI parses differently from OpenAI Responses, and it is
the one thing no recorded cell exercises: no xAI cell in the corpus runs code
execution, so BOTH differentials stay green with the override deleted. That is
not a hypothetical -- the TypeScript switch dropped exactly this on the buffered
side, because the shared transform called the OpenAI module function directly
and the subclass override was never consulted.

A mutation sweep confirmed the gap on the stream side too: replacing the file
list with `[]` in `xai/stream_registry.py` left all 32 differential cells
passing. So the coverage has to come from a synthetic item, and it lives here
rather than in either differential because it is not corpus-derived.
"""

from __future__ import annotations

import json
from typing import Any

from combycode_llm_sdk.llm.providers.xai.responses_registry import XAI_RESPONSES_REGISTRY
from combycode_llm_sdk.llm.providers.xai.stream_registry import XAI_STREAM_REGISTRY
from combycode_llm_sdk.util.base64 import bytes_to_base64
from combycode_llm_sdk.wire.response_interpreter import build_response
from combycode_llm_sdk.wire.response_specs import get_response_spec
from combycode_llm_sdk.wire.stream_interpreter import create_stream_builder
from combycode_llm_sdk.wire.stream_specs import get_stream_spec


def sse(data: dict[str, Any]) -> dict[str, Any]:
    """The builder takes an SSE FRAME, not the decoded payload."""
    return {"event": data["type"], "data": json.dumps(data)}

#: A byte sequence that is NOT valid utf-8, so a base64 that round-trips it
#: proves the bytes went through unmodified rather than through a string.
PLOT = [137, 80, 78, 71, 13, 10, 26, 10]

#: The xAI shape: output files inline in the `logs` payload as a JSON string,
#: not as OpenAI-style `container_file_citation` annotations.
ITEM: dict[str, Any] = {
    "type": "code_interpreter_call",
    "id": "ci_1",
    "code": "print('hi')",
    "outputs": [
        {
            "type": "logs",
            "logs": json.dumps(
                {
                    "stdout": "hi\n",
                    "output_files": [
                        {
                            "file_name": "plot.png",
                            "mime_type": "image/png",
                            "data": PLOT,
                        }
                    ],
                }
            ),
        }
    ],
}

EXPECTED_FILE = {
    "data": bytes_to_base64(bytes(PLOT)),
    "name": "plot.png",
    "mimeType": "image/png",
    "source": "code_execution",
}


def test_the_buffered_parse_surfaces_the_inline_file() -> None:
    raw = {"id": "resp_1", "status": "completed", "output": [ITEM]}
    built = build_response(
        get_response_spec("xai/responses.response"),
        raw,
        XAI_RESPONSES_REGISTRY,
        extra={"latencyMs": 0, "raw": raw},
    )
    assert json.loads(json.dumps(built)).get("files") == [EXPECTED_FILE]


def test_the_streamed_parse_surfaces_the_inline_file() -> None:
    parse = create_stream_builder(get_stream_spec("xai/responses.stream"), XAI_STREAM_REGISTRY)
    events = parse(sse({"type": "response.output_item.done", "item": ITEM}))
    files = [e for e in json.loads(json.dumps(events)) if e.get("type") == "file"]
    assert files == [{"type": "file", "file": EXPECTED_FILE}]


def test_openai_responses_does_not_read_the_xai_shape() -> None:
    """The override is xAI's alone -- if the base registry also read `logs`,
    the two tests above would pass with the override deleted."""
    from combycode_llm_sdk.llm.providers.openai.responses_stream_registry import (
        OPENAI_RESPONSES_STREAM_REGISTRY,
    )

    parse = create_stream_builder(
        get_stream_spec("openai/responses.stream"), OPENAI_RESPONSES_STREAM_REGISTRY
    )
    events = parse(sse({"type": "response.output_item.done", "item": ITEM}))
    assert [e for e in events if e.get("type") == "file"] == []
