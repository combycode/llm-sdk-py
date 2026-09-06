"""xAI's stream registry: OpenAI Responses', with file extraction replaced.

Transposed from `unified-library-ts/src/llm/providers/xai/stream-registry.ts`.

The stream twin of `responses_registry`, and it exists for the same reason: xAI
returns code-execution files INLINE in the `code_interpreter_call` `logs`
payload rather than as container-file annotations. No recorded xAI cell runs code
execution, so the differential would not notice its absence here either.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ....wire.interpreter import Ctx, Registry
from ..openai.parse_helpers import files_from_responses_output_item
from ..openai.responses_stream_registry import OPENAI_RESPONSES_STREAM_REGISTRY
from .parse_helpers import xai_code_exec_files


def _files(_arg: Any, ctx: Ctx) -> list[dict[str, Any]]:
    raw = ctx.req.get("raw")
    raw = raw if isinstance(raw, Mapping) else {}
    item = raw.get("item")
    item = item if isinstance(item, Mapping) else {}
    return [
        {"type": "file", "file": f}
        for f in [*files_from_responses_output_item(item), *xai_code_exec_files(item)]
    ]


XAI_STREAM_REGISTRY = Registry(
    transforms={**OPENAI_RESPONSES_STREAM_REGISTRY.transforms, "oaiRespStreamFiles": _files},
    builders=dict(OPENAI_RESPONSES_STREAM_REGISTRY.builders),
    predicates=dict(OPENAI_RESPONSES_STREAM_REGISTRY.predicates),
    effects=dict(OPENAI_RESPONSES_STREAM_REGISTRY.effects),
)

__all__ = ["XAI_STREAM_REGISTRY"]
