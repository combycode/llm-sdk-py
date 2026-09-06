"""xAI's response registry: OpenAI Responses', with file extraction replaced.

Transposed from `unified-library-ts/src/llm/providers/xai/responses-registry.ts`.

It exists because switching the TypeScript adapter to the spec dropped this: the
shared transform called the OpenAI module function directly and the subclass
override was never consulted. No recorded xAI cell runs code execution, so the
response differential stayed green -- a unit test caught it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ....wire.interpreter import Ctx, Registry
from ..openai.parse_helpers import files_from_responses_output_item
from ..openai.responses_registry import OPENAI_RESPONSES_REGISTRY
from .parse_helpers import xai_code_exec_files


def _files(_arg: Any, ctx: Ctx) -> list[dict[str, Any]]:
    item = ctx.item.value if ctx.item else {}
    item = item if isinstance(item, Mapping) else {}
    return [*files_from_responses_output_item(item), *xai_code_exec_files(item)]


XAI_RESPONSES_REGISTRY = Registry(
    transforms={**OPENAI_RESPONSES_REGISTRY.transforms, "oaiRespFiles": _files},
    builders=dict(OPENAI_RESPONSES_REGISTRY.builders),
    predicates=dict(OPENAI_RESPONSES_REGISTRY.predicates),
    effects=dict(OPENAI_RESPONSES_REGISTRY.effects),
)

__all__ = ["XAI_RESPONSES_REGISTRY"]
