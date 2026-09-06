"""OpenAI moderations adapter -- POST /v1/moderations.

Transposed from `unified-library-ts/src/llm/providers/openai/moderations.ts`.

Supports text and image+text content-part input, as described in
<https://platform.openai.com/docs/api-reference/moderations/create>. All HTTP
flows through the injected `EngineFetch`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ....network.types import EngineFetch, HttpRequest
from ....wire.interpreter import Registry, build_from_spec, js_json
from ....wire.service_specs import service_spec
from ...wire_transforms import make_registry

#: `interface OpenAIModerationAdapterConfig` (moderations.ts:20).
OpenAIModerationAdapterConfig = dict[str, Any]

OPENAI_MODERATION_BASE_URL = "https://api.openai.com"
OPENAI_MODERATION_PATH = "/v1/moderations"
OPENAI_MODERATION_DEFAULT_MODEL = "omni-moderation-latest"

#: `ModerationResult` (helpers/moderate-types.ts:33) --
#: `{flagged, categories, categoryScores, categoryAppliedInputTypes?}`.
#:
#: The camelCase of `categoryScores` and `categoryAppliedInputTypes` is not
#: cosmetic: the wire sends `category_scores` and `category_applied_input_types`,
#: and `_parse_raw_result` below is the ONLY place the two spellings meet. This
#: port shipped them snake_cased once, which made every consumer that read the
#: TypeScript field name find nothing.
ModerationResult = dict[str, Any]

#: A text or image_url content part for multimodal moderation input:
#: `{'type': 'text', 'text': ...}` or
#: `{'type': 'image_url', 'image_url': {'url': ...}}`.
ModerationContentPart = dict[str, Any]


class OpenAIModerationAdapter:
    """`class OpenAIModerationAdapter` (moderations.ts:29)."""

    def __init__(self, config: OpenAIModerationAdapterConfig) -> None:
        self._api_key: str = config["apiKey"]
        self._base_url: str = config.get("baseURL") or OPENAI_MODERATION_BASE_URL
        #: Named code these specs need. Empty handles: the moderation spec calls
        #: no adapter-owned rule.
        self._wire_registry: Registry = make_registry({})

    def _from_spec(
        self,
        spec_id: str,
        input_: Mapping[str, Any],
        model: str,
        response_type: str = "json",
        raw_bytes: Any = None,
    ) -> HttpRequest:
        """Build one request from its spec, then add the engine metadata.

        `bodyKind: none` arrives as `no_body` and `raw` as `raw_body`; the engine
        wants the body field absent in the first case and the caller's bytes in
        the second.
        """
        built = build_from_spec(
            service_spec(spec_id),
            input_,
            self._wire_registry,
            "openai",
            None,
            {"baseURL": self._base_url, "apiKey": self._api_key},
        )
        req: HttpRequest = {
            "url": built.url,
            "method": built.method,
            "headers": built.headers or {},
            "provider": "openai",
            "model": model,
            "responseType": response_type,
        }
        if built.raw_body:
            req["body"] = raw_bytes
            req["rawBody"] = True
        elif not built.no_body:
            req["body"] = built.body
        return req

    def build_moderate_request(self, input_: Any, model: str) -> HttpRequest:
        """Report-only classification.

        `input_` reaches the wire untouched: a string, a list of strings, or
        content parts are all accepted.
        """
        return self._from_spec("openai/moderations", {"model": model, "input": input_}, model)

    def moderate(self, input_: Any, model: str, fetch: Any) -> list[ModerationResult]:
        """Classify with a SYNCHRONOUS fetch."""
        return self._read(fetch(self.build_moderate_request(input_, model)))

    async def amoderate(
        self, input_: Any, model: str, fetch: EngineFetch
    ) -> list[ModerationResult]:
        """Classify with an async fetch. The twin of `moderate`; only the wait
        differs, so the response reading is shared below."""
        return self._read(await fetch(self.build_moderate_request(input_, model)))

    @staticmethod
    def _read(res: Mapping[str, Any]) -> list[ModerationResult]:
        if res["status"] >= 400:
            raise RuntimeError(
                f"OpenAI moderations failed ({res['status']}): {js_json(res.get('body'))}"
            )
        data = res.get("body") or {}
        return [_parse_raw_result(r) for r in (data.get("results") or [])]


def _parse_raw_result(raw: Mapping[str, Any]) -> ModerationResult:
    """The wire's snake_case to the facade's camelCase -- the only crossing point."""
    out: ModerationResult = {
        "flagged": raw.get("flagged"),
        "categories": raw.get("categories"),
        "categoryScores": raw.get("category_scores"),
    }
    if raw.get("category_applied_input_types") is not None:
        out["categoryAppliedInputTypes"] = raw["category_applied_input_types"]
    return out


__all__ = [
    "OPENAI_MODERATION_BASE_URL",
    "OPENAI_MODERATION_DEFAULT_MODEL",
    "OPENAI_MODERATION_PATH",
    "ModerationContentPart",
    "ModerationResult",
    "OpenAIModerationAdapter",
]
