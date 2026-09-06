"""The named-code registry the wire specs delegate to.

Everything here is bound to the REAL library internals, never reimplemented -- if
the interpreter's output matches the adapter's, it is because the spec drove the
same code, not because I wrote a second copy that happens to agree.

What lands in this file is the honest answer to "what cannot be data": structural
message/content transformation, schema-shape rules, and one variant rule that is
arithmetic rather than a pattern.

Transposed from `unified-library-ts/src/llm/wire-transforms.ts`.

TWO TRANSPOSITIONS worth stating, because everything here reaches the wire:

  the request is read through `get_path`, not by attribute access. That is the
  interpreter's own convention, it accepts a mapping or an object exactly as
  TypeScript's optional chaining accepts either, and it returns `MISSING` where
  TypeScript returns `undefined`. `_undef` converts at the boundary of any ported
  helper whose TypeScript parameter is `T | undefined`, because `MISSING` is an
  object and therefore truthy in Python.

  `js_truthy` / `js_string` are used wherever the TypeScript relies on
  `Boolean(x)` or `String(x)`. `Boolean([])` is `true` in JavaScript and
  `bool([])` is `False` in Python; `String(true)` is `"true"` and `str(True)` is
  `"True"`. Both differences change what a provider receives.

The KEYS of the four tables are the names the etalon specs use. They stay
camelCase, unchanged, because `specs/` is byte-identical with the TypeScript tree
and a renamed key is a spec that no longer resolves.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..util.base64 import bytes_to_base64
from ..util.hash import fnv1a32_hex
from ..util.source_image import (
    google_image_part,
    google_veo_image,
    normalize_image_source,
    openai_image_ref,
    to_data_url,
    xai_image_ref,
    xai_video_ref,
)
from ..wire.interpreter import MISSING, Ctx, Registry, get_path, is_obj, js_string, js_truthy
from .audio.voices import resolve_voice
from .moderation.native import build_native_moderation
from .providers.google.tiers import google_request_tier
from .providers.openai.tiers import openai_request_tier
from .providers.xai.tiers import xai_request_tier
from .types.schema_utils import ensure_additional_properties, strict_support

#: The hand-written adapter methods the specs cannot express as data.
#:
#: All optional: a registry built by ONE adapter to drive its own spec carries
#: only its own handle, and each transform below is reached only from that
#: provider's spec. Requiring the full set would force every adapter to import
#: every other adapter just to build its own request -- so `make_registry({})` is
#: a supported call, and a rule that needs an absent handle is simply never
#: reached.
#:
#: Keys, snake_cased from `interface AdapterHandles`: `anthropic`, `google`,
#: `openai_responses`, `openai_completions`, `google_interactions`,
#: `openrouter_media`.
AdapterHandles = Mapping[str, Any]


def _undef(v: Any) -> Any:
    """`MISSING` -> `None` at the boundary of a ported `T | undefined` parameter.

    The interpreter returns `MISSING` for an absent path; the ported helpers take
    `None` for `undefined`. Without this a helper's `if not voice` sees a truthy
    sentinel object and takes the wrong branch.
    """
    return None if v is MISSING else v


def _nullish(v: Any, fallback: Any) -> Any:
    """`v ?? fallback` -- null AND undefined, and nothing else."""
    return fallback if v is None or v is MISSING else v


def _is_function_tool_value(t: Any) -> bool:
    """`(t: any) => !t?.type || t.type === 'function'` -- wire-transforms.ts:30.

    `!t?.type` is a truthiness test, so `type: ''` counts as a function tool.
    Treating it as a builtin would send a tool with no name to the provider
    (type-helpers.test.ts:48).
    """
    ty = get_path(t, "type")
    return not js_truthy(ty) or ty == "function"


def _tools_include_builtin(ctx: Ctx, builtin: str) -> bool:
    """`ctx.req.tools?.some(t => !isFunctionToolValue(t) && t.type === <builtin>)`."""
    tools = get_path(ctx.req, "tools")
    if not isinstance(tools, list):
        return False
    return any(
        not _is_function_tool_value(t) and get_path(t, "type") == builtin for t in tools
    )


def make_registry(a: AdapterHandles) -> Registry:
    transforms: dict[str, Any] = {}
    builders: dict[str, Any] = {}
    predicates: dict[str, Any] = {}
    effects: dict[str, Any] = {}

    # -- schema shape rules --------------------------------------------------

    transforms["ensureAdditionalProperties"] = lambda v, ctx: ensure_additional_properties(v)

    def anthropic_tool_schema(tool: Any, ctx: Ctx) -> Any:
        """Anthropic: `input_schema` is hardened only on the strict path."""
        params = get_path(tool, "parameters")
        return (
            ensure_additional_properties(params)
            if get_path(tool, "strict") is True
            else params
        )

    transforms["anthropicToolSchema"] = anthropic_tool_schema

    def openai_tool_params(tool: Any, ctx: Ctx) -> Any:
        """OpenAI chat-completions: same rule, different field."""
        params = get_path(tool, "parameters")
        return (
            ensure_additional_properties(params)
            if get_path(tool, "strict") is True
            else params
        )

    transforms["openaiToolParams"] = openai_tool_params

    def openai_structured_strict(_v: Any, ctx: Ctx) -> Any:
        """OpenAI structured output: strict defaults to whatever the schema supports."""
        s = get_path(ctx.req, "structured")
        schema = ensure_additional_properties(get_path(s, "schema"))
        return _nullish(get_path(s, "strict"), strict_support(schema, "openai").ok)

    transforms["openaiStructuredStrict"] = openai_structured_strict

    def openai_structured_schema(_v: Any, ctx: Ctx) -> Any:
        s = get_path(ctx.req, "structured")
        raw = get_path(s, "schema")
        schema = ensure_additional_properties(raw)
        strict = _nullish(get_path(s, "strict"), strict_support(schema, "openai").ok)
        return schema if js_truthy(strict) else raw

    transforms["openaiStructuredSchema"] = openai_structured_schema

    transforms["openaiResponsesStructuredSchema"] = lambda _v, ctx: (
        # Responses always sends the hardened schema (differs from completions).
        ensure_additional_properties(get_path(ctx.req, "structured.schema"))
    )

    def openai_responses_tool_strict(tool: Any, ctx: Ctx) -> Any:
        """Responses tools: strict defaults from schema support, per tool."""
        return _nullish(
            get_path(tool, "strict"),
            strict_support(
                ensure_additional_properties(get_path(tool, "parameters")), "openai"
            ).ok,
        )

    transforms["openaiResponsesToolStrict"] = openai_responses_tool_strict

    # -- provider value maps that are already functions in the library -------

    transforms["googleTier"] = lambda v, ctx: google_request_tier(_undef(v))
    transforms["openaiTier"] = lambda v, ctx: openai_request_tier(_undef(v))

    # -- audio ---------------------------------------------------------------

    transforms["resolveVoiceOpenAI"] = lambda _v, ctx: _nullish(
        resolve_voice("openai", _undef(get_path(ctx.req, "audio.voice"))), "alloy"
    )
    transforms["resolveVoiceGoogle"] = lambda _v, ctx: resolve_voice(
        "google", _undef(get_path(ctx.req, "audio.voice"))
    )

    def openai_audio_format(_v: Any, ctx: Ctx) -> Any:
        f = get_path(ctx.req, "audio.format")
        return "wav" if not js_truthy(f) or f == "aac" else f

    transforms["openaiAudioFormat"] = openai_audio_format

    # -- moderation ----------------------------------------------------------

    transforms["nativeModeration"] = lambda _v, ctx: build_native_moderation(
        _undef(get_path(ctx.req, "moderation")),
        _undef(get_path(ctx.req, "providerOptions.moderationPolicy")),
    )

    # -- model path ----------------------------------------------------------

    def _models_prefixed(ctx: Ctx) -> str:
        model = get_path(ctx.req, "model")
        return model if model.startswith("models/") else f"models/{model}"

    transforms["googleModelPath"] = lambda _v, ctx: _models_prefixed(ctx)
    transforms["googleGeneratePath"] = lambda _v, ctx: (
        f"/v1beta/{_models_prefixed(ctx)}:generateContent"
    )

    #: providerOptions.cachedContent is forwarded only when it is a non-empty string.
    transforms["stringOnly"] = lambda v, ctx: v if isinstance(v, str) and v else None

    # -- media: structural parts, reusing the library's own normalisers ------

    #: Veo first-frame image.
    transforms["googleVeoImagePart"] = lambda _v, ctx: google_veo_image(
        normalize_image_source(get_path(ctx.req, "sourceImage"))
    )
    #: Source image for an image-to-image edit.
    transforms["googleSourceImagePart"] = lambda _v, ctx: google_image_part(
        normalize_image_source(get_path(ctx.req, "sourceImage"))
    )
    #: OpenAI image reference (data URL or file_id) for the source image.
    transforms["openaiSourceImageRef"] = lambda _v, ctx: openai_image_ref(
        normalize_image_source(get_path(ctx.req, "sourceImage"))
    )
    #: Same, for an edit mask.
    transforms["openaiMaskRef"] = lambda _v, ctx: openai_image_ref(
        normalize_image_source(get_path(ctx.req, "mask"))
    )
    #: OpenAI TTS voice. Reads `params.voice`, NOT the chat request's
    #: `audio.voice` -- reusing the chat transform here silently produced the
    #: default, which the differential caught.
    transforms["openaiTtsVoice"] = lambda _v, ctx: _nullish(
        resolve_voice("openai", _undef(get_path(ctx.req, "params.voice"))), "alloy"
    )
    #: Realtime audio goes on the wire base64-encoded.
    transforms["realtimeAudioBase64"] = lambda _v, ctx: bytes_to_base64(
        get_path(ctx.req, "audio")
    )
    #: turnComplete defaults to true; only an explicit `false` suppresses it.
    transforms["realtimeTurnComplete"] = lambda _v, ctx: (
        get_path(ctx.req, "turnComplete") is not False
    )

    #: xAI media refs, reusing the library's own normalisers.
    transforms["xaiSourceImageRef"] = lambda _v, ctx: xai_image_ref(
        normalize_image_source(get_path(ctx.req, "sourceImage"))
    )
    transforms["xaiSourceVideoRef"] = lambda _v, ctx: xai_video_ref(
        get_path(ctx.req, "sourceVideo")
    )
    #: OpenRouter sends the source image as a data URL inside a chat part.
    transforms["openrouterDataUrl"] = lambda _v, ctx: to_data_url(
        normalize_image_source(get_path(ctx.req, "sourceImage"))
    )
    #: image_config is built by the adapter's own private helper.
    transforms["openrouterImageConfig"] = lambda _v, ctx: a["openrouter_media"].image_config(
        get_path(ctx.req, "params")
    )

    def xai_batch_name(_v: Any, ctx: Ctx) -> str:
        """xAI names a batch from its contents; mirrors the adapter's batchName()."""
        reqs = _nullish(get_path(ctx.req, "requests"), [])
        # `Array.prototype.join` renders null and undefined as the empty string.
        ids = "\0".join(
            "" if (c is None or c is MISSING) else js_string(c)
            for c in (get_path(r, "customId") for r in reqs)
        )
        return f"batch_{len(reqs)}_{fnv1a32_hex(ids)}"

    transforms["xaiBatchName"] = xai_batch_name

    def google_custom_metadata(v: Any, ctx: Ctx) -> list[dict[str, str]]:
        """Google file-search metadata is a LIST of {key,value} pairs with STRING
        values -- an object is rejected, and so is a numeric value, which is why
        the coercion belongs to the rule rather than to the caller."""
        src = _nullish(v, {})
        return [{"key": key, "value": js_string(value)} for key, value in src.items()]

    transforms["googleCustomMetadata"] = google_custom_metadata

    #: Embeddings: `input` is always an array on the wire.
    transforms["asArray"] = lambda v, ctx: v if isinstance(v, list) else [v]
    #: Google batch names itself after the request count -- deterministic, unlike xAI's.
    transforms["requestCount"] = lambda _v, ctx: js_string(
        len(_nullish(get_path(ctx.req, "requests"), []))
    )

    #: Sora sends `seconds` as a string even though the unified param is a number.
    transforms["stringify"] = lambda v, ctx: js_string(v)

    #: Gemini TTS voice, with the adapter's default.
    transforms["googleTtsVoice"] = lambda _v, ctx: _nullish(
        resolve_voice("google", _undef(get_path(ctx.req, "params.voice"))), "Kore"
    )

    # -- the variant rule a pattern cannot express (FINDING) -----------------
    #: Version arithmetic: family-then-version ids compared against 4.6.
    #: This is the one variant that resists being data, and it is exactly the
    #: knowledge a catalog spec-pin would carry instead.

    # -- builders ------------------------------------------------------------
    # Structural message/content transformation -- genuinely code, reused verbatim.

    def anthropic_messages(ctx: Ctx) -> list[Any]:
        req = ctx.req
        cache_auto_last = get_path(req, "cache") == "auto"
        messages = get_path(req, "messages")
        last_idx = len(messages) - 1
        return [
            a["anthropic"].build_message(
                m, req, cache_auto_last and i == last_idx, ctx.notes
            )
            for i, m in enumerate(messages)
        ]

    builders["anthropicMessages"] = anthropic_messages

    def google_contents(ctx: Ctx) -> list[Any]:
        out: list[Any] = []
        for msg in get_path(ctx.req, "messages"):
            if get_path(msg, "role") == "system":
                continue  # carried by systemInstruction
            out.append(a["google"].build_content(msg, ctx.notes))
        return out

    builders["googleContents"] = google_contents

    def openai_chat_messages(ctx: Ctx) -> list[Any]:
        out: list[Any] = []
        system = get_path(ctx.req, "system")
        if js_truthy(system):
            out.append({"role": "system", "content": system})
        for msg in get_path(ctx.req, "messages"):
            out.extend(a["openai_completions"].build_messages(msg, ctx.notes))
        return out

    builders["openaiChatMessages"] = openai_chat_messages

    def google_interactions_input(ctx: Ctx) -> list[Any]:
        out: list[Any] = []
        for msg in get_path(ctx.req, "messages"):
            out.extend(a["google_interactions"].build_input_items(msg, ctx.notes))
        return out

    builders["googleInteractionsInput"] = google_interactions_input

    def openai_responses_input(ctx: Ctx) -> list[Any]:
        out: list[Any] = []
        tool_names: dict[str, str] = {}
        for msg in get_path(ctx.req, "messages"):
            out.extend(a["openai_responses"].build_input_items(msg, tool_names, ctx.notes))
        return out

    builders["openaiResponsesInput"] = openai_responses_input

    # -- predicates ----------------------------------------------------------

    def realtime_has_text(ctx: Ctx) -> bool:
        """Realtime text is sent when it is non-null -- an empty string still
        counts, which `truthy` would miss."""
        v = get_path(ctx.req, "text")
        return v is not None and v is not MISSING

    predicates["realtimeHasText"] = realtime_has_text

    predicates["isFunctionTool"] = lambda ctx: _is_function_tool_value(
        ctx.item.value if ctx.item else MISSING
    )
    predicates["toolChoiceIsString"] = lambda ctx: isinstance(
        get_path(ctx.req, "toolChoice"), str
    )

    def _cache_flag(ctx: Ctx, key: str) -> bool:
        cache = get_path(ctx.req, "cache")
        return cache == "auto" or (is_obj(cache) and js_truthy(get_path(cache, key)))

    predicates["anthropicCacheSystem"] = lambda ctx: _cache_flag(ctx, "system")
    predicates["anthropicCacheTools"] = lambda ctx: _cache_flag(ctx, "tools")

    def has_file_ref(ctx: Ctx) -> bool:
        """Content-derived: needs to look inside message parts."""
        for m in get_path(ctx.req, "messages"):
            content = get_path(m, "content")
            if isinstance(content, str):
                continue
            for p in content:
                s = get_path(p, "source")
                if get_path(s, "type") in ("provider_ref", "file"):
                    return True
        return False

    predicates["hasFileRef"] = has_file_ref

    def has_audio_input(ctx: Ctx) -> bool:
        for m in get_path(ctx.req, "messages"):
            content = get_path(m, "content")
            if isinstance(content, list) and any(
                get_path(p, "type") == "audio" for p in content
            ):
                return True
        return False

    predicates["hasAudioInput"] = has_audio_input

    predicates["usesCodeExec"] = lambda ctx: _tools_include_builtin(ctx, "code_interpreter")

    def has_user_profile_id(ctx: Ctx) -> bool:
        v = get_path(ctx.req, "providerOptions.userProfileId")
        return isinstance(v, str) and len(v) > 0

    predicates["hasUserProfileId"] = has_user_profile_id

    #: image_config ships only when the adapter's helper produced something.
    predicates["openrouterHasImageConfig"] = lambda ctx: (
        len(_nullish(a["openrouter_media"].image_config(get_path(ctx.req, "params")), {})) > 0
    )

    #: Google emits speechConfig only when a voice actually resolves.
    predicates["googleHasVoice"] = lambda ctx: js_truthy(
        resolve_voice("google", _undef(get_path(ctx.req, "audio.voice")))
    )

    #: Only the multi-agent grok uses reasoning.effort (as an agent count).
    predicates["xaiMultiAgent"] = lambda ctx: "multi-agent" in js_string(
        get_path(ctx.req, "model")
    )

    def wants_native_moderation(ctx: Ctx) -> bool:
        mod = get_path(ctx.req, "moderation")
        asked = js_truthy(mod) and get_path(mod, "mode") != "emulate"
        policy = get_path(ctx.req, "providerOptions.moderationPolicy")
        return bool(asked or js_truthy(policy))

    predicates["wantsNativeModeration"] = wants_native_moderation

    # -- effects: flavor overlays, the ops that need code rather than data ---

    def xai_system_into_input(ctx: Ctx) -> None:
        """xAI puts the system prompt in `input` as a role:system item. Structural,
        so it is code -- but it is the ONLY structural op in four overlays."""
        system = get_path(ctx.req, "system")
        if js_truthy(system) and js_truthy(ctx.body.get("instructions")):
            ctx.body["input"].insert(0, {"role": "system", "content": system})
            del ctx.body["instructions"]

    effects["xaiSystemIntoInput"] = xai_system_into_input

    def xai_service_tier(ctx: Ctx) -> None:
        """xAI's tier enum is DEFAULT|PRIORITY, narrower than the inherited map."""
        t = xai_request_tier(_undef(get_path(ctx.req, "serviceTier")))
        if t:
            ctx.body["service_tier"] = t
        else:
            ctx.body.pop("service_tier", None)

    effects["xaiServiceTier"] = xai_service_tier

    def xai_code_exec_include(ctx: Ctx) -> None:
        """Code-execution outputs are returned only when asked for by `include`."""
        if not _tools_include_builtin(ctx, "code_interpreter"):
            return
        existing = ctx.body.get("include")
        base = existing if isinstance(existing, list) else []
        # `new Set([...])` then spread: insertion order, duplicates dropped.
        ctx.body["include"] = list(dict.fromkeys([*base, "code_interpreter_call.outputs"]))

    effects["xaiCodeExecInclude"] = xai_code_exec_include

    def openrouter_online(ctx: Ctx) -> None:
        """OpenRouter expresses web search as a `:online` model suffix, not a tool."""
        if _tools_include_builtin(ctx, "web_search"):
            model = ctx.body.get("model")
            if isinstance(model, str) and model and not model.endswith(":online"):
                ctx.body["model"] = f"{model}:online"
        # An empty `tools` is never meaningful to any provider, and it is
        # exactly what a request carrying ONLY an unsupported hosted tool leaves
        # behind. Dropping it unconditionally rather than on the web_search path
        # alone: `builtin_tools=["code_interpreter"]` reached OpenRouter as
        # `tools: []`, measured live 2026-09-04.
        if isinstance(ctx.body.get("tools"), list) and len(ctx.body["tools"]) == 0:
            del ctx.body["tools"]

    effects["openrouterOnline"] = openrouter_online

    def lift_max_tokens_above_budget(ctx: Ctx) -> None:
        """Anthropic requires max_tokens > budget_tokens -- a cross-field rule, so
        it cannot be a field mapping. One of only two effects in four specs."""
        budget = get_path(ctx.body, "thinking.budget_tokens")
        if budget is MISSING:
            return
        if ctx.body["max_tokens"] <= budget:
            ctx.body["max_tokens"] = budget + 1024

    effects["liftMaxTokensAboveBudget"] = lift_max_tokens_above_budget

    return Registry(
        transforms=transforms, builders=builders, predicates=predicates, effects=effects
    )


__all__ = ["AdapterHandles", "make_registry"]
