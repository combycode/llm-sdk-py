"""Native OpenAI inline-moderation wire helpers.

build_native_moderation -> the `moderation` request field OpenAI accepts on both
the Responses API and Chat Completions.

parse_native_moderation -> reads the `moderation` field OpenAI returns. Handles
BOTH wire shapes:
  - Responses API:      moderation.{input,output} = moderation_result | error
  - Chat Completions:   moderation.{input,output} = moderation_results | error,
                        where moderation_results wraps `results: [moderation_result]`.

Transposed from `unified-library-ts/src/llm/moderation/native.ts`.

`ModerationResult` / `ModerationEntry` / `ModerationReport` are interfaces in
files this batch does not port, so the object literals TypeScript builds are
built here as plain mappings with the same fields, snake_cased. The keys READ off
the provider's payload (`category_scores`, `category_applied_input_types`) are
wire names and are untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...wire.interpreter import js_string, js_truthy
from .types import MODERATION_DEFAULT_MODEL


def build_native_moderation(
    mod: Mapping[str, Any] | None = None,
    policy: Any = None,
) -> dict[str, Any]:
    """The `moderation` request field for the OpenAI native path.

    `policy` is an OpenAI-only opt-in (via `providerOptions.moderationPolicy`) for
    server-side BLOCKING -- `{ input?: { mode: 'score'|'block' }, output?: {...} }`;
    our unified moderation stays report-only, so it's a passthrough, not a
    first-class knob.
    """
    model = None if mod is None else mod.get("model")
    out: dict[str, Any] = {"model": MODERATION_DEFAULT_MODEL if model is None else model}
    # `policy && typeof policy === 'object'` -- null and primitives are excluded,
    # objects and arrays are not.
    if js_truthy(policy) and isinstance(policy, (Mapping, list)):
        out["policy"] = policy
    return out


def parse_native_moderation(raw: Any) -> dict[str, Any] | None:
    """Parse OpenAI's returned `moderation` object into a unified report, or None
    when the server returned nothing usable."""
    if not js_truthy(raw) or not isinstance(raw, Mapping):
        return None
    m: Mapping[str, Any] = raw
    entry_in = _parse_entry(m.get("input"))
    entry_out = _parse_entry(m.get("output"))
    if entry_in is None and entry_out is None:
        return None
    report: dict[str, Any] = {"source": "native"}
    if entry_in is not None:
        report["input"] = entry_in
    if entry_out is not None:
        report["output"] = entry_out
    return report


def _parse_entry(entry: Any) -> dict[str, Any] | None:
    if not js_truthy(entry) or not isinstance(entry, Mapping):
        return None
    e: Mapping[str, Any] = entry
    if e.get("type") == "error":
        message = e.get("message")
        if message is None:
            message = e.get("code")
        if message is None:
            message = "moderation error"
        return {"error": js_string(message)}
    # Chat Completions wraps the result(s) in a `moderation_results` envelope.
    if e.get("type") == "moderation_results" and isinstance(e.get("results"), list):
        results = e["results"]
        first = results[0] if results else None
        return _to_result(first) if isinstance(first, Mapping) else None
    # Responses API returns the `moderation_result` directly.
    if e.get("type") == "moderation_result" or js_truthy(e.get("categories")):
        return _to_result(e)
    return None


def _to_result(r: Mapping[str, Any]) -> dict[str, Any]:
    categories = r.get("categories")
    scores = r.get("category_scores")
    out: dict[str, Any] = {
        # camelCase: this is the WIRE-level unified shape the shared specs name
        # and the recorded corpus froze, not the snake_case public API. The
        # SOURCE keys stay snake_case because they are OpenAI's own field names.
        "flagged": js_truthy(r.get("flagged")),
        "categories": {} if categories is None else categories,
        "categoryScores": {} if scores is None else scores,
    }
    # The TypeScript literal always carries the key, but with `undefined` when
    # the provider omitted it (omni models only) -- and `undefined` disappears on
    # JSON round-trip. A Python None would survive as an explicit null, which is
    # a different response, so the key is omitted instead.
    applied = r.get("category_applied_input_types")
    if applied is not None:
        out["categoryAppliedInputTypes"] = applied
    return out


__all__ = ["build_native_moderation", "parse_native_moderation"]
