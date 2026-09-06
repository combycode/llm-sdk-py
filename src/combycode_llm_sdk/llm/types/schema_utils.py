"""Shared JSON Schema utilities for provider-agnostic schema preprocessing.

Transposed from `unified-library-ts/src/llm/types/schema-utils.ts`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

#: `type JsonSchema = Record<string, unknown>` (src/llm/types/tools.ts:58).
JsonSchema = dict[str, Any]


def ensure_additional_properties(schema: JsonSchema) -> JsonSchema:
    """Recursively ensure every object-typed schema has `additionalProperties: false`.

    Required by OpenAI strict mode and Anthropic structured output -- providers
    reject schemas without this explicit flag. Safe across all providers.
    """
    # `{ ...schema }`: spreading `undefined` or `null` in JavaScript yields `{}`
    # rather than throwing, and `openaiResponsesStructuredSchema` reaches here
    # with an absent schema.
    result: dict[str, Any] = dict(schema) if isinstance(schema, Mapping) else {}

    if result.get("type") == "object" and "additionalProperties" not in result:
        result["additionalProperties"] = False

    # `result.properties && typeof result.properties === 'object'`. Written as a
    # type test alone, with NO truthiness: `{}` is truthy in JavaScript, so an
    # empty `properties` is still copied into a fresh object. A Python `if
    # result.get("properties")` would skip that copy and leave the caller's own
    # dict aliased into the result.
    if isinstance(result.get("properties"), Mapping):
        props = dict(result["properties"])
        for key, val in list(props.items()):
            # `val && typeof val === 'object' && !Array.isArray(val)`
            if isinstance(val, Mapping):
                props[key] = ensure_additional_properties(dict(val))
        result["properties"] = props

    # Array items may also be object schemas. A TUPLE form (`items: [...]`) is an
    # array, which `Array.isArray` excludes and `Mapping` excludes too.
    if isinstance(result.get("items"), Mapping):
        result["items"] = ensure_additional_properties(result["items"])

    return result


#: Whose strict-mode rules to judge a schema against.
StrictDialect = Literal["openai", "anthropic"]

#: Keywords Anthropic's strict mode rejects outright, measured against the live API
#: (2026-08-16, claude-haiku-4.5). The list is asymmetric -- `minItems` is accepted
#: while `maxItems` is not, `maxLength` accepted while `maximum` is not -- so treat
#: it as an evolving surface and fall back on the unknown rather than assume
#: support. `exclusiveMaximum` is included by symmetry with the measured
#: `exclusiveMinimum`; it was not itself tested.
_ANTHROPIC_UNSUPPORTED = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "maxItems",
    }
)


@dataclass(frozen=True, slots=True)
class StrictSupport:
    """`{ ok: boolean; reason?: string }`."""

    ok: bool
    reason: str | None = None


def strict_support(schema: JsonSchema | None, dialect: StrictDialect) -> StrictSupport:
    """Can this schema satisfy the provider's strict mode AS WRITTEN?

    Strict mode is worth defaulting to -- it is what makes a provider constrain
    the tool name and arguments DURING generation rather than checking after. But
    the two providers constrain different things, and a schema that violates
    either is rejected with a 400, not quietly degraded:

      openai     every property must appear in `required`, at every nesting level
      anthropic  a set of validation keywords is simply unsupported

    So strict is requested only where it can be honoured. The alternative --
    rewriting the schema to fit, promoting optional properties to
    required-and-nullable -- changes what the tool actually receives, and the
    receiving end is the caller's code.
    """

    def visit(node: Any, path: str) -> str | None:
        # `!node || typeof node !== 'object' || Array.isArray(node)`: only a
        # non-null, non-array object is walked. Deliberately NOT a truthiness
        # test -- an empty `{}` is truthy in JavaScript and gets walked, and
        # `strictSupport({type:'object', properties:{}})` depends on it
        # (strict-schema.test.ts:139).
        if not isinstance(node, Mapping):
            return None
        n: Mapping[str, Any] = node
        at = path or "(root)"

        # A `$ref` points into `$defs`, which this walk does not resolve -- so the
        # properties behind it are never checked and the schema would sail through
        # only to be rejected by the API. Not-verified is treated as not-safe.
        if isinstance(n.get("$ref"), str):
            return f"{at}: '$ref' cannot be verified without resolution"

        if dialect == "anthropic":
            for key in n:
                if key in _ANTHROPIC_UNSUPPORTED:
                    return f"{at}: '{key}' is not supported under strict"

        props = n.get("properties")

        # BOTH providers refuse an open object under strict -- the whole point of
        # strict is that the argument shape is closed. Absent is fine (it gets
        # supplied as false); an explicit `true` survives conforming and is
        # rejected. OpenAI: "'additionalProperties' is required to be supplied and
        # to be false". Anthropic: "For 'object' type, 'additionalProperties:
        # true' is not supported".
        if "additionalProperties" in n and n["additionalProperties"] is not False:
            return f"{at}: 'additionalProperties' must be false under strict"

        # A free-form object -- `{ type: 'object' }` with no `properties` -- is not
        # expressible under OpenAI strict at all: "object schema missing
        # properties". An EMPTY `properties: {}` is fine, so no-argument tools are
        # unaffected. Nested, the API reports this against the PARENT ("Extra
        # required key 'input' supplied"), which sends you looking in the wrong
        # place. Anthropic accepts this shape.
        if dialect == "openai" and n.get("type") == "object" and "properties" not in n:
            return (
                f"{at}: an object schema with no 'properties' cannot be strict "
                f"(a free-form object is not expressible)"
            )

        # `props && typeof props === 'object'`: a type test, not a truthiness one.
        # `{}` is truthy in JavaScript.
        if isinstance(props, Mapping):
            if dialect == "openai":
                required = set(n["required"]) if isinstance(n.get("required"), list) else set()
                missing = [k for k in props if k not in required]
                if len(missing) > 0:
                    return f"{at}: {', '.join(missing)} not listed in 'required'"
            for key, val in props.items():
                r = visit(val, f"{path}.{key}" if path else key)
                if r:
                    return r

        for key in ("items", "additionalProperties"):
            r = visit(n.get(key), f"{path}.{key}" if path else key)
            if r:
                return r
        for key in ("anyOf", "oneOf", "allOf"):
            branches = n.get(key)
            if not isinstance(branches, list):
                continue
            for i, sub in enumerate(branches):
                r = visit(sub, f"{at}.{key}[{i}]")
                if r:
                    return r
        return None

    reason = visit(schema, "")
    return StrictSupport(ok=False, reason=reason) if reason else StrictSupport(ok=True)


__all__ = [
    "JsonSchema",
    "StrictDialect",
    "StrictSupport",
    "ensure_additional_properties",
    "strict_support",
]
