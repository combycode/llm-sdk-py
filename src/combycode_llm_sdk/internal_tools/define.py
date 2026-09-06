"""A declaration in, an executable tool out.

The six steps a prompt-backed tool would otherwise restate around its one
paragraph of instructions: apply the schema's defaults, pick a variant, render,
size the answer, call, and read the JSON back. Written once here, the paragraph
that mattered is the largest thing in the tool's file.

Transposed from `unified-library-ts/src/plugins/internal-tools/runner/define.ts`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .json_format import JSON_FORMAT, compose_json_system_prompt
from .template import parse_json_with_fences, render_template
from .types import (
    InternalTool,
    InternalToolContext,
    InternalToolError,
    JsonSchema,
    LLMToolDefinition,
    PromptVariant,
    ResolveMaxTokensContext,
    select_variant,
)


def _attach_structure_guidance(
    base_system: str,
    output_format: str,
    output_schema: JsonSchema | None,
    output_example: Any,
) -> str:
    """Show the model the shape, rather than only checking it afterwards.

    A schema the model never saw is a test it was never told it was sitting.
    Both halves are worth having -- the answer is still checked -- but only one
    of them changes what comes back.
    """
    if output_format != JSON_FORMAT:
        return base_system

    parts = [base_system]
    if output_schema is not None:
        parts += [
            "\n## Output schema (your JSON must match)",
            "```json",
            json.dumps(dict(output_schema), indent=2),
            "```",
        ]
    if output_example is not None:
        parts += [
            "\n## Output example (copy this shape exactly)",
            "```json",
            json.dumps(output_example, indent=2),
            "```",
        ]
    return "\n".join(parts)


def _single_variant(definition: LLMToolDefinition) -> PromptVariant:
    """A definition with no variants still goes through the variant path."""
    return PromptVariant(
        id="default",
        system_prompt=definition.system_prompt,
        user_template=definition.user_template,
        output_example=definition.output_example,
        output_schema=definition.output_schema,
        resolve_max_tokens=definition.resolve_max_tokens,
        is_default=True,
    )


def apply_schema_defaults(value: Any, schema: JsonSchema | None) -> dict[str, Any]:
    """Fill in what the schema says a caller may leave out.

    Applied BEFORE rendering, so `{{audience}}` with a declared default is never
    a hole in the prompt. A default that only reached the model as an absent
    variable would be a default in name only.
    """
    base: dict[str, Any] = dict(value) if isinstance(value, Mapping) else {}
    if not schema or schema.get("type") != "object":
        return base
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return base
    for key, prop in properties.items():
        if key in base or not isinstance(prop, Mapping):
            continue
        if "default" in prop:
            base[key] = prop["default"]
    return base


def define_llm_tool(definition: LLMToolDefinition) -> InternalTool:
    """The declaration, as something the runner can call."""

    def execute(value: Any, ctx: InternalToolContext) -> Any:
        client = ctx.client
        model_id = ctx.model_id
        if client is None or not model_id:
            raise InternalToolError(
                f"tool {definition.id} needs an LLM client and a model in its context "
                "(the runner supplies both; this tool was called by something else)",
                tool_id=definition.id,
            )

        provider, _, model_name = model_id.partition("/")
        if not model_name:
            provider, model_name = model_id, model_id

        variables = apply_schema_defaults(value, definition.input_schema)
        if definition.prepare_input is not None:
            variables = definition.prepare_input(variables)

        variant = select_variant(
            definition.variants or (_single_variant(definition),),
            provider=provider,
            model=model_name,
            mode=variables.get("mode"),
        )

        schema = variant.output_schema or definition.output_schema
        example = variant.output_example if variant.output_example is not None else (
            definition.output_example
        )
        resolver = variant.resolve_max_tokens or definition.resolve_max_tokens

        rendered = render_template(variant.system_prompt, variables)
        with_structure = _attach_structure_guidance(
            rendered, definition.output_format, schema, example
        )
        system = (
            compose_json_system_prompt(with_structure)
            if definition.output_format == JSON_FORMAT
            else with_structure
        )
        user = render_template(variant.user_template, variables)

        max_tokens = definition.model_preference.max_tokens
        if resolver is not None:
            if ctx.counter is None:
                raise InternalToolError(
                    f"tool {definition.id}: resolve_max_tokens needs a token counter "
                    "in its context",
                    tool_id=definition.id,
                )
            max_tokens = resolver(
                variables,
                ResolveMaxTokensContext(
                    provider=provider, model=model_name, counter=ctx.counter
                ),
            )

        answer = client.complete(
            user,
            system=system,
            max_tokens=max_tokens,
            temperature=definition.model_preference.temperature,
        )
        if ctx.record_completion is not None:
            ctx.record_completion(answer)

        if definition.output_format != JSON_FORMAT:
            return answer.text

        try:
            return parse_json_with_fences(answer.text)
        except ValueError as exc:
            # A truncated answer and a chatty one both arrive here as "not
            # JSON", and they are opposite problems: one wants a bigger budget,
            # the other a better prompt. Live, a reasoning model spent the
            # tool's whole 200-token allowance on thinking and returned
            # `{"label": "Negative", ` -- which read as a prompt failure and is
            # not one, and would have been re-run identically on every
            # fallback model in the chain.
            cause = (
                f"the answer was cut off at max_tokens={max_tokens} "
                "(a reasoning model spends this budget before it writes)"
                if answer.finish_reason == "length"
                else str(exc)
            )
            raise InternalToolError(
                f"tool {definition.id} produced non-JSON output: {cause}. "
                f"Raw (first 500 chars): {answer.text[:500]}",
                tool_id=definition.id,
            ) from exc

    return InternalTool(
        id=definition.id,
        namespace=definition.namespace,
        name=definition.name,
        version=definition.version,
        description=definition.description,
        input_schema=definition.input_schema,
        output_schema=definition.output_schema,
        execute=execute,
        model_preference=definition.model_preference,
        recommended_threshold=definition.recommended_threshold,
        tags=tuple(definition.tags),
        signature=definition.signature,
        definition=definition,
    )


__all__ = ["apply_schema_defaults", "define_llm_tool"]
