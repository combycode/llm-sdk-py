"""What a tool is, what it prefers to run on, and what it is handed at run time.

The split that matters here is between `LLMToolDefinition` -- the DECLARATION, a
plain value a bench harness can read, diff and re-render without paying to run
it -- and `InternalTool`, the executable the runner resolves and calls. One
becomes the other through `define_llm_tool`, and the declaration survives on
`tool.definition` rather than being consumed by the conversion.

Transposed from `unified-library-ts/src/plugins/internal-tools/types.ts` and
`runner/types.ts`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

JsonSchema = Mapping[str, Any]


class InternalToolError(RuntimeError):
    """A tool could not be resolved, could not be given its input, or failed.

    One class rather than four, with `tool_id` on every instance: a caller
    catching this is deciding whether to fall back, and that decision is the
    same for a tool that is missing and a tool that refused its arguments.
    """

    def __init__(self, message: str, *, tool_id: str | None = None) -> None:
        super().__init__(message)
        self.tool_id = tool_id


@dataclass(frozen=True)
class ModelPreference:
    """Which models a tool wants, and how much of one it may spend.

    A preference, not a binding: the runner walks preferred, then fallbacks, and
    a benchmark-derived `compat` list overrides both. The tool says what it was
    written against; the deployment says what it can afford.
    """

    preferred_model: str | None = None
    fallback_models: Sequence[str] = ()
    max_tokens: int | None = None
    temperature: float | None = None


@dataclass
class InternalToolContext:
    """What the runner hands a tool for the duration of one call.

    `client` and `model_id` are populated only for the LLM path; a tool that
    needs them and finds them absent has been run by something that does not
    know how to run it, which is worth a clear error rather than an
    `AttributeError` three frames down.
    """

    hooks: Any = None
    #: An `LLM` pinned to the chosen model, built and pooled by the runner.
    client: Any = None
    #: `"provider/model"` for that client.
    model_id: str | None = None
    tool_id: str | None = None
    #: A token counter, for definitions that size their own output.
    counter: Any = None
    #: Called by the tool after each completion, so the runner can capture
    #: usage it never sees directly.
    record_completion: Callable[[Any], None] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptVariant:
    """One tool, one of its prompts.

    A model that reliably follows a terser instruction, or a mode that wants a
    different schema, is a variant -- not a second tool with a copied id and a
    diverging description.
    """

    id: str
    system_prompt: str = ""
    user_template: str = ""
    description: str | None = None
    output_example: Any = None
    output_schema: JsonSchema | None = None
    resolve_max_tokens: Callable[[Mapping[str, Any], Any], int] | None = None
    modes: Sequence[str] = ()
    supported_providers: Sequence[str] = ()
    supported_models: Sequence[str] = ()
    is_default: bool = False


@dataclass(frozen=True)
class ResolveMaxTokensContext:
    """What a definition's `resolve_max_tokens` gets to decide with."""

    provider: str
    model: str
    counter: Any


@dataclass(frozen=True)
class LLMToolDefinition:
    """A tool whose implementation is a prompt and a schema.

    Every field is data. That is the point: the prompt can be read off a
    registered tool, versioned in the id, and swapped per model without
    touching code.
    """

    id: str
    namespace: str
    name: str
    version: str
    description: str
    input_schema: JsonSchema
    model_preference: ModelPreference = field(default_factory=ModelPreference)
    output_schema: JsonSchema | None = None
    system_prompt: str = ""
    user_template: str = ""
    #: `"json"` (parse the answer and show the model the schema) or `"text"`.
    output_format: str = "text"
    #: A last chance to derive template variables the caller did not pass.
    prepare_input: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    #: Size the answer from the input, when a fixed `max_tokens` would either
    #: truncate the long cases or overpay for the short ones.
    resolve_max_tokens: Callable[[Mapping[str, Any], ResolveMaxTokensContext], int] | None = None
    output_example: Any = None
    variants: Sequence[PromptVariant] = ()
    recommended_threshold: float | None = None
    tags: Sequence[str] = ()
    signature: str | None = None


@dataclass
class InternalTool:
    """An executable tool, however it was written.

    `define_llm_tool` produces one whose body is a model call; a plain Python
    function wrapped by hand produces one whose body is code. The runner cannot
    tell them apart, which is what lets a prompt-backed tool be replaced by a
    real implementation without touching a caller.
    """

    id: str
    namespace: str
    name: str
    version: str
    description: str
    input_schema: JsonSchema
    execute: Callable[[Any, InternalToolContext], Any]
    output_schema: JsonSchema | None = None
    model_preference: ModelPreference | None = None
    #: Minimum average benchmark score for a model to enter this tool's
    #: recommended chain.
    recommended_threshold: float | None = None
    signature: str | None = None
    signed_by: str | None = None
    tags: Sequence[str] = ()
    #: The declaration this tool was built from, when it came from one. Kept so
    #: a harness can re-render a prompt without running it.
    definition: LLMToolDefinition | None = None


class ToolBackend(Protocol):
    """Where tools come from: this repo, a directory, a remote index."""

    @property
    def name(self) -> str: ...

    def list(self) -> Sequence[InternalTool]: ...

    def get(self, tool_id: str) -> InternalTool | None: ...


@dataclass(frozen=True)
class ModelFilter:
    """Restrict a search to models a tool is known to work on."""

    provider: str
    model: str
    min_score: float | None = None


@dataclass(frozen=True)
class ToolFilter:
    """Which tools to consider."""

    namespace: str | None = None
    prefix: str | None = None
    tag: str | None = None
    model: ModelFilter | None = None


@dataclass(frozen=True)
class ToolCompat:
    """Models that scored well enough on a tool, cheapest first."""

    recommended: Sequence[str] = ()


#: Benchmark results keyed by tool id. The runner ships without a bench
#: subsystem, so this is declared rather than produced here: a deployment that
#: has one passes its results in.
CompatFile = Mapping[str, ToolCompat]


def select_variant(
    variants: Sequence[PromptVariant],
    *,
    provider: str,
    model: str,
    mode: str | None = None,
) -> PromptVariant:
    """The best-matching variant. Most specific wins.

    Mode narrows the field first, then model, then provider, then the default
    within whatever field is left. A tool with one variant always gets it.
    """
    if not variants:
        raise ValueError("select_variant: no variants to choose from")

    full_id = f"{provider}/{model}"
    by_mode = [v for v in variants if mode and mode in tuple(v.modes)]
    candidates = by_mode or list(variants)

    for variant in candidates:
        if full_id in tuple(variant.supported_models):
            return variant
    for variant in candidates:
        if provider in tuple(variant.supported_providers):
            return variant
    for variant in candidates:
        if variant.is_default:
            return variant
    if by_mode:
        return by_mode[0]
    for variant in variants:
        if variant.is_default:
            return variant

    raise ValueError(
        "select_variant: no match and no default. "
        f"mode={mode!r}, provider={provider!r}, model={model!r}, "
        f"variants=[{', '.join(v.id for v in variants)}]"
    )


__all__ = [
    "CompatFile",
    "InternalTool",
    "InternalToolContext",
    "InternalToolError",
    "JsonSchema",
    "LLMToolDefinition",
    "ModelFilter",
    "ModelPreference",
    "PromptVariant",
    "ResolveMaxTokensContext",
    "ToolBackend",
    "ToolCompat",
    "ToolFilter",
    "select_variant",
]
