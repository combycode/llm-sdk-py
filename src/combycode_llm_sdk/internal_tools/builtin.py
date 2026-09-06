"""The five tools every LLM app eventually needs, written as prompts.

Transposed from `unified-library-ts/src/plugins/internal-tools/builtin/`.

    summarize  compaction, and the one ContextGuard reaches for
    classify   routing, intent, moderation
    structure  typed data out of prose
    score      self-evaluation, judge-LLM patterns, ranking
    clarify    interactive flows -- what to ask when the prompt is short

They are data, not code: an id carrying its version, two schemas, a system
prompt and a user template. That is what lets a deployment read the prompt off a
registered tool, pin a version, or swap the wording for one model without
touching a caller -- and what lets any of them be replaced by a real
implementation later, since the runner cannot tell a prompt-backed tool from a
Python one.

Domain-heavy tools (fact-extract, prompt-improve, title, ...) ship separately
under `extensions/` and are registered by hand. `RunnerContextTools` treats a
missing `orxa:fact-extract@1.0.0` as "no facts" for exactly that reason.
"""

from __future__ import annotations

import json
import math
from typing import Any

from .define import define_llm_tool
from .template import format_bulleted_list, format_numbered_list
from .types import InternalTool, LLMToolDefinition, ModelPreference, PromptVariant

#: What these tools ask for. Small and cheap on purpose: a compaction primitive
#: that costs as much as the conversation it compacts is not a saving, and every
#: one of these runs on the hot path of something else.
_FAST = ModelPreference(
    preferred_model="openai/gpt-5.4-nano",
    fallback_models=(
        "google/gemini-3.1-flash-lite-preview",
        "anthropic/claude-haiku-4-5",
    ),
    max_tokens=300,
    temperature=0.0,
)


def _fast(*, max_tokens: int, temperature: float = 0.0) -> ModelPreference:
    return ModelPreference(
        preferred_model=_FAST.preferred_model,
        fallback_models=_FAST.fallback_models,
        max_tokens=max_tokens,
        temperature=temperature,
    )


# -- summarize ---------------------------------------------------------------

_SUMMARIZE_EXAMPLE = {
    "summary": (
        "MIT researchers built a lithium-sulfur battery charging in 5 minutes, "
        "lasting 1000+ cycles."
    ),
    "keyPoints": [
        "5-minute charge time",
        "1000+ cycle lifespan",
        "Lithium-sulfur chemistry",
    ],
}

_SUMMARIZE_USER = """Summarize the following content.

{{lengthLine}}
{{focusLine}}

Content:
{{content}}"""

_STRICT_SYSTEM = """You are a summarization tool. Follow these rules strictly:

1. OUTPUT OBJECT: Return exactly one JSON object with exactly two fields: "summary" (string) and "keyPoints" (array of strings). NO markdown, NO prose around the JSON.
2. FACT PRESERVATION: preserve every date, name, path, number, unit, quoted term, and ID EXACTLY as written in the source. Do NOT normalize, round, paraphrase, or substitute them.
3. NO INFERENCE: "summary" and "keyPoints" items MUST contain only claims explicitly in the source. Do NOT strengthen, broaden, soften, or generalize.
4. FOCUS SCOPING: if a focus is provided, BOTH "summary" AND every item in "keyPoints" MUST cover ONLY facts directly relevant to that focus. Exclude unrelated facts entirely.
5. COVERAGE (no focus): "summary" captures the main result plus the most decision-relevant qualifier or limitation. "keyPoints" covers the main result plus supporting facts and any explicit caveat, constraint, or blocker stated in the source.
6. LENGTH: "summary" MUST stay under the provided limit. Treat the limit as HARD -- stop short rather than overshoot. Target 10-15% under the limit as safety margin.
7. "keyPoints" ITEMS: each is a short string with one fact or a tightly-related cluster. Copy exact wording where possible.
8. NO INVENTION: if the source lacks detail, return a smaller "keyPoints" array. NEVER invent facts.
9. NEGATIVE EXAMPLES: do NOT rewrite "over 1000 cycles" as "1000+ cycles". Do NOT rewrite "Scientists at MIT" as "MIT scientists". Do NOT omit an explicit limitation when it's a main qualifier."""

_BALANCED_SYSTEM = """You are a summarizer. Follow these rules:

1. OUTPUT: Return exactly one JSON object with "summary" (string) and "keyPoints" (array of strings). No markdown.
2. PRESERVE KEY FACTS: keep exact dates, names, paths, numbers, units, and IDs verbatim. You may paraphrase surrounding narrative.
3. FOCUS SCOPING (HARD BINDING RULE): when a focus is provided, treat it as an allow-list. ONLY facts that are literally about the focus keywords may appear in "summary" or "keyPoints". Everything else -- error codes, status codes, pagination, metrics, ports, unrelated features -- MUST be EXCLUDED, even if the source mentions them.
   Procedure for "keyPoints":
     (a) List every candidate fact from the source.
     (b) For each, ask: does this fact DIRECTLY describe {focus}? If it merely COEXISTS in the source, EXCLUDE it.
     (c) Keep only the facts that pass (b). A 2-item array is correct if only 2 facts match the focus.
   A keyPoints item that mentions ANYTHING outside the focus is a RULE VIOLATION -- drop it entirely rather than trim it.
4. COVERAGE (no focus only): capture the main result and include any important explicit caveat or limitation from the source.
5. LENGTH: stay at or under the requested limit -- treat it as a ceiling, not a target.
6. NO INVENTION: every fact must come from the source."""

_STRICT_CHECKS = """

Final checks before answering:
1. "summary" stays under the limit (target 10-15% under).
2. Every included fact with a date, name, path, number, unit, or ID matches the source exactly.
3. "keyPoints" contains only supported facts.
4. If the source includes an important explicit limitation or blocker, it's included (unless excluded by focus)."""

#: How much room a summary needs beyond the characters it may occupy. The
#: multiplier covers JSON framing and the keyPoints array; the constant covers
#: models that spend output budget on reasoning before writing anything.
_SUMMARY_TOKEN_SLACK = 1.6
_SUMMARY_TOKEN_FLOOR = 400
_SUMMARY_DEFAULT_TOKENS = 800


def _summarize_prepare(value: dict[str, Any]) -> dict[str, Any]:
    max_length = value.get("maxLength")
    max_tokens = value.get("maxTokens")
    if max_length:
        length_line = f"Length target: keep the summary under {max_length} characters."
    elif max_tokens:
        length_line = f"Length target: keep the summary under {max_tokens} tokens."
    else:
        length_line = "Length target: concise (one or two sentences)."
    focus = value.get("focus")
    return {**value, "lengthLine": length_line, "focusLine": f"Focus: {focus}" if focus else ""}


def _summarize_max_tokens(value: Any, ctx: Any) -> int:
    """Size the answer from the requested length, not from a fixed ceiling.

    A fixed number either truncates the long cases or overpays for the short
    ones, and this tool is called with limits spanning two orders of magnitude.
    """
    max_tokens = value.get("maxTokens")
    if isinstance(max_tokens, (int, float)) and max_tokens > 0:
        return int(max_tokens)
    max_length = value.get("maxLength")
    if isinstance(max_length, (int, float)) and max_length > 0:
        summary_tokens = int(
            ctx.counter.estimate_message(
                "x" * int(max_length), {"provider": ctx.provider, "model": ctx.model}
            )
        )
        return math.ceil(summary_tokens * _SUMMARY_TOKEN_SLACK) + _SUMMARY_TOKEN_FLOOR
    return _SUMMARY_DEFAULT_TOKENS


summarize_tool: InternalTool = define_llm_tool(
    LLMToolDefinition(
        id="orxa:summarize@1.0.0",
        namespace="orxa",
        name="summarize",
        version="1.0.0",
        description=(
            "Summarize long content while preserving key facts verbatim. maxLength "
            "(chars) or maxTokens (tokens) constrains size."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Text to summarize"},
                "maxLength": {
                    "type": "integer",
                    "minimum": 20,
                    "description": "Max characters for summary text",
                },
                "maxTokens": {
                    "type": "integer",
                    "minimum": 20,
                    "description": "Max tokens for the response (takes precedence over maxLength)",
                },
                "focus": {"type": "string", "description": "Optional focus area", "default": ""},
            },
            "required": ["content"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "keyPoints": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary", "keyPoints"],
        },
        output_format="json",
        # Two prompts, because strict fact-preservation helps the models that
        # over-paraphrase and hurts the ones that go too literal to compress.
        variants=(
            PromptVariant(
                id="strict",
                description=(
                    "Very strict fact preservation -- for models that over-paraphrase "
                    "(Haiku, Flash-Lite)."
                ),
                supported_providers=("anthropic", "google"),
                system_prompt=_STRICT_SYSTEM,
                user_template=_SUMMARIZE_USER + _STRICT_CHECKS,
                output_example=_SUMMARIZE_EXAMPLE,
            ),
            PromptVariant(
                id="balanced",
                description=(
                    "Balanced compression -- default for OpenAI and other providers "
                    "that summarize well."
                ),
                is_default=True,
                system_prompt=_BALANCED_SYSTEM,
                user_template=_SUMMARIZE_USER,
                output_example=_SUMMARIZE_EXAMPLE,
            ),
        ),
        prepare_input=_summarize_prepare,
        resolve_max_tokens=_summarize_max_tokens,
        model_preference=_fast(max_tokens=800, temperature=0.3),
        recommended_threshold=95,
        tags=("summarize", "compaction", "text"),
    )
)


# -- classify ----------------------------------------------------------------

classify_tool: InternalTool = define_llm_tool(
    LLMToolDefinition(
        id="orxa:classify@1.0.0",
        namespace="orxa",
        name="classify",
        version="1.0.0",
        description="Select the most relevant suggestion from a list, with confidence score",
        input_schema={
            "type": "object",
            "properties": {
                "request": {"type": "string"},
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": ["request", "suggestions"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "selectedIndex": {"type": "integer", "minimum": 0},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reasoning": {"type": "string"},
            },
            "required": ["selectedIndex", "confidence", "reasoning"],
        },
        system_prompt=(
            "You are a deterministic classifier. Given a request and a list of "
            "suggestions, pick the single best match."
        ),
        user_template="""Given the request: "{{request}}"

Select the most relevant option from the following:
{{suggestionsList}}

Return a JSON object with:
- selectedIndex: the 0-based index of the best match
- confidence: a number from 0 to 1 indicating confidence
- reasoning: a brief explanation of why this option was selected""",
        output_format="json",
        # Numbered from ZERO, matching the index the model is asked to return.
        # Numbering from one and expecting a zero-based answer is an off-by-one
        # the model cannot see and the caller cannot debug.
        prepare_input=lambda value: {
            **value,
            "suggestionsList": format_numbered_list(value.get("suggestions") or [], 0),
        },
        model_preference=_fast(max_tokens=300),
        tags=("classify", "routing", "fast"),
    )
)


# -- structure ---------------------------------------------------------------

structure_tool: InternalTool = define_llm_tool(
    LLMToolDefinition(
        id="orxa:structure@1.0.0",
        namespace="orxa",
        name="structure",
        version="1.0.0",
        description="Transform unstructured request into structured JSON matching a given schema",
        input_schema={
            "type": "object",
            "properties": {
                "request": {"type": "string", "description": "Unstructured text to parse"},
                "schema": {
                    "type": "object",
                    "description": "Target JSON Schema the output must match",
                },
            },
            "required": ["request", "schema"],
        },
        system_prompt=(
            "You extract structured data from text. Your output must be valid JSON "
            "matching the target schema exactly -- no extra fields, no missing "
            "required fields."
        ),
        user_template="""Parse the following request into structured data matching the given schema.

Request: {{request}}

Target JSON Schema:
{{schema}}

Return a valid JSON object matching the schema exactly.""",
        output_format="json",
        # INDENTED, which is the only thing this adds: the renderer already
        # json-encodes a non-string, so the choice is between one dense line and
        # something a model can read structure off. A nested schema on one line
        # is where the "no extra fields, no missing required fields" instruction
        # starts getting ignored.
        prepare_input=lambda value: {
            **value,
            "schema": json.dumps(value.get("schema"), indent=2),
        },
        model_preference=_fast(max_tokens=600),
        tags=("structure", "json", "parsing", "extraction"),
    )
)


# -- score -------------------------------------------------------------------

score_tool: InternalTool = define_llm_tool(
    LLMToolDefinition(
        id="orxa:score@1.0.0",
        namespace="orxa",
        name="score",
        version="1.0.0",
        description=(
            "Score how well an answer fulfills a task (0-100 overall, per-criterion breakdown)"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task description the answer is meant to fulfill",
                },
                "answer": {"type": "string", "description": "The answer to evaluate"},
                "criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Evaluation criteria (e.g. accuracy, completeness, clarity)",
                    "default": ["accuracy", "completeness", "clarity"],
                },
            },
            "required": ["task", "answer"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 100},
                "breakdown": {"type": "object"},
                "feedback": {"type": "string"},
            },
            "required": ["score", "breakdown", "feedback"],
        },
        system_prompt=(
            "You are an impartial evaluator. Score answers strictly based on the given "
            "criteria. The overall score must equal the average of breakdown scores."
        ),
        user_template="""Score how well the following answer fulfills the task.

Task:
{{task}}

Answer:
{{answer}}

Evaluation Criteria:
{{criteriaList}}

Return a JSON object with:
- score: overall score from 0 to 100 (must equal the average of breakdown values)
- breakdown: object with a 0-100 score for each criterion (keys match criterion names)
- feedback: concise, constructive feedback on the answer""",
        output_format="json",
        prepare_input=lambda value: {
            **value,
            "criteriaList": format_bulleted_list(value.get("criteria") or []),
        },
        # A generous ceiling, unlike its siblings: a reasoning model spends
        # output budget on thought before it writes the breakdown, and a
        # truncated answer here reads as a low score rather than as a failure.
        model_preference=_fast(max_tokens=2000),
        tags=("score", "evaluation", "benchmark"),
    )
)


# -- clarify -----------------------------------------------------------------

clarify_tool: InternalTool = define_llm_tool(
    LLMToolDefinition(
        id="orxa:clarify@1.0.0",
        namespace="orxa",
        name="clarify",
        version="1.0.0",
        description=(
            "Validate whether a prompt meets given requirements; produce clarifying "
            "questions for gaps"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "requirements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": ["prompt", "requirements"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "satisfied": {"type": "boolean"},
                "missingRequirements": {"type": "array", "items": {"type": "string"}},
                "clarificationQuestions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["satisfied", "missingRequirements", "clarificationQuestions"],
        },
        system_prompt=(
            "You validate prompts against requirements. Be strict -- only mark "
            "satisfied=true if EVERY requirement is addressed in the prompt."
        ),
        user_template="""Check if the following prompt satisfies all the listed requirements.

Prompt:
{{prompt}}

Requirements:
{{requirementsList}}

Return a JSON object with:
- satisfied: boolean indicating if all requirements are met
- missingRequirements: array of requirements that are not satisfied (empty if all met)
- clarificationQuestions: array of questions to ask to gather missing information""",
        output_format="json",
        prepare_input=lambda value: {
            **value,
            "requirementsList": format_bulleted_list(value.get("requirements") or []),
        },
        model_preference=_fast(max_tokens=500, temperature=0.2),
        tags=("clarify", "validation", "requirements"),
    )
)


#: The core set, in the order the TypeScript registers them.
BUILTIN_TOOLS: tuple[InternalTool, ...] = (
    summarize_tool,
    classify_tool,
    score_tool,
    structure_tool,
    clarify_tool,
)


def register_builtin_tools(backend: Any) -> None:
    """Put the core set on a backend."""
    for tool in BUILTIN_TOOLS:
        backend.register(tool)


__all__ = [
    "BUILTIN_TOOLS",
    "clarify_tool",
    "classify_tool",
    "register_builtin_tools",
    "score_tool",
    "structure_tool",
    "summarize_tool",
]
