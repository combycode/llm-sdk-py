"""`complete` / `acomplete` -- one call, no client to hold.

Transposed from `unified-library-ts/src/helpers/one-shot.ts`.

    result = complete(
        model="anthropic/claude-haiku-4.5",
        api_key=key,
        system="Reply in one sentence.",
        prompt="What is Python?",
        max_tokens=60,
    )
    result.text

The helper builds a client, makes the call and destroys the client before
returning, so a caller who wanted one answer does not leak one. `acomplete` is
the same over `AsyncLLM` -- the `a`-prefix the API contract specifies for
module-level functions, and neither wraps the other.

`max_cost_usd=` gates the call BEFORE it is sent: the estimator prices the
request and the call is refused if the chosen bound is over the limit, so the
provider is never reached. Off unless asked for -- a guard that ran by default
would refuse work nobody had budgeted.

The bound is the interesting choice. `expected` by default, because judging
every call by `high` refuses affordable work and judging by `low` is not a
budget at all; `budget_bound="high"` is there for a caller who would rather
refuse a call that MIGHT be expensive than pay for one that turned out to be.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..llm.output_errors import InvalidFinalOutputError
from ..results import Completion
from .agent_loop import DEFAULT_MAX_STEPS, arun_tools, run_tools
from .client_resolver import is_namespaced_model_id, parse_model_id, parse_model_tier
from .content import load_content
from .json_schema import SchemaError
from .llm import LLM, AsyncLLM
from .structured import is_schema_class, parse_into, to_wire

#: Everything `complete` forwards to the client, by its public name.
_FORWARDED = (
    "system",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "structured",
    "audio",
    "output_modalities",
    "reasoning",
    "cache",
    "service_tier",
    "provider_options",
    "builtin_tools",
    "state",
    "stateful",
    "timeout",
)


#: Which estimate bound `max_cost_usd` is compared against when the caller
#: does not say. See the module docstring for why it is not `high`.
DEFAULT_BUDGET_BOUND = "expected"


def _enforce_budget(kwargs: Mapping[str, Any], input_: Any) -> None:
    """Price the call and refuse it rather than send it.

    Runs only when the caller set a limit. The estimate is the SAME one
    `estimate_cost()` returns, so a caller who checked the price themselves and
    a caller who set a limit are judged by one number rather than two.

    Every guess behind that number rides along on the error: an estimate is not
    a measurement, and "over budget" without the assumptions is a refusal
    nobody can argue with or act on.
    """
    limit = kwargs.get("max_cost_usd")
    if limit is None:
        return

    from ..estimate import BudgetExceededError, estimate_cost

    bound = kwargs.get("budget_bound") or DEFAULT_BUDGET_BOUND
    if bound not in ("low", "expected", "high"):
        raise ValueError(
            f"complete: budget_bound must be low, expected or high -- got {bound!r}"
        )

    estimate = estimate_cost(
        model=parse_model_tier(kwargs["model"])["modelId"],
        provider=kwargs.get("provider"),
        prompt=input_,
        output_tokens=kwargs.get("max_tokens"),
        engine=kwargs.get("engine"),
    )
    cost = float(getattr(estimate, bound))
    if cost > float(limit):
        raise BudgetExceededError(
            estimate=estimate,
            cost_usd=cost,
            limit_usd=float(limit),
            # Nothing is spent: this is a per-call ceiling, not a running
            # ledger. `Budget` is the object that keeps a running total.
            spent_usd=0.0,
            bound=bound,
        )


def _provider_of(model: str, provider: str | None) -> str | None:
    if provider:
        return provider
    return parse_model_id(model)[0] if is_namespaced_model_id(model) else None


def _has_audio(input_: Any) -> bool:
    """Whether the built input carries an audio part."""
    if isinstance(input_, str) or not isinstance(input_, list) or not input_:
        return False
    if isinstance(input_[0], Mapping) and "role" in input_[0]:
        return any(
            isinstance(m.get("content"), list)
            and any(p.get("type") == "audio" for p in m["content"])
            for m in input_
        )
    return any(p.get("type") == "audio" for p in input_)


def _resolve_attachments(
    raw: Sequence[Any] | None,
) -> list[dict[str, Any]] | None:
    """Paths, URLs and bytes become parts; a part passes through untouched."""
    if not raw:
        return None
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, (str, Path, bytes, bytearray)):
            out.append(load_content(bytes(item) if isinstance(item, bytearray) else item))
        else:
            out.append(dict(item))
    return out


def _prompt_of(kwargs: Mapping[str, Any]) -> Any:
    """The input, under either name the contract uses.

    `prompt=` is the TypeScript's single name and takes any of the three shapes;
    `messages=` is the Python addition, and it is what a caller writes when they
    have a transcript -- `03_multi_turn.py` uses it. Reading only `prompt` left
    `messages=` unread and sent an EMPTY text block, which Anthropic rejects with
    `text content blocks must be non-empty`.
    """
    for name in ("prompt", "messages", "input"):
        value = kwargs.get(name)
        if value is not None and value != "":
            return value
    raise TypeError(
        "complete: nothing to send. Pass prompt= (a string, content parts, or "
        "messages) or messages=."
    )


def _build_input(prompt: Any, attachments: list[dict[str, Any]] | None) -> Any:
    """Fold the attachments into the prompt, whatever shape the prompt is.

    Attachments go BEFORE the text in every case: models attend to an image or a
    document better when the instruction about it follows, and the corpus
    scenarios are written that way.
    """
    if not attachments:
        return prompt

    if isinstance(prompt, str):
        return [{"role": "user", "content": [*attachments, {"type": "text", "text": prompt}]}]

    if isinstance(prompt, list) and prompt and not (
        isinstance(prompt[0], Mapping) and "role" in prompt[0]
    ):
        return [*attachments, *prompt]

    messages = list(prompt or [])
    first_user = next((i for i, m in enumerate(messages) if m.get("role") == "user"), -1)
    if first_user == -1:
        return [*messages, {"role": "user", "content": attachments}]

    target = messages[first_user]
    content = target.get("content")
    merged = (
        [*attachments, {"type": "text", "text": content}]
        if isinstance(content, str)
        else [*attachments, *(content or [])]
    )
    messages[first_user] = {**target, "content": merged}
    return messages


def _plan(kwargs: dict[str, Any]) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Everything both `complete` and `acomplete` decide before the call.

    Returns `(client_config, input, options)`. Shared so the two entry points
    cannot drift about which api an audio prompt routes to or which tier a
    `model:tier` suffix means -- the TypeScript's tools/no-tools branches DID
    drift that way, silently dropping four options on one side.
    """
    model = kwargs["model"]
    provider = kwargs.get("provider")
    input_ = _build_input(
        _prompt_of(kwargs), _resolve_attachments(kwargs.get("attachments"))
    )
    # After the input is built and before anything is sent: the estimate prices
    # what would actually go out, attachments included.
    _enforce_budget(kwargs, input_)

    # `model:tier` sugar -- strip a recognised tier suffix; an explicit
    # service_tier= wins over it.
    parsed = parse_model_tier(model)
    service_tier = kwargs.get("service_tier") or parsed.get("serviceTier")

    client_config: dict[str, Any] = {
        "model": parsed["modelId"],
        "provider": provider,
        "api_key": kwargs.get("api_key"),
        "engine": kwargs.get("engine"),
        "transport": kwargs.get("transport"),
        **(kwargs.get("client") or {}),
    }
    # OpenAI takes input audio only through Chat Completions -- the Responses
    # API, which is its default here, rejects `input_audio`. Route audio there
    # unless the caller pinned an api themselves.
    if not client_config.get("api") and _provider_of(model, provider) == "openai" and _has_audio(
        input_
    ):
        client_config["api"] = "completions"

    options = {k: kwargs[k] for k in _FORWARDED if kwargs.get(k) is not None}
    if service_tier:
        options["service_tier"] = service_tier
    # A dataclass reaches the wire as the schema it describes. Sending the class
    # itself made the request carry an EMPTY schema, and Anthropic answered
    # "Empty schema ({}) that accepts any JSON" -- found by a live run.
    wire_structured = to_wire(kwargs.get("structured"))
    if wire_structured is not None:
        options["structured"] = wire_structured
    return {k: v for k, v in client_config.items() if v is not None}, input_, options


def _wants_parse(structured: Any) -> bool:
    if is_schema_class(structured):
        return True
    return bool(structured.get("schema")) if isinstance(structured, Mapping) else False


def _result(completion: Completion, structured: Any) -> Completion:
    """Attach `parsed` when a schema was asked for.

    A failed parse leaves `parsed` as None and `text` intact rather than raising:
    `12_structured_parse.py` requires exactly that, so a caller can retry or fall
    back on the raw answer. Raising would make a merely disappointing answer
    indistinguishable from a broken request.

    `Completion` is frozen, so this rebuilds rather than assigns.
    """
    if not _wants_parse(structured):
        return completion
    from dataclasses import replace

    try:
        return replace(completion, parsed=parse_into(structured, completion.text))
    except (InvalidFinalOutputError, SchemaError):
        # The two ways a disappointing answer arrives: it was not JSON, or it
        # was JSON of the wrong shape. Narrow on purpose -- a bug in the
        # instantiation code itself must still surface as a bug.
        return completion


def complete(**kwargs: Any) -> Completion:
    """One completion, synchronously. See the module docstring."""
    client_config, input_, options = _plan(kwargs)
    tools = kwargs.get("tools")
    max_steps = kwargs.get("max_steps") or DEFAULT_MAX_STEPS
    llm = LLM(**client_config)
    try:
        if tools:
            answer = run_tools(
                lambda messages, definitions: llm.complete(
                    messages, **{**options, "tools": definitions}
                ),
                input_,
                tools,
                max_steps=max_steps,
            )
        else:
            answer = llm.complete(input_, **options)
        return _result(answer, kwargs.get("structured"))
    finally:
        # Destroyed even on failure: the caller asked for one answer, not for a
        # client, and a leaked one keeps its hooks subscribed.
        llm.destroy()


async def acomplete(**kwargs: Any) -> Completion:
    """One completion, asynchronously. The twin of `complete`."""
    client_config, input_, options = _plan(kwargs)
    tools = kwargs.get("tools")
    max_steps = kwargs.get("max_steps") or DEFAULT_MAX_STEPS
    llm = AsyncLLM(**client_config)
    try:
        if tools:
            answer = await arun_tools(
                lambda messages, definitions: llm.complete(
                    messages, **{**options, "tools": definitions}
                ),
                input_,
                tools,
                max_steps=max_steps,
            )
        else:
            answer = await llm.complete(input_, **options)
        return _result(answer, kwargs.get("structured"))
    finally:
        llm.destroy()


__all__ = ["acomplete", "complete"]
