"""Trying models in order until one answers.

Two strategies, and which one runs is decided by the candidates themselves:

- **All OpenRouter** -- one request carrying a `models` array, and OpenRouter
  falls over server-side. One round trip instead of N, and the response says who
  actually served it.
- **Anything else** -- try each in turn, client-side.

The part that matters is WHICH failures are worth another model. A rate limit or
a 500 is; an auth failure, a malformed request or a content filter is not --
sending the same bad request to a second provider spends money to be told the
same thing. So the fallback list is a denylist by default and the caller can
override it.

Transposed from `unified-library-ts/src/helpers/route.ts`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..network.errors import LLMError

#: Failures another model might survive. Deliberately excludes `auth`,
#: `invalid_request`, `content_filter` and `context_overflow`: a model swap
#: cannot fix a request that is wrong, and trying anyway just costs more.
DEFAULT_FALLBACK_KINDS: frozenset[str] = frozenset(
    {
        "rate_limit",
        "server_error",
        "model_not_found",
        "timeout",
        "network",
        "quota_exceeded",
        "unsupported",
    }
)


@dataclass(frozen=True)
class RouteAttempt:
    """One candidate, and why it did not answer."""

    model: str
    error: str | None = None
    kind: str | None = None


@dataclass
class RouteResult:
    """The completion, plus who served it and what was tried first."""

    completion: Any
    served_by: str
    attempts: Sequence[RouteAttempt] = field(default_factory=tuple)

    # The answer is the point, so the common reads pass straight through rather
    # than making every caller reach into `.completion`.
    @property
    def text(self) -> str:
        return str(getattr(self.completion, "text", ""))

    @property
    def model(self) -> str:
        """Who actually served it.

        The same value as `served_by`, under the name a completion uses -- so a
        caller can swap `complete()` for `route()` without rewriting the line
        that reads the model back.
        """
        return self.served_by

    @property
    def parsed(self) -> Any:
        return getattr(self.completion, "parsed", None)


def _provider_of(model: str) -> str | None:
    from .client_resolver import is_namespaced_model_id, parse_model_id

    return parse_model_id(model)[0] if is_namespaced_model_id(model) else None


def _bare(model: str) -> str:
    from .client_resolver import is_namespaced_model_id, parse_model_id

    return parse_model_id(model)[1] if is_namespaced_model_id(model) else model


def route(
    *,
    models: Sequence[str],
    fallback_on: Sequence[str] | None = None,
    **options: Any,
) -> RouteResult:
    """Complete with the first model that works.

        answer = route(models=["openai/gpt-5.4-nano", "anthropic/claude-haiku-4.5"],
                       prompt="hello", api_key=key)
        answer.text
        answer.served_by
    """
    from .one_shot import complete

    candidates = list(models or ())
    if not candidates:
        raise ValueError("route: `models` must be a non-empty list.")

    # OpenRouter routes for us when every candidate is one of its own, which is
    # a round trip saved rather than a different behaviour. It wants the BARE
    # ids, not our `openrouter/` namespace.
    if len(candidates) > 1 and all(_provider_of(m) == "openrouter" for m in candidates):
        provider_options = dict(options.pop("provider_options", None) or {})
        openrouter = dict(provider_options.get("openrouter") or {})
        openrouter["models"] = [_bare(m) for m in candidates]
        provider_options["openrouter"] = openrouter
        answer = complete(
            model=candidates[0], provider_options=provider_options, **options
        )
        return RouteResult(
            completion=answer,
            served_by=getattr(answer, "model", "") or candidates[0],
            attempts=(RouteAttempt(model=candidates[0]),),
        )

    kinds = set(fallback_on) if fallback_on is not None else set(DEFAULT_FALLBACK_KINDS)
    attempts: list[RouteAttempt] = []
    for model in candidates:
        try:
            answer = complete(model=model, **options)
        except LLMError as exc:
            kind = getattr(exc, "kind", None)
            attempts.append(RouteAttempt(model=model, error=str(exc), kind=kind))
            # A failure this list does not name is one another model will meet
            # too, so it is raised rather than retried.
            if kind is None or kind not in kinds:
                raise
        else:
            attempts.append(RouteAttempt(model=model))
            return RouteResult(completion=answer, served_by=model, attempts=tuple(attempts))

    detail = "\n".join(
        f"  {a.model} [{a.kind or 'error'}]: {a.error}" for a in attempts
    )
    raise RuntimeError(f"route: all {len(candidates)} model(s) failed:\n{detail}")


__all__ = ["DEFAULT_FALLBACK_KINDS", "RouteAttempt", "RouteResult", "route"]
