"""Did the model that answered match the one that was asked for?

A provider may serve a request with a different model than the one named:
routing, a deprecation alias, a silent substitution during an incident. The
answer still looks fine, the bill still arrives, and the only record that it
happened is in the response's own `model` field -- which nothing reads.

The adapter returns EVIDENCE, not a boolean: which checks ran, which passed, and
what the provider actually claimed. A bare True/False hides the reason, and the
reason is the entire point of asking.

Distinct from `check_provenance()`, which asks whether a FILE carries a C2PA or
SynthID signal. Same word, different question: that one is about content, this
one is about which model produced a response.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

#: A dated snapshot suffix: `-2026-01-15`, or a bare `-20260115`.
_SNAPSHOT = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8}|latest|preview)$")

MATCH = "match"
MISMATCH = "mismatch"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProvenanceCheck:
    """One thing that was checked, and what it found."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class ProvenanceReport:
    """What the provider claimed, and whether it matches what was asked."""

    #: `match` | `mismatch` | `unknown`. Three, not two: a response that names
    #: no model has not matched and has not been substituted either, and
    #: collapsing that into `mismatch` would raise an alarm about a provider
    #: that simply reports less.
    verdict: str
    requested_model: str = ""
    served_model: str = ""
    checks: Sequence[ProvenanceCheck] = ()
    #: Anything the response carried that identifies the serving stack.
    evidence: Mapping[str, Any] = field(default_factory=dict)
    #: The completion this verdict was read from, so a caller who wanted the
    #: answer as well does not pay for a second call to get it.
    completion: Any = None

    @property
    def matched(self) -> bool:
        return self.verdict == MATCH


def normalise_model(model: str) -> str:
    """A model id with any dated-snapshot suffix removed.

    `gpt-5.4-nano-2026-01-15` and `gpt-5.4-nano` are the same model: a provider
    pinning a version is normal, and reading it as a substitution would make the
    check cry wolf on every well-behaved request.
    """
    return _SNAPSHOT.sub("", model.strip())


class OpenAIProvenanceAdapter:
    """Reads what an OpenAI-shaped response says about itself."""

    #: Fields that identify the serving stack rather than the model. Reported as
    #: evidence, never as a verdict: a fingerprint changing means the backend
    #: changed, which is not the same as being served another model.
    EVIDENCE_FIELDS: tuple[str, ...] = (
        "system_fingerprint",
        "id",
        "created",
        "service_tier",
    )

    provider = "openai"

    def served_model(self, response: Mapping[str, Any]) -> str:
        return str(response.get("model") or "")

    def check(
        self, *, requested_model: str, response: Mapping[str, Any]
    ) -> ProvenanceReport:
        """Compare what was asked for with what came back."""
        served = self.served_model(response)
        checks: list[ProvenanceCheck] = []

        checks.append(
            ProvenanceCheck(
                name="response_names_a_model",
                passed=bool(served),
                detail=served or "the response carried no model field",
            )
        )
        if not served:
            # Nothing to compare. Not a mismatch: a provider that reports less
            # has not substituted anything, and saying it did would train the
            # reader to ignore the check.
            return ProvenanceReport(
                verdict=UNKNOWN,
                requested_model=requested_model,
                checks=tuple(checks),
                evidence=self._evidence(response),
            )

        wanted = normalise_model(requested_model)
        got = normalise_model(served)
        exact = served == requested_model
        family = got == wanted

        checks.append(
            ProvenanceCheck(
                name="model_family",
                passed=family,
                detail=f"asked for {wanted!r}, served {got!r}",
            )
        )
        checks.append(
            ProvenanceCheck(
                name="dated_snapshot",
                passed=family and not exact,
                detail=(
                    f"{served!r} is a pinned snapshot of {wanted!r}"
                    if family and not exact
                    else "the served id is not a dated snapshot of the requested one"
                ),
            )
        )
        evidence = self._evidence(response)
        checks.append(
            ProvenanceCheck(
                name="serving_stack_identified",
                passed=bool(evidence),
                detail=", ".join(sorted(evidence)) or "no identifying fields on the response",
            )
        )

        return ProvenanceReport(
            verdict=MATCH if family else MISMATCH,
            requested_model=requested_model,
            served_model=served,
            checks=tuple(checks),
            evidence=evidence,
        )

    def _evidence(self, response: Mapping[str, Any]) -> dict[str, Any]:
        return {
            name: response[name] for name in self.EVIDENCE_FIELDS if response.get(name) is not None
        }


class AnthropicProvenanceAdapter(OpenAIProvenanceAdapter):
    """The same comparison against an Anthropic-shaped response.

    A subclass rather than a copy: the question and the verdicts are identical
    and only the evidence fields differ, so two implementations would be two
    places for the family comparison to drift.
    """

    EVIDENCE_FIELDS: tuple[str, ...] = ("id", "stop_reason", "stop_sequence")
    provider = "anthropic"


__all__ = [
    "MATCH",
    "MISMATCH",
    "UNKNOWN",
    "AnthropicProvenanceAdapter",
    "OpenAIProvenanceAdapter",
    "ProvenanceCheck",
    "ProvenanceReport",
    "normalise_model",
]


def provenance_adapter(provider: str) -> OpenAIProvenanceAdapter:
    """The reader for one provider's response shape."""
    if provider == "anthropic":
        return AnthropicProvenanceAdapter()
    return OpenAIProvenanceAdapter()


def check_provenance(
    *,
    model: str,
    prompt: Any = None,
    api_key: str | None = None,
    provider: str | None = None,
    engine: Any = None,
    **options: Any,
) -> ProvenanceReport:
    """Make one call, then ask whether the model that answered is the one asked for.

        report = check_provenance(model="openai/gpt-5.4-nano", prompt="hi", api_key=key)
        report.verdict          # match | mismatch | unknown
        report.checks           # what was checked, and what each found

    A real request, deliberately: a provider substitutes at serve time, so the
    only way to find out is to be served. The completion is on the report as
    `completion` for a caller who wants the answer too rather than paying twice.
    """
    from .helpers.client_resolver import is_namespaced_model_id, parse_model_id
    from .helpers.one_shot import complete

    provider_name, model_name = (
        parse_model_id(model) if is_namespaced_model_id(model) else (provider, model)
    )
    if not provider_name:
        raise ValueError(
            'check_provenance: name the provider, either as provider= or as '
            '"provider/model".'
        )

    answer = complete(model=model, prompt=prompt, api_key=api_key, engine=engine, **options)
    # The RAW response, not our parsed view: the model field we are checking is
    # the provider's own, and reading our normalised copy would be checking our
    # own bookkeeping rather than theirs.
    raw = getattr(answer, "raw", None)
    report = provenance_adapter(provider_name).check(
        requested_model=model_name, response=raw if isinstance(raw, Mapping) else {}
    )
    return replace(report, completion=answer)
