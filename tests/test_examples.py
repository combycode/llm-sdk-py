"""The reviewed examples, run as a gate.

`examples/features-highlight/` is self-verifying: stub transports, no keys, no
network, and every file exits non-zero when the behaviour it documents stops
being true. That makes them executable tests, and leaving them out of the suite
means a regression is found only when somebody remembers to run them by hand.

Two lists, because the two failures mean different things:

- PASSING must keep passing. A file that leaves this list has regressed.
- BLOCKED must fail with `NotImplementedError` and nothing else. That is the
  honest state of an unported feature, and the check that keeps it honest: a
  blocked example failing for some OTHER reason is a real break hiding behind
  an expected failure.

Adding a ported feature means moving a name from the second list to the first,
in the same commit that ports it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "features-highlight"
SRC = Path(__file__).resolve().parents[1] / "src"

#: Examples whose feature is ported. Every one must exit 0.
PASSING = [
    "openai_compatible_server",
    "agent_context_layers",
    "agent_guardrails_and_sampling",
    "checkpoint_persistence",
    "content_moderation",
    "context_anchored_strategy",
    "context_guard_in_the_loop",
    "context_window_guard",
    "cost_estimate_and_budget",
    "cost_unpriced_models",
    "engine_retry_policy",
    "event_stream",
    "exact_token_count",
    "final_answer_phase",
    "lazy_tools",
    "llm_backed_tools",
    "mcp_lazy_server",
    "model_filters",
    "model_selector",
    "multi_agent_delegation",
    "provenance_adapter",
    "response_cache",
    "retrieve_output_file",
    "scheduled_and_batched_work",
    "server_state_google_interactions",
    "steps_chain_and_fanout",
    "streamed_citations",
    "telemetry_redaction",
    "telemetry_traces",
    "tool_approval_gate",
    "tool_call_attribution",
    "tool_catalog_search",
    "tool_optional_parameters",
    "tool_permissions",
    "transcribe_structured",
    "unified_names",
]

#: Examples whose feature is not ported. Each must fail by NAMING what is
#: missing, never by crashing some other way.
BLOCKED: list[str] = []

#: Neither ported nor unported: these fail on a defect in the EXAMPLE, checked
#: against the TypeScript so that "the example is wrong" is a measurement rather
#: than a preference. Listed so their absence from both lists above is a
#: decision on the record rather than an oversight.
#:
#: - `response_shape_check` filters `code == "response_shape"`, which no
#:   implementation emits: both this port and the TypeScript emit
#:   `response_shape_unknown_field` / `_missing_field`, and the TypeScript's own
#:   twin of that example filters with `startsWith("response_shape_")`.
KNOWN_BAD_EXAMPLE = ["response_shape_check"]


def run(name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / f"{name}.py")],
        capture_output=True,
        text=True,
        cwd=EXAMPLES.parents[1],
        env={**os.environ, "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8"},
        timeout=120,
        check=False,
    )


def test_every_example_is_accounted_for() -> None:
    """No file may be silently absent from both lists.

    Without this, adding an example and forgetting to list it means it is never
    run -- and an example nobody runs is documentation, which is exactly what
    these were written not to be.
    """
    on_disk = {p.stem for p in EXAMPLES.glob("[a-z]*.py") if p.stem != "_check"}
    listed = set(PASSING) | set(BLOCKED) | set(KNOWN_BAD_EXAMPLE)
    assert on_disk == listed, f"unlisted: {sorted(on_disk - listed)}; stale: {sorted(listed - on_disk)}"


@pytest.mark.parametrize("name", PASSING)
def test_a_ported_example_still_passes(name: str) -> None:
    result = run(name)
    assert result.returncode == 0, f"{name} regressed:\n{result.stdout}\n{result.stderr}"


@pytest.mark.parametrize("name", BLOCKED)
def test_a_blocked_example_fails_by_naming_what_is_missing(name: str) -> None:
    result = run(name)
    assert result.returncode != 0, f"{name} now passes -- move it to PASSING"
    assert "NotImplementedError" in result.stderr, (
        f"{name} failed for a reason other than an unported name, which is a real "
        f"break hiding behind an expected failure:\n{result.stderr[-2000:]}"
    )


@pytest.mark.parametrize("name", KNOWN_BAD_EXAMPLE)
def test_a_known_bad_example_fails_only_on_its_own_defect(name: str) -> None:
    """Still failing, and still for the reason on the record.

    Without this, a known-bad entry is a place to park anything: an example
    listed here that broke for a NEW reason would look exactly like one that
    never worked, and one that started passing would stay listed forever.
    """
    result = run(name)
    assert result.returncode != 0, (
        f"{name} now passes -- the example was fixed, so move it to PASSING"
    )
    assert "NotImplementedError" not in result.stderr, (
        f"{name} is failing on an unported name, not on the documented defect. "
        f"That is a regression wearing a known-bad label:\n{result.stderr[-2000:]}"
    )
    assert "FAIL:" in (result.stdout + result.stderr), (
        f"{name} failed without reaching its own checks, so the recorded reason "
        f"is no longer the reason:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
