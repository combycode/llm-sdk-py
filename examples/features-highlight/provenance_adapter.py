"""Driving the provenance adapter directly, when you already hold a response.

`check_provenance()` is the one-call helper. This is the path you take when the
response is already in hand -- an audit re-checking stored transcripts, say --
and you want the evidence without making the call again.

The adapter returns evidence, not a boolean: which checks ran, which passed,
and what the provider actually claimed. A bare True/False hides the reason, and
the reason is the entire point of asking.

Deterministic: no network.
"""

from _check import check, report

from combycode_llm_sdk import OpenAIProvenanceAdapter

adapter = OpenAIProvenanceAdapter()

report_ = adapter.check(
    requested_model="gpt-5.4-nano",
    response={
        "id": "resp_1",
        "model": "gpt-5.4-nano-2026-01-15",
        "system_fingerprint": "fp_abc123",
        "usage": {"input_tokens": 10, "output_tokens": 3},
    },
)

check(report_.verdict in {"match", "mismatch", "unknown"}, f"unexpected verdict {report_.verdict}")
check(len(report_.checks) > 0, "the report must say which checks ran")

# A dated snapshot of the model we asked for is a MATCH, not a mismatch -- the
# provider pinning a version is normal and must not read as substitution.
check(report_.verdict == "match", f"a dated snapshot should match, got {report_.verdict}")

# A different family is what we are actually looking for.
substituted = adapter.check(
    requested_model="gpt-5.4-nano",
    response={"id": "r", "model": "gpt-3.5-turbo", "usage": {"input_tokens": 1, "output_tokens": 1}},
)
check(substituted.verdict == "mismatch", "a different model must be reported as a mismatch")

report(verdict=report_.verdict, checks=len(report_.checks), substituted=substituted.verdict)
