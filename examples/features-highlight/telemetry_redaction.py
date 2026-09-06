"""URLs and headers are always redacted. The switch governs FREE TEXT.

A provider's `error.message` routinely echoes request content straight back: a
moderation refusal quotes the prompt, a validation error names the offending
field and its value. That text is the payload, so whether it reaches your
telemetry backend has to be a decision, not an accident.

Default is on (redacted), which is the safe direction: turning redaction OFF is
a deliberate act by someone who knows what their backend stores.

Deterministic: no network.
"""

from _check import check, report

from combycode_llm_sdk import Engine, TelemetryAdapter

# ── default: free text is redacted ──────────────────────────────────────────
safe = TelemetryAdapter()
engine = Engine(register_as_default=False, plugins=[safe])
engine.emit_warning(
    source="llm",
    code="provider_error",
    message="Invalid value for 'email': alice@example.com",
)

recorded = safe.events[-1]
check("alice@example.com" not in str(recorded), "free text must be redacted by default")
check(recorded.code == "provider_error", "the CODE is structural and must survive")

# ── opt out, deliberately ───────────────────────────────────────────────────
verbose = TelemetryAdapter(redact_free_text=False)
engine2 = Engine(register_as_default=False, plugins=[verbose])
engine2.emit_warning(
    source="llm",
    code="provider_error",
    message="Invalid value for 'email': alice@example.com",
)
check("alice@example.com" in str(verbose.events[-1]), "opting out must actually opt out")

# ── never optional ──────────────────────────────────────────────────────────
# API keys live in URLs (Google) and headers (everyone else). Those are redacted
# whatever the switch says: there is no legitimate reason to export them.
engine2.emit_request(url="https://api.example.com/v1?key=sk-secret-123", headers={"authorization": "Bearer sk-secret-123"})
dumped = str(verbose.events[-1])
check("sk-secret-123" not in dumped, "credentials must be redacted regardless of the switch")

report(default_redacted=True, opt_out_works=True, credentials_always_redacted=True)
