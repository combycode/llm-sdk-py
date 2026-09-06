"""Two fields travel with a tool round-trip and are easy to miss.

Most turns leave them absent, so code written against a simple case works until
the day a provider sends them -- and then fails in a way that looks like a model
error rather than a missing field.

    call_id    correlates a result with the call that asked for it. Required
               when a turn contains several calls; guessing by order is wrong
               the moment they complete out of order.
    signature  an opaque provider token (Gemini's thought signature) that must
               be echoed back verbatim or the next turn is rejected.

The SDK round-trips both. `assistant_message()` exists so a caller never has to
know that -- reconstructing history by hand is where these get dropped.

Deterministic: no network.
"""

from _check import check, report

from combycode_llm_sdk import ToolCall, ToolResult

call = ToolCall(
    id="call_abc123",
    name="get_weather",
    arguments={"city": "Paris"},
    signature="opaque-provider-token",
)

result = ToolResult.for_call(call, content="sunny")

# The result carries the id of the call it answers, not a positional guess.
check(result.call_id == call.id, "a result must name the call it answers")

# The signature survives the round trip untouched. It is opaque: the SDK must
# not parse, normalise or shorten it.
check(result.signature == call.signature, "an opaque signature must round-trip verbatim")

# Absent is normal, and must not be confused with empty.
plain = ToolCall(id="call_2", name="ping", arguments={})
check(plain.signature is None, "no signature must be None, not an empty string")

report(call_id=result.call_id, signature_preserved=result.signature == call.signature)
