"""A bad REQUEST is a 400. A bad RESPONSE is a 200 with a field missing.

That asymmetry is the problem. The parse succeeds, the field we read is simply
not there any more, and the value becomes None. If that field was
`usage.output_tokens`, cost reporting quietly goes to zero and nothing errors.

The shape check compares what arrived against what the response spec declares
and reports the difference as a warning -- off by default (it costs a
comparison per response), on when you want to know.

Deterministic: a stub returns a response with a renamed field.
"""

from _check import check, report

from combycode_llm_sdk import LLM, Engine, TransportResponse

# `output_tokens` has been renamed by the provider. Everything still parses.
DRIFTED = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "completion_tokens": 5},
    "surprise_field": "nobody declared this",
}


def stub(request):
    return TransportResponse(status=200, body=DRIFTED)


warnings: list[object] = []
engine = Engine(
    api_keys={"anthropic": "k"},
    transport=stub,
    check_response_shapes=True,
    register_as_default=False,
)


@engine.on_warning
def collect(ctx) -> None:
    warnings.append(ctx)


llm = LLM(model="anthropic/claude-haiku-4.5", engine=engine)
result = llm.complete("hi")

# The call still succeeds. Failing here would be worse than the drift: an
# unknown field is not a reason to deny the caller an answer they were given.
check(result.text == "ok", "an unexpected shape must not break a usable response")

shape_warnings = [w for w in warnings if w.code == "response_shape"]
check(len(shape_warnings) > 0, "a drifted response must produce a shape warning")

detail = str(shape_warnings[0])
check("surprise_field" in detail, "an undeclared field should be named")

report(text=result.text, shape_warnings=len(shape_warnings))
