"""Server-side conversation state, where the provider keeps the history.

Google's server-side state lives on the (Beta) Interactions API, opted into with
`api="interactions"`. Instead of resending the whole conversation each turn, the
provider holds it and you send the new message plus a handle.

The SDK's server-state handling sends only what the second turn needs. The
saving is real on long conversations; the risk is that the handle is silently
dropped and every turn quietly resends everything, so the example asserts on
what actually goes on the wire.

Deterministic: stub transport captures the request bodies.
"""

from _check import check, report

from combycode_llm_sdk import LLM, TransportResponse

sent: list[dict] = []


def stub(request):
    sent.append(request.body)
    return TransportResponse(
        status=200,
        # The real Interactions shape: a step list of typed content, and a
        # `name` that IS the server-side handle. Not generateContent's
        # candidates -- a stub that mimics the wrong API tests nothing.
        body={
            "name": "interactions/int_123",
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": "ok"}]}],
            "usage": {"input_tokens": 10, "output_tokens": 2},
        },
    )


llm = LLM(model="google/gemini-2.5-flash", api_key="k", api="interactions", transport=stub)

first = llm.complete("My name is Ada.")
second = llm.complete("What is my name?", state=first.state)

check(first.state is not None, "the first turn must return a server-state handle")
check(len(sent) == 2, f"expected two requests, got {len(sent)}")

# The second request references the handle instead of resending the history.
second_body = str(sent[1])
check("int_123" in second_body, "the second turn must reference the server-side handle")
check("My name is Ada" not in second_body, "the second turn must NOT resend the history")

report(turns=len(sent), state_used=True, text=second.text)
