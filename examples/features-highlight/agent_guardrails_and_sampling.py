"""Guardrails, sampling that only goes where the wire accepts it, and second chances.

Three things an agent needs that a single `complete()` call does not:

1. Sampling parameters are not universally supported. Sending `temperature` to
   a model that rejects it is a 400; dropping it silently is worse. The SDK
   sends what the wire accepts and WARNS about what it dropped.

2. Some failures are not the network's -- a malformed tool call, a hallucinated
   tool name. Resending the identical request would never fix them, so the retry
   layer correctly leaves them alone, and the run used to just end there.
   `reflect_and_retry` gives the model a bounded number of second chances with
   the error fed back to it.

3. Guardrails run before and after: a moderation check, a budget ceiling.

Deterministic: stub transport, no keys, no network.
"""

from _check import check, report

from combycode_llm_sdk import Agent, ReflectAndRetry, TransportResponse, tool

STUB = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude",
    "content": [{"type": "text", "text": "done"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 3},
}


def stub(request):
    return TransportResponse(status=200, body=STUB)


@tool
def lookup(city: str) -> str:
    """Look up a city."""
    return "found"


warnings: list[object] = []

agent = Agent(
    model="anthropic/claude-haiku-4.5",
    api_key="k",
    transport=stub,
    tools=[lookup],
    system="You are terse.",
    # Bounded second chances for mistakes the MODEL made. Bounded on purpose:
    # unbounded reflection is an infinite loop with a bill attached.
    reflect_and_retry=ReflectAndRetry(max_attempts=2, on=["malformed_tool_call", "unknown_tool"]),
    # Sampling: sent where accepted, reported where not.
    temperature=0.2,
    top_p=0.9,
)


@agent.on_warning
def collect(ctx) -> None:
    warnings.append(ctx)


result = agent.complete("Where am I?")
check(result.text == "done", "the agent should complete")

# A guardrail is a plain callable: no framework, no registry, no base class --
# and it raises YOUR exception, not one of ours. A budget is a policy the
# library has no opinion about.
class BudgetExceeded(RuntimeError):
    pass


def budget_guard(ctx) -> None:
    if ctx.cost and ctx.cost.total > 1.00:
        raise BudgetExceeded(ctx.cost.total)


guarded = Agent(
    model="anthropic/claude-haiku-4.5",
    api_key="k",
    transport=stub,
    before=[lambda ctx: None],
    after=[budget_guard],
)
check(guarded.complete("hi").text == "done", "guardrails must not disturb a healthy run")

report(text=result.text, warnings=[getattr(w, "code", None) for w in warnings])
