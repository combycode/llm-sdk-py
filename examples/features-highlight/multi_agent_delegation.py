"""Several agents, and whose work is whose.

`delegate(name, description, agent)` makes an agent a TOOL another agent can
call. `handoff(...)` is the same tool with the sub-agent's usage and name
attached to the answer. `Observer` watches one agent through the engine's hook
bus. `ClientPool` hands out clients that share a single engine.

What has to be right here is attribution. A specialist called as a tool runs
INSIDE its caller's run, on the same engine and usually the same model, so
anything that scopes by run -- or guesses by model -- files the specialist's
tokens under the caller. Nothing downstream can tell: every number stays
plausible, the per-agent cost is simply wrong, and the specialist looks free.

Deterministic: scripted transports, no key, no network.
"""

import json

from _check import check, report

from combycode_llm_sdk import (
    Agent,
    ClientPool,
    Engine,
    Observer,
    TransportResponse,
    delegate,
    handoff,
)
from combycode_llm_sdk.transport import TransportRequest

MODEL = "anthropic/claude-haiku-4.5"
QUESTION = "what is 6*7"
EXPERT_TOKENS = 11
BOSS_TOOL_TURN_TOKENS = 1
BOSS_ANSWER_TOKENS = 7


def text_body(text: str, output_tokens: int) -> dict:
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 3, "output_tokens": output_tokens},
        "stop_reason": "end_turn",
    }


def tool_body(name: str, **arguments: object) -> dict:
    return {
        "content": [{"type": "tool_use", "id": "c1", "name": name, "input": arguments}],
        "usage": {"input_tokens": 3, "output_tokens": BOSS_TOOL_TURN_TOKENS},
        "stop_reason": "tool_use",
    }


class Script:
    """A transport that answers from a script and keeps what it was sent."""

    def __init__(self, *bodies: dict) -> None:
        self.bodies = list(bodies)
        self.seen: list[TransportRequest] = []

    def __call__(self, request: TransportRequest) -> TransportResponse:
        self.seen.append(request)
        # The last body repeats, so a tool loop needs no turn-by-turn script.
        body = self.bodies.pop(0) if len(self.bodies) > 1 else self.bodies[0]
        return TransportResponse(status=200, body=body)

    def prompts(self) -> list[str]:
        return [
            " ".join(part["text"] for part in request.body["messages"][0]["content"])
            for request in self.seen
        ]


# One engine for the whole fleet: two engines against one provider means two
# rate limiters, and the concurrency configured for it silently doubles.
shared = Engine(register_as_default=False)

expert_transport = Script(text_body("42", EXPERT_TOKENS))
expert = Agent(
    model=MODEL, api_key="k", engine=shared, transport=expert_transport,
    system="You are an arithmetic expert.",
)

# ── an agent as a tool ──────────────────────────────────────────────────────
ask = delegate("ask_expert", "Ask the expert a question.", expert)

check(ask(task=QUESTION) == "42", "delegate must return the sub-agent's own answer")
# Only the task crosses. The specialist starts from its OWN system prompt and an
# empty history -- which is the point of a specialist, and also the thing that
# surprises a caller who expected the conversation to travel with the task.
check(expert_transport.prompts() == [QUESTION], "the sub-agent must see the task and nothing else")
check(
    expert_transport.seen[0].body["system"] == "You are an arithmetic expert.",
    "the sub-agent must keep its own system prompt",
)

# ── the same tool, with the answer attributed ───────────────────────────────
# `delegate` returns bare text, so a parent holding three specialists cannot
# tell which one replied or what the reply cost.
escalate = handoff("ask_expert", "Ask the expert a question.", expert)
answered = json.loads(escalate(task=QUESTION))

check(answered["text"] == "42", "a handoff must carry the sub-agent's answer")
check(answered["agent_name"] == "ask_expert", "a handoff must say WHICH specialist answered")
# `usage` is None when the provider reported none, so a handoff that lost it
# looks exactly like a provider that never sent it.
check(
    answered["usage"] is not None and answered["usage"]["output_tokens"] == EXPERT_TOKENS,
    f"a handoff must report what the sub-agent's own run cost -- got {answered['usage']}",
)

# ── watching two agents through one run ─────────────────────────────────────
boss_transport = Script(
    tool_body("ask_expert", task=QUESTION),
    text_body("the expert says 42", BOSS_ANSWER_TOKENS),
)
boss = Agent(model=MODEL, api_key="k", engine=shared, transport=boss_transport, tools=[escalate])

for_expert: list = []
for_boss: list = []
with (
    Observer(expert, "on_completion", for_expert.append),
    Observer(boss, "on_completion", for_boss.append) as watching_boss,
):
    answer = watching_boss.run("ask the expert what 6*7 is")

check(answer.text == "the expert says 42", f"the boss must answer -- got {answer.text!r}")
# Three completions happened on one engine, one model and one run. Each observer
# sees only its own agent's: the specialist's turn is the specialist's, and the
# boss's two turns stay the boss's.
check(len(for_expert) == 1, f"the sub-agent's turn is its own -- saw {len(for_expert)}")
check(len(for_boss) == 2, f"the boss's own turns are the boss's -- saw {len(for_boss)}")
check(
    for_expert[0].usage.output_tokens == EXPERT_TOKENS,
    "the sub-agent's usage must reach its own observer",
)
# And the same line drawn in the billing: two runs happened, and adding them
# would make one call look like it billed for two models at once.
check(
    answer.usage.output_tokens == BOSS_TOOL_TURN_TOKENS + BOSS_ANSWER_TOKENS,
    f"a parent's usage must not absorb its sub-agent's -- got {answer.usage.output_tokens}",
)

# ── the clients they share ──────────────────────────────────────────────────
pool = ClientPool(engine=shared)
haiku = pool.get("anthropic", "claude-haiku-4.5", api_key="k")
sonnet = pool.get("anthropic", "claude-sonnet-4.5", api_key="k")

check(pool.get("anthropic", "claude-haiku-4.5", api_key="k") is haiku, "one config, one client")
# `LLM` binds its model at construction, so keying by provider alone would hand
# back a client pinned to whichever model asked first.
check(haiku is not sonnet, "a second model must not be given the first one's client")
check({id(c.engine) for c in (haiku, sonnet)} == {id(shared)}, "the pool's engine is the fleet's")

report(
    answer=answer.text,
    expert_turns=len(for_expert),
    boss_turns=len(for_boss),
    expert_tokens=for_expert[0].usage.output_tokens,
    boss_tokens=answer.usage.output_tokens,
    clients=pool.size,
)
