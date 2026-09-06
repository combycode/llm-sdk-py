"""This library behind an OpenAI-shaped HTTP API -- exercised without a socket.

Register a model, point an existing OpenAI client at the port, and the request
reaches whichever provider that model was registered with. `create_server` is
the one-call form of it: a dict of agents becomes model ids, each with an agent
loader and a conversation store already wired.

`handle()` is a PURE function from a parsed request to a response -- no socket,
no framework, no globals, and the socket shell is a separate file that only
turns bytes into the `HttpRequest` built by hand below. That is the decision
worth demonstrating, so this file demonstrates it the way it pays off: every
route here is reached by CALLING the handler, and nothing binds a port. A
server whose auth, routing and error mapping could only be observed through a
listening socket is one where none of the three is ever tested.

Deterministic: stub transport, no keys, no network, no port.
"""

from typing import Any

from _check import check, report

from combycode_llm_sdk import LLM, Agent, OaiServer, TransportResponse, create_server
from combycode_llm_sdk.server import BearerKeyAuth, HttpRequest, ServerEntry

MODEL = "fast"
AGENT_MODEL = "assistant"
PROVIDER_MODEL = "openai/gpt-4o-mini"
KEY = "sk-local"

ANSWER = "Hi."


class Provider:
    """The upstream this server routes to, and whether it was ever reached."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: Any) -> TransportResponse:
        self.calls += 1
        return TransportResponse(
            status=200,
            body={
                "id": "resp_provider",
                "status": "completed",
                # A real Responses item carries its type and role: the official
                # SDK types both as fixed ("Always `message`", "Always
                # `assistant`"), and every parser keys on the type to tell a
                # message from a tool call or a reasoning block. Without it the
                # item is skipped and the answer arrives empty -- which is what
                # this stub used to assert against.
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": ANSWER}],
                    }
                ],
                "usage": {"input_tokens": 11, "output_tokens": 3},
            },
        )


def chat(model: str, key: str | None = None) -> HttpRequest:
    """What an OpenAI client sends, already parsed -- which is all the shell does."""
    return HttpRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={} if key is None else {"authorization": f"Bearer {key}"},
        body={"model": model, "messages": [{"role": "user", "content": "hi"}]},
    )


provider = Provider()
server = OaiServer(
    entries=[ServerEntry(model=MODEL, client=LLM(model=PROVIDER_MODEL, api_key="k",
                                                 transport=provider))],
    auth=BearerKeyAuth({KEY: "alice"}),
)

answered = server.handle(chat(MODEL, KEY))
check(answered.status == 200, "an authenticated chat request is answered")
check(answered.body["choices"][0]["message"]["content"] == ANSWER,
      "the provider's answer comes back in OpenAI's shape")
# The model id is the one the operator registered, not the provider's own: that
# is what lets a client switch providers without changing the request.
check(answered.body["model"] == MODEL, "the reply names the registered id")
check(answered.body["usage"]["total_tokens"] > 0, "usage is never a false zero")

served = provider.calls
refused = server.handle(chat(MODEL))
check(refused.status == 401, "a request with no credential is refused")
check(refused.body["error"]["type"] == "authentication_error", "and told which kind of refusal")
check(provider.calls == served, "the refused request never reached the provider")

# Liveness is answered before any credential is checked: a probe carries none,
# and one that answers 401 reads as a dead process.
health = server.handle(HttpRequest(method="GET", path="/health"))
check(health.status == 200, "the health route is public")

listed = server.handle(HttpRequest(method="GET", path="/v1/models",
                                   headers={"authorization": f"Bearer {KEY}"}))
check([row["id"] for row in listed.body["data"]] == [MODEL], "what is registered is listed")

# The same handler, wired the short way: an Agent -- its tools, its system
# prompt -- answering an ordinary OpenAI request.
agent_provider = Provider()
helper = create_server(
    agents={AGENT_MODEL: Agent(model=PROVIDER_MODEL, api_key="k", transport=agent_provider,
                               system="You are terse.")}
)
via_agent = helper.handle(chat(AGENT_MODEL))
check(via_agent.status == 200, "create_server registers each agent as a model id")
check(via_agent.body["choices"][0]["message"]["content"] == ANSWER, "and the agent answers it")

report(model=answered.body["model"], answered=answered.status, refused=refused.status,
       via_agent=via_agent.body["model"])
