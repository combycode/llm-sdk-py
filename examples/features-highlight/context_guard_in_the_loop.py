"""The guard shrinks the request that is actually sent, not a copy of it.

Three objects, each wiring itself to the bus when it is constructed, exactly as
the TypeScript does:

  `ContextMeasurer(hooks=...)` listens on `onMessageResolve` -- the seam where
  the client hands over the message list before building a request -- measures
  it, and publishes the reading on `onContextMeasure`.

  `ContextGuard(hooks=...)` listens on THAT, finds the strategy for the
  conversation, and hands it a reading plus a set of tools.

  The strategy compacts the live list IN PLACE, so the shrunk conversation is
  the one that goes out. A guard that cannot make it fit sets `abort`, and the
  client refuses rather than paying to build a request the window cannot hold.

A hook rather than a direct call because more than one listener may care about
context pressure -- a guard that compacts, telemetry that records, a caller that
only wants a warning -- and none of them should have to know about each other.

**This needs the async client.** Compaction may call a model to summarise, so
`react` is async; the sync client emits with `emit_sync`, which starts async
handlers without awaiting them, and the bus warns when that happens rather than
letting a subscribed guard look like one that simply had nothing to do.

Deterministic: a stub transport, no key, no network. What is asserted is the
body the transport RECEIVED -- asserting on the guard's return value would have
passed the entire time it was unreachable.
"""

import asyncio
from typing import Any

from _check import check, report

from combycode_llm_sdk import ConversationHistory, Engine, LLMError
from combycode_llm_sdk.catalog.catalog import resolve_catalog
from combycode_llm_sdk.context.context_guard import ContextGuard
from combycode_llm_sdk.context.guard import ContextMeasurer
from combycode_llm_sdk.context.strategies import TruncateStrategy
from combycode_llm_sdk.context.strategy_types import TriggerLevel
from combycode_llm_sdk.helpers.llm import default_adapter_factory
from combycode_llm_sdk.llm.async_client import AsyncLLMClient
from combycode_llm_sdk.transport import TransportRequest, TransportResponse

PROVIDER = "anthropic"
MODEL = "claude-haiku-4.5"
TURNS = 14
KEEP_RECENT = 4


class Recorder:
    """A transport that answers once and keeps what it was handed."""

    def __init__(self) -> None:
        self.seen: list[TransportRequest] = []

    def __call__(self, request: TransportRequest) -> TransportResponse:
        self.seen.append(request)
        return TransportResponse(
            body={
                "content": [{"type": "text", "text": "noted"}],
                "usage": {"input_tokens": 9, "output_tokens": 2},
                "stop_reason": "end_turn",
            }
        )

    def messages_sent(self) -> list[Any]:
        body = self.seen[0].body
        assert isinstance(body, dict)
        return body["messages"]


def conversation(turns: int = TURNS) -> ConversationHistory:
    """The transcript, as the object the guard reasons about.

    A plain list would be enough to SEND, but not to guard: the guard keeps its
    place on the trigger ladder in `history.metadata`, so it can tell a new
    crossing from the same pressure re-reported. Without a history it has
    nowhere to remember that, and correctly does nothing.
    """
    history = ConversationHistory("demo")
    for i in range(turns):
        history.append(
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " * 60}
        )
    return history


def text_of(message: dict[str, Any]) -> str:
    """The message's text, whichever shape the builder normalised it into."""
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


async def send(history: ConversationHistory, *, guarded: bool, ceiling: float = 0.95) -> Recorder:
    """One call, with or without a guard attached to the engine's bus."""
    recorder = Recorder()
    engine = Engine(register_as_default=False)
    afetch, _ = engine.afetches_for(recorder)
    if guarded:
        measurer = ContextMeasurer(resolve_catalog("defaults"), hooks=engine.hooks)
        ContextGuard(
            hooks=engine.hooks,
            measurer=measurer,
            strategies={
                "demo": TruncateStrategy(
                    keep_recent=KEEP_RECENT,
                    decline_ceiling=ceiling,
                    # Reacts at any size, so the arithmetic here is about the
                    # wiring rather than about a particular model's window.
                    triggers=[TriggerLevel("urgent", 0.0)],
                )
            },
            default_strategy="demo",
        )
    client = AsyncLLMClient(
        {
            "provider": PROVIDER,
            "model": MODEL,
            "apiKey": "k",
            "fetch": afetch,
            "hooks": engine.hooks,
            "adapter": default_adapter_factory(),
        }
    )
    await client.complete(history.messages(), {"history": history})
    return recorder


async def main() -> None:
    # -- unattached, the whole conversation goes out -------------------------
    #
    # The control. Without it, "the request got smaller" could be true because
    # the example never sent a large one.

    full = await send(conversation(), guarded=False)
    check(
        len(full.messages_sent()) == TURNS,
        f"with no guard, every turn should reach the provider, got {len(full.messages_sent())}",
    )

    # -- attached, the request ITSELF is smaller -----------------------------

    kept = await send(conversation(), guarded=True)
    sent = kept.messages_sent()
    check(len(sent) == KEEP_RECENT, f"the guard kept {KEEP_RECENT} turns, got {len(sent)}")
    check(
        text_of(sent[-1]).startswith(f"turn {TURNS - 1}"),
        "and it kept the RECENT ones -- the last exchange is what the next answer depends on",
    )

    # -- a guard that cannot make it fit refuses the call --------------------
    #
    # Nothing still holding a conversation clears a ceiling of zero, so this
    # stands in for the real case: compaction ran, it was not enough, and
    # sending anyway would buy a provider rejection after paying to build the
    # request.

    refused = Recorder()
    engine = Engine(register_as_default=False)
    afetch, _ = engine.afetches_for(refused)
    measurer = ContextMeasurer(resolve_catalog("defaults"), hooks=engine.hooks)
    ContextGuard(
        hooks=engine.hooks,
        measurer=measurer,
        strategies={
            "demo": TruncateStrategy(
                keep_recent=2, decline_ceiling=0.0, triggers=[TriggerLevel("urgent", 0.0)]
            )
        },
        default_strategy="demo",
    )
    client = AsyncLLMClient(
        {
            "provider": PROVIDER,
            "model": MODEL,
            "apiKey": "k",
            "fetch": afetch,
            "hooks": engine.hooks,
            "adapter": default_adapter_factory(),
        }
    )
    try:
        doomed = conversation()
        await client.complete(doomed.messages(), {"history": doomed})
        aborted = False
    except LLMError:
        aborted = True

    check(aborted, "a guard that declines must refuse the call")
    check(refused.seen == [], "and refusing means nothing reached the provider at all")

    report(
        unguarded=len(full.messages_sent()),
        guarded=len(sent),
        kept_the_recent_end=text_of(sent[-1])[:8].strip(),
        declined=aborted,
    )


asyncio.run(main())
