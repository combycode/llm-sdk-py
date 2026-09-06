"""The async surface, shown once so its shape is fixed by this review too.

Two forms, and both are here because both are public: the module-level
`acomplete()` for a one-off, and `AsyncLLM` when you hold a client.

Not a wrapper in either direction: sync `LLM` does blocking I/O, `AsyncLLM`
drives the event loop. `asyncio.run()` inside a sync facade is what explodes in
a notebook or a web handler -- which is exactly where Python LLM code lives.
"""

import asyncio

from _bench import api_key, bench, model

from combycode_llm_sdk import AsyncLLM, acomplete


async def run() -> str:
    # One-off: no client to hold on to.
    first = await acomplete(
        model=model(),
        api_key=api_key(),
        prompt="Reply with exactly: OK",
        max_tokens=16,
    )

    # Held client, streamed. Same events and same `event.type` as the sync
    # `LLM.stream()` -- learning one teaches the other.
    llm = AsyncLLM(model=model(), api_key=api_key())
    text = ""
    async for event in llm.stream("Count from 1 to 5."):
        if event.type == "text":
            text += event.text

    return f"{first.text.strip()}|{text.strip()}"


bench(lambda: asyncio.run(run()))
