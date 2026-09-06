"""Streaming is a plain `for`, not a callback and not an async requirement.

The async twin is `AsyncLLM(...).stream(...)` with `async for` -- see
04b_streaming_async.py. Neither is implemented in terms of the other.
"""

from _bench import api_key, bench, model

from combycode_llm_sdk import LLM


def main() -> str:
    llm = LLM(model=model(), api_key=api_key())
    text = ""
    for event in llm.stream("Count from 1 to 5."):
        if event.type == "text":
            text += event.text
    return text.strip()


bench(main)
