"""Carrying a conversation forward without rebuilding it by hand.

`result.assistant_message()` gives back a message ready to append, so the caller
never has to know how this provider represents tool calls, thought signatures or
reasoning blocks in history. Getting that wrong by hand is a common and
confusing source of provider errors.
"""

from _bench import api_key, bench, model

from combycode_llm_sdk import LLM, Message


def main() -> str:
    llm = LLM(model=model(), api_key=api_key())

    history: list[Message] = [{"role": "user", "content": "My name is Ada."}]
    first = llm.complete(history, max_tokens=64)

    history.append(first.assistant_message())
    history.append({"role": "user", "content": "What is my name? One word."})

    second = llm.complete(history, max_tokens=16)
    return second.text.strip()


bench(main)
