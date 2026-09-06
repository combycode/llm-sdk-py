"""Messages are plain dicts, the way Python LLM users already write them.

`Message` is a TypedDict, so a checker still catches a bad role or a missing key
-- but nothing forces a caller to import and construct a class to say something
as ordinary as "user: hi".
"""

from _bench import api_key, bench, model

from combycode_llm_sdk import complete


def main() -> str:
    result = complete(
        model=model(),
        api_key=api_key(),
        messages=[
            {"role": "user", "content": "My name is Ada."},
            {"role": "assistant", "content": "Nice to meet you, Ada."},
            {"role": "user", "content": "What is my name? One word."},
        ],
        max_tokens=16,
    )
    return result.text.strip()


bench(main)
