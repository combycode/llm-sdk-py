"""A large file uploaded once and referenced, instead of inlined every call.

Same `attachments=` parameter. Whether a file is inlined as base64 or uploaded
to the provider's file store is the SDK's decision, made from size and provider
capability -- not a second API the caller has to learn.
"""

from pathlib import Path

from _bench import api_key, bench, model

from combycode_llm_sdk import complete

FIXTURES = Path(__file__).parent.parent / "fixtures"


def main() -> str:
    result = complete(
        model=model(),
        api_key=api_key(),
        prompt="Summarise this document in one sentence.",
        attachments=[FIXTURES / "sample.pdf"],
        max_tokens=128,
    )
    return result.text.strip()


bench(main)
