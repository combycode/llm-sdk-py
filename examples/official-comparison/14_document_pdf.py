"""A PDF is an attachment like any other -- same parameter, same call."""

from pathlib import Path

from _bench import api_key, bench, model

from combycode_llm_sdk import complete

FIXTURES = Path(__file__).parent.parent / "fixtures"


def main() -> str:
    result = complete(
        model=model(),
        api_key=api_key(),
        prompt="What is the title of this document? Answer in one line.",
        attachments=[FIXTURES / "sample.pdf"],
        max_tokens=64,
    )
    return result.text.strip()


bench(main)
