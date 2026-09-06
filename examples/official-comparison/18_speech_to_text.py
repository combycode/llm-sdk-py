"""A dedicated transcription model, not a chat model listening (that is 15).

`audio=` takes the same `str | Path | bytes` union as `attachments`, so there is
one rule for "how do I hand this library a file" across the whole surface.
"""

from pathlib import Path

from _bench import api_key, bench, model

from combycode_llm_sdk import transcribe

FIXTURES = Path(__file__).parent.parent / "fixtures"


def main() -> str:
    result = transcribe(
        model=model(),
        api_key=api_key(),
        audio=FIXTURES / "hello.wav",
    )
    return result.text.strip()


bench(main)
