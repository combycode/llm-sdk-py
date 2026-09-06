"""Audio input to a chat model -- still `attachments`, not a special call.

Distinct from 18_speech_to_text.py, which uses a dedicated transcription model.
Here a multimodal chat model listens and answers a question about the audio.
"""

from pathlib import Path

from _bench import api_key, bench, model

from combycode_llm_sdk import complete

FIXTURES = Path(__file__).parent.parent / "fixtures"


def main() -> str:
    result = complete(
        model=model(),
        api_key=api_key(),
        prompt="What word is spoken in this audio? One word.",
        attachments=[FIXTURES / "hello.wav"],
        # 256, not 32. A model that opens with "The word spoken in the audio
        # is" has spent the smaller budget before reaching the word, and the
        # cell then reports a transcription failure that is really a truncation.
        max_tokens=256,
    )
    return result.text.strip()


bench(main)
