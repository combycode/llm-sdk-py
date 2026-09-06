"""Text to speech through the same media handle as image generation.

`voice` is a UNIFIED alias -- `neutral` / `warm` / `bright` / `deep` -- resolved
per provider (`alloy` on OpenAI, `Kore` on Google). A provider's own voice id is
passed through untouched, so a caller who knows their catalogue is never limited
to ours; but a corpus file that runs against every provider has to use the alias,
because "alloy" is not a voice Google has.
"""

from pathlib import Path

from _bench import api_key, bench, model

from combycode_llm_sdk import MediaOutput


def main() -> str:
    media = MediaOutput(
        model=model(),
        api_key=api_key(),
        dir=Path("./.media-out"),
    )
    clips = media.generate_audio(
        prompt="Hello from the unified SDK.",
        params={"voice": "neutral"},
    )
    return clips[0].meta.mime_type.split("/")[-1] if clips else ""


bench(main)
