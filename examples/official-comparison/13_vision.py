"""`attachments` accepts a path, a URL or raw bytes.

`pathlib.Path` is what Python users actually hold; accepting only `str` would
make every caller write `str(path)`. The SDK builds the right image part per
provider (base64 / inline_data / image_url) underneath.
"""

from pathlib import Path

from _bench import api_key, bench, model

from combycode_llm_sdk import complete

FIXTURES = Path(__file__).parent.parent / "fixtures"


def main() -> str:
    result = complete(
        model=model(),
        api_key=api_key(),
        prompt="What color is this image? One word.",
        attachments=[FIXTURES / "red.png"],
        max_tokens=32,
    )
    return result.text.strip()


bench(main)
