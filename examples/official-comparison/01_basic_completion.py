from _bench import api_key, bench, model

from combycode_llm_sdk import complete


def main() -> str:
    result = complete(
        model=model(),
        api_key=api_key(),
        prompt="Reply with exactly: OK",
        max_tokens=16,
    )
    return result.text.strip()


bench(main)
