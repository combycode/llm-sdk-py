"""Provider-hosted tools, named uniformly instead of per-provider.

`builtin_tools=["web_search"]` maps to each provider's own hosted tool. They are
OFF unless asked for: a hosted tool costs money and sends data out of the
process, so it is never implied.
"""

from _bench import api_key, bench, model

from combycode_llm_sdk import complete


def main() -> str:
    result = complete(
        model=model(),
        api_key=api_key(),
        prompt="Who won the 2022 FIFA World Cup? One word.",
        builtin_tools=["web_search"],
        max_tokens=256,
    )
    return result.text.strip()


bench(main)
