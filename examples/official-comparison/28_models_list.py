"""Two different questions, two different functions.

`list_models()` returns our curated catalog -- pricing, capabilities, context
window -- and needs no network. `list_models_live()` asks the provider what it
is serving right now. Conflating them would mean either a needless HTTP call or
a silently poorer answer.
"""

from _bench import api_key, bench, model

from combycode_llm_sdk import list_models_live


def main() -> str:
    provider = model().split("/")[0]
    ids = list_models_live(provider=provider, api_key=api_key())
    return f"found:{len(ids)}" if ids else "none"


bench(main)
