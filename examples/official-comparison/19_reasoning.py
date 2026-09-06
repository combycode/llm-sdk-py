"""Reasoning effort is one option; the thinking text comes back separately.

`result.thinking` is None when a model did not think or does not expose it --
distinguishable from "" (it thought and returned nothing), which matters when
you are deciding whether a model actually supports the feature.
"""

from _bench import api_key, bench, model

from combycode_llm_sdk import LLM


def main() -> str:
    llm = LLM(model=model(), api_key=api_key())
    result = llm.complete(
        # The shared catalog's scenario 19, which the TypeScript twin also asks.
        # It used to pose the bat-and-ball problem, and that is a COGNITIVE TRAP
        # rather than a reasoning check: measured live on 2026-09-06 the same
        # model at the same effort answered 0.05, then 0.10, then 0.05 -- so the
        # cell reported a library defect roughly half the time and told nobody
        # anything about reasoning when it passed.
        "If 2x=10 what is x? Reply with just the number.",
        reasoning={"effort": "low"},
        max_tokens=2048,
    )
    return result.text.strip()


bench(main)
