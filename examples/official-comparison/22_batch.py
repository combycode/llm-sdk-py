"""Many prompts at the provider's batch price, one call, results in order.

`batch()` blocks until the job finishes. The non-blocking form is
`submit_batch()`, which returns a job handle you can poll -- batch jobs can take
hours, so both shapes exist rather than forcing a thread.
"""

from _bench import api_key, bench, model

from combycode_llm_sdk import batch

#: Providers with a batch API. OpenRouter has none -- it is a router, and there
#: is no upstream job to submit to -- so the scenario is reported as not
#: applicable rather than as a failure. A permanently red cell teaches everyone
#: to skim past the failures, which is where a real one then hides.
BATCHLESS = {"openrouter"}


def main() -> str:
    provider = model().split("/", 1)[0]
    if provider in BATCHLESS:
        return f"n/a: {provider} has no batch API"

    results = batch(
        model=model(),
        api_key=api_key(),
        requests=[
            {"custom_id": "a", "prompt": "Reply with exactly: A"},
            {"custom_id": "b", "prompt": "Reply with exactly: B"},
        ],
        max_tokens=16,
    )
    ok = [r for r in results if r.success]
    return f"ok:{len(ok)}"


bench(main)
