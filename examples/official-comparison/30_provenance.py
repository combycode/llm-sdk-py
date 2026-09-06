"""Did this response really come from the model we asked for?

Returns evidence, not a boolean: which checks ran, which passed, and what the
provider actually claimed. A bare True/False would hide the reason, and the
reason is the entire point of asking.
"""

from _bench import api_key, bench, model

from combycode_llm_sdk import check_provenance


def main() -> str:
    report = check_provenance(
        model=model(),
        api_key=api_key(),
        prompt="Reply with exactly: OK",
    )
    return f"{report.verdict}:{len(report.checks)}"


bench(main)
