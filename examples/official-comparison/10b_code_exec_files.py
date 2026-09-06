"""Hosted code execution, plus the files it produced.

`result.files` is always a list -- empty when the model produced none -- so
callers never branch on None for the ordinary case.

This is also the scenario behind the tool-constraint mechanic: Google refuses
code execution beside a PDF or video attachment, so the request goes without it
and the caller is TOLD through `result.warnings`, rather than silently losing a
capability they asked for.
"""

import os

from _bench import api_key, bench, model

from combycode_llm_sdk import complete

#: `samples-catalog.json` marks 10b unsupported for these: neither hosts a
#: code-execution sandbox of its own, so asserting a file output there measures
#: the provider's absence rather than this library's handling of one.
NO_CODE_EXECUTION = ("xai", "openrouter")


def main() -> str:
    provider = (os.environ.get("LLM_MODEL") or "").split("/")[0]
    if provider in NO_CODE_EXECUTION:
        return f"n/a: {provider} hosts no code execution"

    result = complete(
        model=model(),
        api_key=api_key(),
        # "Produce ... as a downloadable file" is the operative clause. Asked
        # only to "save them to fib.txt", a model writes one inside the sandbox
        # and emits no artifact, so `result.files` is honestly 0 and the cell
        # reads as a library failure to collect files that were never sent.
        prompt=(
            "Use code execution: compute the first 10 Fibonacci numbers and save "
            "them to fib.txt. Produce that file as a downloadable file and give "
            "me a link to it."
        ),
        builtin_tools=["code_interpreter"],
        max_tokens=8000,
    )
    for warning in result.warnings:
        print(f"warning: {warning.message}")
    return str(len(result.files))


bench(main)
