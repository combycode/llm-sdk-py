"""Documents the PROVIDER indexes, asked about through a tool.

Create a corpus, add a document only it could answer from, wait for indexing,
then ask. `as_tool()` goes into the same `tools=` list as everything else -- one
list, one concept -- and the provider runs the search itself.

`wait_indexed()` is not optional politeness: indexing is asynchronous on every
provider, and asking early returns an answer built on nothing, which reads
exactly like a document that does not contain what you asked for.

Closing the block deletes the corpus. `Corpus.create()` made it, so the `with`
owns it, and a hosted store left behind is a bill nobody is reading.
"""

import os

from _bench import api_key, bench, model

from combycode_llm_sdk import Corpus, complete

# A fact no model can know and no search can find. If the answer comes back, it
# came from the document.
DOCUMENT = (
    "Internal maintenance note, revision 7.\n"
    "The Zorblatt regulator coolant pump is part number 4217.\n"
    "It is replaced every 90 days by the night shift.\n"
)


# Providers that host no corpus API at all. Not a gap in the library: there is
# nothing to call, and reporting a failure every run would hide the real ones.
NO_CORPUS_API = ("anthropic", "openrouter")


def main() -> str:
    provider = (os.environ.get("LLM_MODEL") or "").split("/")[0]
    if provider in NO_CORPUS_API:
        return f"n/a: {provider} hosts no corpus API"
    management_key = os.environ.get("XAI_MANAGEMENT_API_KEY")

    with Corpus.create(name="sdk-retrieval-demo", model=model(), api_key=api_key(),
                       management_api_key=management_key) as corpus:
        corpus.add_document(DOCUMENT, label="maintenance-note.txt")
        corpus.wait_indexed()

        result = complete(
            model=model(),
            api_key=api_key(),
            prompt="What is the part number of the Zorblatt regulator coolant pump?",
            tools=[corpus.as_tool()],
            max_tokens=256,
        )
    return result.text.strip()


bench(main)
