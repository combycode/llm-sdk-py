"""Sources arrive DURING a stream, not after it.

A finished answer's sources are on `completion.citations`. Streaming needs them
earlier: a UI rendering footnotes as the text lands cannot wait for the turn to
end. So `stream()` yields a `CitationEvent` per source.

A citation does NOT arrive when the search runs. Providers search early and cite
while writing, so citation events interleave with text deltas -- the ordering
below is the real one, taken off a live Anthropic stream.

The four wire shapes agree on nothing, which is why unifying them is worth
doing:

    messages     content_block_delta -> delta.type 'citations_delta' (+cited_text)
    responses    response.output_text.annotation.added   (OpenAI and xAI)
    completions  choices[].delta.annotations[]           (OpenRouter `:online`)
    generate     a LATE chunk whose groundingMetadata is populated

Deterministic -- no key, no network. The SSE below was RECORDED from a live
Anthropic stream and replayed through the same parser a real call uses, so what
this asserts is what a provider actually sends.
"""

import json

from _check import check, report

from combycode_llm_sdk.streaming import parse_stream

# Recorded off a live stream, trimmed to the fields the parser reads.
RECORDED = [
    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Python "}},
    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "3.14.7"}},
    {"type": "content_block_delta", "delta": {
        "type": "citations_delta",
        "citation": {"type": "web_search_result_location",
                     "url": "https://www.python.org/downloads/",
                     "title": "Download Python | Python.org",
                     "cited_text": "Python 3.14.7 Aug. 5, 2026"}}},
    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " is current."}},
]

lines = [f"data: {json.dumps(chunk)}" for chunk in RECORDED]

order: list[str] = []
citations = []
text = ""
for event in parse_stream("messages", iter(lines)):
    order.append(event.type)
    if event.type == "text":
        text += event.text
    elif event.type == "citation":
        citations.append(event.citation)

# The source lands BETWEEN text deltas -- what makes live footnotes possible.
check(order == ["text", "text", "citation", "text"], f"unexpected event order: {order}")
check(text == "Python 3.14.7 is current.", "text deltas must be unaffected")
check(len(citations) == 1, "the cited source must be reported")

cite = citations[0]
check(cite.url == "https://www.python.org/downloads/", "url must survive")
# `text` is the passage the source supports. Anthropic is the only provider that
# reports one; elsewhere it is empty rather than filled in from the answer.
check(cite.text == "Python 3.14.7 Aug. 5, 2026", "the cited passage must survive")

report(order=",".join(order), text=text,
       citation={"url": cite.url, "title": cite.title, "passage": cite.text})
