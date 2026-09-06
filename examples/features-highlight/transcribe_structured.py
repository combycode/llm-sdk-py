"""`transcribe()` always returns text. The structured extras are optional -- and gated.

Segments, word timings and language detection are not universally available, and
on OpenAI each is gated to a DIFFERENT model. Returning empty lists for all of
them would make "this model cannot do it" indistinguishable from "there were no
words", which is the difference between a limitation and a result.

So the extras are None when unavailable and a list when supported, and the SDK
warns rather than silently dropping a request for something that cannot work.

Deterministic: stub transport.
"""

from _check import check, report

from combycode_llm_sdk import TransportResponse, transcribe

BODY = {
    "text": "hello world",
    "language": "en",
    "segments": [{"id": 0, "start": 0.0, "end": 1.2, "text": "hello world"}],
    "words": [
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.6, "end": 1.2},
    ],
}


def stub(request):
    return TransportResponse(status=200, body=BODY)


result = transcribe(
    model="openai/whisper-1",
    api_key="k",
    audio=b"\x00\x01",
    timestamps="word",
    transport=stub,
)

# Text is the guarantee.
check(result.text == "hello world", "text is always present")

# The extras are typed values, not raw dicts.
check(result.language == "en", "language should be surfaced when detected")
check(result.segments is not None and len(result.segments) == 1, "segments should parse")
check(result.words is not None and len(result.words) == 2, "word timings should parse")
check(result.words[0].start == 0.0, "word timings carry real numbers")

# Not requested is None, not [] -- absence and emptiness are different answers.
plain = transcribe(model="openai/whisper-1", api_key="k", audio=b"\x00", transport=stub)
check(plain.text == "hello world", "a plain transcription still returns text")

report(text=result.text, language=result.language, words=len(result.words))
