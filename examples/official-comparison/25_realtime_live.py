"""A live session: send a turn over a socket, read events back.

The same event objects and the same `event.type` as `LLM.stream()`, and the same
iterator protocol -- `for event in session` is the loop a Python caller already
writes. A realtime session is a different transport, not a different idea.

One hint differs per provider, and it is the provider's rule rather than ours:
Gemini Live models are AUDIO-NATIVE and reject a TEXT-only session outright
("the requested combination of response modalities (TEXT) is not supported"),
while OpenAI realtime answers in text. So each is asked for what it does, and
the session API is identical either way -- audio arrives as `audio` events
carrying decoded bytes, text as `text` events.

A context manager because it holds a socket. `turn_end` is the signal to stop
reading: the stream ends when the connection does, not when the model stops
talking, so waiting for the end of iteration would hang.
"""

import os

from _bench import api_key, bench, model

from combycode_llm_sdk import Realtime

PROVIDER = (os.environ.get("LLM_MODEL") or "").split("/")[0]
MODALITIES = ["audio"] if PROVIDER == "google" else ["text"]


def main() -> str:
    with Realtime(model=model(), api_key=api_key(), modalities=MODALITIES,
                  instructions="Answer in as few words as possible.") as session:
        session.send("Name the capital of France. One word.")

        text = ""
        audio_bytes = 0
        for event in session:
            if event.type == "text":
                text += event.text
            elif event.type == "audio":
                audio_bytes += len(event.audio)
            elif event.type == "error":
                # Raised, not returned: an error string would read as an answer
                # and the scenario would report a pass for a session that failed.
                raise RuntimeError(event.message)
            elif event.type == "turn_end":
                break

    return text.strip() or (f"audio:{audio_bytes}" if audio_bytes else "empty")


bench(main)
