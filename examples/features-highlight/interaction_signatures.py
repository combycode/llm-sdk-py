"""A turn that hands state back, and a failure that says why.

Google Interactions returns a `thought` step carrying nothing but a `signature`.
It is not decoration: measured on 2026-09-29 against `gemini-3.1-flash-lite`,
echoing the step on the next turn is accepted, and echoing it with the signature
corrupted is refused `400 Corrupted thought signature` -- the server reads it.
This library used to drop it on both paths, so every multi-turn Interactions
conversation silently discarded the model's own reasoning continuity.

The carrier is `response["signatures"]` -> `message["origin"]["signatures"]` ->
the next request. It is OPAQUE (nothing here reads it) and PROVIDER-BOUND (only
the adapter that minted it sends it back), which is why building history with
`assistant_message()` matters rather than assembling messages by hand.

Deterministic: the bodies below are the shapes measured live, replayed through
the real adapter. No key, no network.
"""

from __future__ import annotations

import json
from typing import Any

from _check import check, report

from combycode_llm_sdk import LLM, Engine, TransportResponse

SIGNATURE = "EnMKcQFpFH0TSvkjoKwOEghRRZdGvC3ICA0FZOoJaq4F2i"

INTERACTION: dict[str, Any] = {
    "id": "int_1",
    "object": "interaction",
    "status": "completed",
    "model": "gemini-3.1-flash-lite",
    "steps": [
        # The whole step. No content, no text -- a signature and nothing else.
        {"type": "thought", "signature": SIGNATURE},
        {"type": "model_output", "content": [{"type": "text", "text": "OK"}]},
    ],
    "usage": {"prompt_tokens": 3, "candidates_tokens": 1, "total_tokens": 4},
}

FAILED: dict[str, Any] = {
    "id": "int_2",
    "object": "interaction",
    "status": "failed",
    "model": "gemini-3.1-flash-lite",
    "steps": [],
    "errors": [
        {"code": "https://developers.google.com/errors/internal", "message": "The model failed."}
    ],
    "usage": {"prompt_tokens": 3, "candidates_tokens": 0, "total_tokens": 3},
}


def replay(body: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """Answers with one recorded body and records what we sent."""
    sent: list[dict[str, Any]] = []

    def transport(request: Any) -> Any:
        payload = getattr(request, "body", None)
        if isinstance(payload, (bytes, bytearray)):
            payload = json.loads(payload.decode("utf-8"))
        sent.append(payload if isinstance(payload, dict) else {})
        return TransportResponse(status=200, body=body, headers={})

    return transport, sent


def llm_for(body: dict[str, Any]) -> tuple[LLM, list[dict[str, Any]], Engine]:
    transport, sent = replay(body)
    engine = Engine(transport=transport, register_as_default=False, api_keys={"google": "k"})
    return (
        LLM(model="google/gemini-3.1-flash-lite", engine=engine, api="interactions"),
        sent,
        engine,
    )


# -- 1. the signature survives the parse, verbatim ---------------------------
llm, sent, engine = llm_for(INTERACTION)
first = llm.complete("Think, then say OK.", max_tokens=16, stateful=False)
signed = first.signatures
check(isinstance(signed, list) and len(signed) == 1, "the signed step must survive the parse")
check(bool(signed) and signed[0]["signature"] == SIGNATURE, "and survive it unchanged")

# -- 2. and reaches the next request, in the position it arrived in ----------
assistant = first.assistant_message()
check(bool(assistant["origin"].get("signatures")), "assistant_message() carries it into history")

history = [
    {"role": "user", "content": "Think, then say OK."},
    assistant,
    {"role": "user", "content": "Now say DONE."},
]
llm.complete(history, max_tokens=16, stateful=False)
types = [i["type"] for i in sent[1]["input"]]
# Before the model_output it preceded -- which is where the API takes it back.
check(
    types == ["user_input", "thought", "model_output", "user_input"],
    f"the signed step must ride in its own position, got {types}",
)
echoed = " -> ".join(types)

# -- 3. never another provider's signature -----------------------------------
# Provider-bound by contract: a blob minted elsewhere is meaningless here.
llm2, sent2, _ = llm_for(INTERACTION)
foreign = {
    "role": "assistant",
    "content": [{"type": "text", "text": "OK"}],
    "origin": {
        "provider": "openai",
        "model": "gpt-5.4-nano",
        "signatures": [{"type": "thought", "signature": "not-ours"}],
    },
}
llm2.complete([{"role": "user", "content": "hi"}, foreign], max_tokens=16, stateful=False)
foreign_types = [i["type"] for i in sent2[0]["input"]]
check("thought" not in foreign_types, "another provider's signature must never go out")

# -- 4. a failed interaction says why ----------------------------------------
# It used to arrive as `finishReason: "error"` and nothing else: an empty
# answer, no exception to catch, and no way to tell a content refusal from a
# platform fault.
llm3, _, _ = llm_for(FAILED)
failed = llm3.complete("Say OK.", max_tokens=16, stateful=False)
check(failed.finish_reason == "error", "a failed interaction must not read as a clean finish")
error = failed.error or {}
check(bool(error.get("message")), "and it must say what happened")

report(
    echoed=echoed,
    foreign_dropped=" -> ".join(foreign_types),
    failed_code=error.get("code"),
    failed_message=error.get("message"),
)
