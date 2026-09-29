"""Asking WHY the prompt cache missed, not just whether it did.

`usage.cached_tokens` reports how much was reused. On a long system prompt the
useful question is the other one -- what broke the prefix -- and two providers
answer it: Anthropic `diagnostics.cache_miss_reason`, OpenAI
`prompt_cache_options` -> `prompt_cache_diagnostics`. One option asks; one
optional response field carries the reply.

The shapes are NOT symmetric, and the asymmetry is the thing to see here: OpenAI
reports a hit as `status="hit"`, and Anthropic reports it by saying nothing at
all. So this library never turns Anthropic's silence into a hit -- that would
publish an inference as the provider's answer.

Deterministic: the bodies below were MEASURED on 2026-09-29 (claude-haiku-4-5 on
GA /v1/messages, gpt-5.6-luna on /v1/responses) and are replayed through the real
adapters. No key, no network.
"""

from __future__ import annotations

import json
from typing import Any

from _check import check, report

from combycode_llm_sdk import LLM, Engine, TransportResponse
from combycode_llm_sdk.streaming import parse_stream

ANTHROPIC_BASE: dict[str, Any] = {
    "id": "msg_01x",
    "model": "claude-haiku-4-5-20251001",
    "role": "assistant",
    "type": "message",
    "content": [{"type": "text", "text": "OK"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 9, "output_tokens": 2, "cache_read_input_tokens": 9901},
}

OPENAI_BASE: dict[str, Any] = {
    "id": "resp_01x",
    "model": "gpt-5.6-luna",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "OK"}],
        }
    ],
    "usage": {
        "input_tokens": 10,
        "output_tokens": 2,
        "input_tokens_details": {"cached_tokens": 9910},
    },
}

GOOGLE_BASE: dict[str, Any] = {
    "candidates": [{"content": {"parts": [{"text": "OK"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1, "totalTokenCount": 6},
}


def replay(body: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """A transport that answers with one recorded body, recording what we sent."""
    sent: list[dict[str, Any]] = []

    def transport(request: Any) -> Any:
        payload = getattr(request, "body", None)
        if isinstance(payload, (bytes, bytearray)):
            payload = json.loads(payload.decode("utf-8"))
        sent.append(payload if isinstance(payload, dict) else {})
        return TransportResponse(status=200, body=body, headers={})

    return transport, sent


def run(model: str, body: dict[str, Any], **options: Any) -> tuple[Any, list[dict[str, Any]], list[str]]:
    transport, sent = replay(body)
    provider = model.split("/", 1)[0]
    engine = Engine(
        transport=transport, register_as_default=False, api_keys={provider: "k"}
    )
    adjusted: list[str] = []
    engine.hooks.on(
        "onWarning",
        lambda ctx: adjusted.append(str(dict(ctx).get("message")))
        if dict(ctx).get("code") == "request_adjusted"
        else None,
    )
    llm = LLM(model=model, engine=engine)
    result = llm.complete("Say OK.", **options)
    return result, sent, adjusted


# -- 1. Anthropic: a miss, in Anthropic's own vocabulary ----------------------
miss_body = dict(ANTHROPIC_BASE)
miss_body["diagnostics"] = {
    "cache_miss_reason": {"type": "system_changed", "cache_missed_input_tokens": 9197}
}
res, sent, _ = run(
    "anthropic/claude-haiku-4.5",
    miss_body,
    max_tokens=8,
    cache_diagnostics={"compareWith": "msg_01previous"},
)
diag = res.cache_diagnostics
check(diag is not None and diag["status"] == "miss", "a reported miss must read as a miss")
check(diag is not None and diag["reason"] == "system_changed", "the provider word must survive")
check(diag is not None and diag["missedTokens"] == 9197, "the token cost must be carried")
check(
    sent[0].get("diagnostics") == {"previous_message_id": "msg_01previous"},
    "the comparison id must go out as diagnostics.previous_message_id",
)

# -- 2. Anthropic: a HIT is silence, and stays silence ------------------------
# Measured: a request whose prefix WAS reused returns `diagnostics: null` -- byte
# for byte what an undiagnosed request returns. There is no hit variant.
hit_body = dict(ANTHROPIC_BASE)
hit_body["diagnostics"] = None
res, sent, _ = run(
    "anthropic/claude-haiku-4.5", hit_body, max_tokens=8, cache_diagnostics={}
)
check(
    res.cache_diagnostics is None,
    "Anthropic said nothing; inventing a hit would be our guess, not its answer",
)
check(res.usage.cached_tokens == 9901, "usage is what answers whether the cache was used")
# Opting in with nothing to compare sends an explicit null: omitting the field
# opts out entirely.
check(
    sent[0].get("diagnostics") == {"previous_message_id": None},
    "opting in with no comparison must send previous_message_id: null",
)

# -- 3. OpenAI: every outcome has a status, including the hit -----------------
oai_hit = dict(OPENAI_BASE)
oai_hit["prompt_cache_diagnostics"] = {"type": "cache_hit"}
res, sent, _ = run(
    "openai/gpt-5.6-luna",
    oai_hit,
    max_tokens=16,
    cache_diagnostics={"compareWith": "resp_01previous"},
)
check(
    res.cache_diagnostics is not None and res.cache_diagnostics["status"] == "hit",
    "OpenAI reports a hit and we must carry it",
)
check(
    sent[0].get("prompt_cache_options", {}).get("comparison_response_id") == "resp_01previous",
    "the comparison id must go out inside prompt_cache_options",
)

# -- 4. An unknown comparison id is a 200, not an error -----------------------
# Both providers answer 200 with a "not found" status, so a stored id can be
# passed without guarding its age -- the call still succeeds.
oai_stale = dict(OPENAI_BASE)
oai_stale["prompt_cache_diagnostics"] = {"type": "comparison_response_not_found"}
res, _, _ = run(
    "openai/gpt-5.6-luna",
    oai_stale,
    max_tokens=16,
    cache_diagnostics={"compareWith": "resp_longGone"},
)
check(
    res.cache_diagnostics is not None
    and res.cache_diagnostics["status"] == "comparison_not_found",
    "the two provider spellings of not-found unify onto one status",
)
check(bool(res.text), "the completion itself must still be there")

# -- 5. Asking where no field exists is said out loud -------------------------
res, _, adjusted = run(
    "google/gemini-3.1-flash",
    GOOGLE_BASE,
    max_tokens=8,
    cache_diagnostics={"compareWith": "whatever"},
)
check(len(adjusted) == 1, "a provider with no field for it must say so, exactly once")
check(res.cache_diagnostics is None, "and nothing must be invented in the response")

# -- 6. streaming answers the same question ----------------------------------
# Both providers send the diagnosis in the stream as well (Anthropic on
# `message_start`, before a token is generated), so which call style you used
# must not change the answer.
STREAMED = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_01x",
            "usage": {"input_tokens": 9, "output_tokens": 0},
            "diagnostics": {
                "cache_miss_reason": {"type": "system_changed", "cache_missed_input_tokens": 9197}
            },
        },
    },
    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "OK"}},
]

streamed = [
    event
    for event in parse_stream("messages", iter(f"data: {json.dumps(c)}" for c in STREAMED))
    if event.type == "cache_diagnostics"
]
check(len(streamed) == 1, "a streamed diagnosis must arrive as its own event")
check(
    bool(streamed) and streamed[0].diagnostics["reason"] == "system_changed",
    "and carry the same reason the buffered path reports",
)

report(
    anthropic_miss="system_changed",
    anthropic_hit_is_silence=True,
    openai_hit="hit",
    openai_stale_id="comparison_not_found",
    google_warned=len(adjusted),
    streamed=streamed[0].diagnostics["reason"] if streamed else None,
)
