"""A reasoning mode almost nothing accepts, and the gate that keeps it from
becoming a 400.

`thinking={"mode": "between_tools"}` tells Anthropic to reason BETWEEN tool
calls rather than before the first answer. Measured 2026-09-29 against every
active Anthropic chat model: exactly one takes it -- `claude-sonnet-5.5` -- and
the other twelve answer `400 "thinking.type.between_tools" is not supported for
this model`, `claude-opus-5.5` among them. A deliberately invalid thinking type
is refused everywhere, so the field is read rather than tolerated: a 200 means
the model genuinely accepts it.

So the library asks the catalog first. Where the answer is not a recorded yes,
the mode is dropped, the request still goes out, and the caller is told -- which
is what Anthropic's own fallback middleware does with this value.

Deterministic: the transport records what we sent. No key, no network.
"""

from __future__ import annotations

import json
from typing import Any

from _check import check, report

from combycode_llm_sdk import LLM, Engine, TransportResponse

REPLY: dict[str, Any] = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [{"type": "text", "text": "OK"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 1},
}


def rig() -> tuple[Engine, list[dict[str, Any]], list[str]]:
    sent: list[dict[str, Any]] = []
    warnings: list[str] = []

    def transport(request: Any) -> Any:
        payload = getattr(request, "body", None)
        if isinstance(payload, (bytes, bytearray)):
            payload = json.loads(payload.decode("utf-8"))
        sent.append(payload if isinstance(payload, dict) else {})
        return TransportResponse(status=200, body=REPLY, headers={})

    engine = Engine(transport=transport, register_as_default=False, api_keys={"anthropic": "k"})
    engine.hooks.on(
        "onWarning",
        lambda ctx: warnings.append(str(dict(ctx).get("message")))
        if dict(ctx).get("code") == "request_adjusted"
        else None,
    )
    return engine, sent, warnings


# -- 1. the model that takes it gets it --------------------------------------
engine, sent, warnings = rig()
LLM(model="anthropic/claude-sonnet-5.5", engine=engine).complete(
    "Say OK.", max_tokens=8, reasoning={"mode": "between_tools"}
)
# Its own thinking type, not a variant of the adaptive shape: the two are
# siblings in Anthropic's own union, and one field cannot carry both.
check(
    sent[0].get("thinking") == {"type": "between_tools"},
    f"expected between_tools on the wire, got {sent[0].get('thinking')}",
)
check(not warnings, "nothing was adjusted, so nothing should be said")
accepted_wire = sent[0].get("thinking")

# -- 2. every other model gets a request that works, and an explanation ------
engine2, sent2, warnings2 = rig()
result = LLM(model="anthropic/claude-opus-5.5", engine=engine2).complete(
    "Say OK.", max_tokens=8, reasoning={"mode": "between_tools"}
)
check("thinking" not in sent2[0], "the mode must not reach a model that refuses it")
check(bool(result.text), "and the call must still succeed")
check(len(warnings2) == 1, "dropping it silently is the failure this prevents")
check(
    bool(warnings2) and "claude-sonnet-5.5" in warnings2[0],
    "the warning should name the model that does take it",
)

# -- 3. the catalog is where the answer lives --------------------------------
models = [m for m in engine.catalog.list() if m["provider"] == "anthropic" and m["type"] == "chat"]
accepted = [m["model"] for m in models if (m.get("reasoning") or {}).get("betweenTools") is True]
check(accepted == ["claude-sonnet-5.5"], f"expected exactly sonnet-5.5 marked, got {accepted}")

# And the mark does NOT spread across the family. `claude-sonnet` is one family
# from 4.5 to 5.5, family annotations are inherited, and three of those four were
# measured refusing it -- a donated YES would be indistinguishable from a
# measured one.
family = [m["model"] for m in models if m.get("family") == "claude-sonnet"]
check(len(family) > 1, "expected the sonnet family to have several members")

report(
    accepted_wire=accepted_wire,
    dropped_for="claude-opus-5.5",
    warned=len(warnings2),
    marked=accepted,
    family_size=len(family),
)
