"""Server-state decision -- the unified "send id vs resend history" brain.

Transposed from `unified-library-ts/src/llm/server-state.ts`.

Several providers keep conversation state server-side (OpenAI Responses
`previous_response_id`, xAI, Google Interactions `previous_interaction_id`). When
the prior assistant turn carries a server id we produced, we can send just that
id plus the new turn instead of the whole transcript.

Safe by default -- reuse only when ALL hold:

- the caller did not opt out (`stateful` is not False),
- the prior id was produced by the SAME provider (cross-provider ids are
  meaningless elsewhere -- history stays portable),
- the provider/model supports it (catalog),
- it is within the retention TTL (catalog duration),
- the model matches, OR the provider is not model-bound (catalog).

Otherwise we fall back to resending full history, which is always correct.
"""

from __future__ import annotations

from typing import Any

from ..util.duration import parse_duration_or_none
from .types.messages import Message

#: `interface ServerStateDecision` (server-state.ts:22):
#: `{previousResponseId?, messages}`. `previousResponseId` is the
#: provider-agnostic id to continue from (the adapter maps it to its own param);
#: `messages` is what to actually send -- trimmed to the new turn(s) when
#: chaining, else the full list.
ServerStateDecision = dict[str, Any]


def resolve_server_state(
    *,
    messages: list[Message],
    provider: str,
    model: str,
    catalog: Any,
    stateful: bool,
    now: float,
) -> ServerStateDecision:
    """Decide whether to chain server-side, and what to send.

    Every early return is `{"messages": messages}` -- the full transcript, which
    is always correct. Chaining is the exception that has to earn its way past
    all five checks.
    """
    if not stateful:
        return {"messages": messages}

    # Most recent assistant turn that carries a server-state id.
    idx = -1
    for i in range(len(messages) - 1, -1, -1):
        origin = messages[i].get("origin")
        if messages[i].get("role") == "assistant" and origin and origin.get("serverStateId"):
            idx = i
            break
    if idx == -1:
        return {"messages": messages}

    origin = messages[idx].get("origin")
    if not origin or not origin.get("serverStateId"):
        return {"messages": messages}
    if origin.get("provider") != provider:
        return {"messages": messages}  # foreign id -- ignore, stay portable
    if not catalog.supports_previous_response_id(provider, model):
        return {"messages": messages}

    # TTL: if expired, the server has forgotten -- resend history.
    ttl_ms = parse_duration_or_none(catalog.get_state_retention(provider, model))
    created_at = messages[idx].get("createdAt")
    if ttl_ms is not None and created_at is not None and now - created_at > ttl_ms:
        return {"messages": messages}

    # Model-bound providers lose context across model swaps -- resend history.
    if (
        catalog.is_state_model_bound(provider, model)
        and origin.get("model")
        and origin.get("model") != model
    ):
        return {"messages": messages}

    # Continue server-side: send only the turns AFTER the stored interaction.
    trimmed = messages[idx + 1 :]
    if not trimmed:
        return {"messages": messages}  # nothing new to add -- just resend
    return {"previousResponseId": origin["serverStateId"], "messages": trimmed}


__all__ = ["ServerStateDecision", "resolve_server_state"]
