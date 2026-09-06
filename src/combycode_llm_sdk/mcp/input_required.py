"""Multi-round-trip requests -- the 2026-07-28 replacement for the back-channel.

On the handshake wire a server that needs sampling, elicitation or roots PUSHES
a request at us in the middle of our call. At 2026-07-28 there is no
back-channel at all: the server instead RETURNS `resultType: "input_required"`
carrying the questions it needs answered, and the client re-issues the SAME call
with the answers plus the server's opaque `requestState`.

The point of this module is that both mechanisms feed the SAME callbacks. A
caller who wired up sampling once gets it on either wire without ever knowing
which is in play -- so `dispatch` here is the very same function that answers a
handshake-era server's pushed request.

Algorithm mirrors mcp-py 2.0.0 `client/_input_required.py`.
Transposed from `unified-library-ts/src/plugins/mcp/input-required.ts`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any

from .errors import McpError, McpErrorCode

#: Rounds before the driver gives up. Matches the TypeScript SDK's default; the
#: C# and Go SDKs use the same value as a hard constant.
DEFAULT_INPUT_REQUIRED_MAX_ROUNDS = 10

#: First sleep when a leg carries only `requestState` and no questions -- the
#: server saying "still working, ask again".
STATE_ONLY_BACKOFF_INITIAL_SECONDS = 0.05
#: Ceiling on that sleep, reached after three consecutive state-only legs.
STATE_ONLY_BACKOFF_CAP_SECONDS = 0.25

#: Answer one embedded question, through the client's sampling / elicitation /
#: roots handling.
InputRequestDispatcher = Callable[[str, Mapping[str, Any]], Any]

#: Re-issue the original call with the collected answers and the latest state.
InputRequiredRetry = Callable[[dict[str, Any] | None, str | None], Any]


def is_input_required(result: Any) -> bool:
    """Whether a result is the server asking for more, rather than an answer.

    The discriminant is `resultType`, and an ABSENT one MUST read as
    `"complete"`: servers on every earlier revision never send the field, so any
    other reading would turn every legacy result into a question.
    """
    return isinstance(result, Mapping) and result.get("resultType") == "input_required"


def _sleep(seconds: float, cancel: threading.Event | None) -> None:
    """Back off, but stay interruptible.

    A bare `time.sleep` here would hold a closing client open for the rest of
    the delay, and the whole reason for this pause is a server that is slow --
    exactly when someone is most likely to give up and close.
    """
    if cancel is not None:
        cancel.wait(seconds)
        return
    threading.Event().wait(seconds)


def run_input_required_driver(
    first: Any,
    *,
    dispatch: InputRequestDispatcher,
    retry: InputRequiredRetry,
    max_rounds: int | None = None,
    cancel: threading.Event | None = None,
) -> Any:
    """Drive an `input_required` result to a terminal one.

    Each round either answers every embedded question and retries with the
    responses, or -- when the server sent state but no questions -- backs off and
    retries empty.

    `requestState` is echoed back byte-exact and never inspected. It is the
    server's sealed continuation token, and a client that tried to interpret one
    would be reading a private format that is free to change.
    """
    rounds_allowed = DEFAULT_INPUT_REQUIRED_MAX_ROUNDS if max_rounds is None else max_rounds
    current = first
    rounds = 0
    state_only_delay = STATE_ONLY_BACKOFF_INITIAL_SECONDS

    while is_input_required(current):
        rounds += 1
        if rounds > rounds_allowed:
            raise McpError(
                f"MCP server returned input_required for more than {rounds_allowed} rounds. "
                "Raise input_required_max_rounds if the server legitimately needs more, or "
                "check the sampling/elicitation handler -- one that never satisfies the "
                "server loops forever.",
                code=McpErrorCode.INTERNAL_ERROR,
            )

        step = current if isinstance(current, Mapping) else {}
        requests = step.get("inputRequests")
        responses: dict[str, Any] | None = None

        if isinstance(requests, Mapping) and requests:
            # A real question resets the backoff: the server is not stalling,
            # it is waiting on us.
            state_only_delay = STATE_ONLY_BACKOFF_INITIAL_SECONDS
            responses = {}
            for key, request in requests.items():
                ask = request if isinstance(request, Mapping) else {}
                responses[str(key)] = dispatch(str(key), ask)
        else:
            # State-only leg: the server is still working. Sleeping before
            # asking again is what keeps a slow tool from becoming a spin loop
            # against the server.
            _sleep(state_only_delay, cancel)
            state_only_delay = min(state_only_delay * 2, STATE_ONLY_BACKOFF_CAP_SECONDS)

        request_state = step.get("requestState")
        current = retry(responses, request_state if isinstance(request_state, str) else None)

    return current


__all__ = [
    "DEFAULT_INPUT_REQUIRED_MAX_ROUNDS",
    "STATE_ONLY_BACKOFF_CAP_SECONDS",
    "STATE_ONLY_BACKOFF_INITIAL_SECONDS",
    "InputRequestDispatcher",
    "InputRequiredRetry",
    "is_input_required",
    "run_input_required_driver",
]
