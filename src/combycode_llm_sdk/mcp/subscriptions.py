"""`subscriptions/listen` -- one stream for every change the server announces.

At 2026-07-28 (SEP-2575) the per-resource `resources/subscribe` RPC and the
standalone notification channel are both replaced by a single long-lived
`subscriptions/listen` request. Three properties of that design decide the shape
of everything here:

- **Every kind is opt-in.** The client names what it wants and the server MUST
  NOT send a kind that was not asked for.
- **The acknowledgement can be NARROWER than the request.** The server answers
  with the subset it actually honoured, so "I asked for it" never implies "I
  will receive it" -- read `honored` / `is_honored()` rather than assuming.
- **Events are level triggers.** "This changed, re-fetch if you care." They
  carry no payload beyond the fact of the change, so two identical events in a
  row collapse safely.

Every frame is stamped with the listen request's id under
`io.modelcontextprotocol/subscriptionId`, which is how a frame is attributed to
one subscription when several are open.

Transposed from `unified-library-ts/src/plugins/mcp/subscriptions.ts`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self

#: Where a frame carries the id of the subscription it belongs to.
MCP_SUBSCRIPTION_ID_META_KEY = "io.modelcontextprotocol/subscriptionId"

#: The ack frame carrying the subset the server actually honoured.
MCP_SUBSCRIPTIONS_ACKNOWLEDGED = "notifications/subscriptions/acknowledged"

#: The notification methods that ride a listen stream at 2026-07-28.
MCP_LISTEN_STREAM_METHODS = (
    "notifications/tools/list_changed",
    "notifications/prompts/list_changed",
    "notifications/resources/list_changed",
    "notifications/resources/updated",
)

McpEventType = Literal[
    "tools_list_changed",
    "prompts_list_changed",
    "resources_list_changed",
    "resource_updated",
]


@dataclass(frozen=True)
class McpSubscriptionFilter:
    """The notification kinds a client opts into.

    snake_case here and camelCase on the wire, like every other public shape in
    this library -- `to_wire()` and `of_wire()` are the only two places that
    know the difference.
    """

    tools_list_changed: bool = False
    prompts_list_changed: bool = False
    resources_list_changed: bool = False
    #: Resource URIs to watch. The replacement for `resources/subscribe`.
    resource_subscriptions: Sequence[str] = ()

    def __post_init__(self) -> None:
        # Normalised to a tuple so equality is by CONTENT. Without it a filter
        # built from a list never equals the same filter read back off the wire,
        # which is exactly the comparison a caller makes to check what was
        # honoured.
        object.__setattr__(self, "resource_subscriptions", tuple(self.resource_subscriptions))

    def to_wire(self) -> dict[str, Any]:
        """The filter as the server expects it.

        Only what was asked for is sent. A filter that spelled out
        `"toolsListChanged": false` would be asking the server to record a
        preference it has no way to act on, and on a strict server an unasked-for
        key is one more thing that can be rejected.
        """
        wire: dict[str, Any] = {}
        if self.tools_list_changed:
            wire["toolsListChanged"] = True
        if self.prompts_list_changed:
            wire["promptsListChanged"] = True
        if self.resources_list_changed:
            wire["resourcesListChanged"] = True
        if self.resource_subscriptions:
            wire["resourceSubscriptions"] = list(self.resource_subscriptions)
        return wire

    @staticmethod
    def of_wire(wire: Mapping[str, Any]) -> McpSubscriptionFilter:
        """A filter the server sent back, as the honoured subset."""
        uris = wire.get("resourceSubscriptions")
        return McpSubscriptionFilter(
            tools_list_changed=wire.get("toolsListChanged") is True,
            prompts_list_changed=wire.get("promptsListChanged") is True,
            resources_list_changed=wire.get("resourcesListChanged") is True,
            resource_subscriptions=tuple(u for u in uris if isinstance(u, str))
            if isinstance(uris, Sequence) and not isinstance(uris, (str, bytes))
            else (),
        )


@dataclass(frozen=True)
class McpServerEvent:
    """A change the server announced. A level trigger: re-fetch if you care."""

    type: McpEventType
    #: Set only on `resource_updated`. Which document went stale.
    uri: str | None = None


@dataclass(frozen=True)
class McpSubscriptionEnd:
    """Why a stream stopped.

    A distinct object rather than a bare error, because "the server tore it
    down cleanly" and "it is still running" are different states and both would
    otherwise be `None`.
    """

    error: BaseException | None = None


def event_from_wire(method: str, params: Any) -> McpServerEvent | None:
    """The event a raw frame announces, or None when it announces none."""
    if method == "notifications/tools/list_changed":
        return McpServerEvent("tools_list_changed")
    if method == "notifications/prompts/list_changed":
        return McpServerEvent("prompts_list_changed")
    if method == "notifications/resources/list_changed":
        return McpServerEvent("resources_list_changed")
    if method == "notifications/resources/updated":
        uri = params.get("uri") if isinstance(params, Mapping) else None
        return McpServerEvent("resource_updated", uri=uri) if isinstance(uri, str) else None
    return None


def subscription_id_from(params: Any) -> str | int | None:
    """The subscription a frame belongs to, or None when it is not a listen frame."""
    meta = params.get("_meta") if isinstance(params, Mapping) else None
    if not isinstance(meta, Mapping):
        return None
    ident = meta.get(MCP_SUBSCRIPTION_ID_META_KEY)
    # `bool` is an `int` in Python and `True` is not an id. Excluded explicitly
    # so a malformed frame cannot be attributed to subscription 1.
    if isinstance(ident, bool):
        return None
    return ident if isinstance(ident, (str, int)) else None


class McpSubscription:
    """One live subscription: what the server honoured, plus event delivery."""

    def __init__(
        self,
        subscription_id: str | int,
        requested: McpSubscriptionFilter,
        on_event: Callable[[McpServerEvent], None],
        on_close: Callable[[], None],
    ) -> None:
        self.id = subscription_id
        self.requested = requested
        #: The subset the server agreed to send. None until the ack arrives --
        #: and it can be NARROWER than what was requested.
        self.honored: McpSubscriptionFilter | None = None
        self._on_event = on_event
        self._on_close = on_close
        self._closed = False
        self._end: McpSubscriptionEnd | None = None

    # -- state ---------------------------------------------------------------

    @property
    def active(self) -> bool:
        """Whether this can still deliver."""
        return not self._closed

    @property
    def ended(self) -> McpSubscriptionEnd | None:
        """How the stream finished, or None while it is still running.

        Worth reading: a subscription that stopped delivering is otherwise
        indistinguishable from one where nothing has changed yet.
        """
        return self._end

    def is_honored(self, kind: str) -> bool:
        """Whether the server acknowledged one kind.

        No separate branch for "not acknowledged yet": `honored` is None until
        the ack arrives, and `getattr(None, kind, None)` is already the same
        "no" as a filter that did not name this kind. A guard for it would look
        load-bearing and never be reached.

        A watched-URI list counts as honoured only when it is NON-EMPTY -- the
        server agreeing to watch nothing is not agreeing to watch anything.
        """
        value = getattr(self.honored, kind, None)
        if isinstance(value, bool):
            return value
        return bool(value) if isinstance(value, Sequence) else False

    # -- delivery ------------------------------------------------------------

    def handle_frame(self, method: str, params: Any) -> bool:
        """Feed one raw stream frame. Whether it belonged to this subscription."""
        if self._closed:
            return False
        if subscription_id_from(params) != self.id:
            return False

        if method == MCP_SUBSCRIPTIONS_ACKNOWLEDGED:
            filter_wire = params.get("notifications") if isinstance(params, Mapping) else None
            # A MISSING filter is malformed, not an empty one. Reading it as
            # empty would silently report that the server honoured nothing,
            # which a caller cannot tell from a server that really did.
            if isinstance(filter_wire, Mapping):
                self.honored = McpSubscriptionFilter.of_wire(filter_wire)
            return True

        event = event_from_wire(method, params)
        if event is None:
            return False
        self._on_event(event)
        return True

    def mark_ended(self, error: BaseException | None = None) -> None:
        """Called by the transport when the underlying stream finishes."""
        if self._closed:
            return
        self._end = McpSubscriptionEnd(error=error)
        self._closed = True
        self._on_close()

    def close(self) -> None:
        """Stop listening. Idempotent, so a `finally` can call it blindly."""
        if self._closed:
            return
        self._closed = True
        self._end = McpSubscriptionEnd()
        self._on_close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "active" if self.active else "ended"
        return f"<McpSubscription {self.id} {state}>"


__all__ = [
    "MCP_LISTEN_STREAM_METHODS",
    "MCP_SUBSCRIPTIONS_ACKNOWLEDGED",
    "MCP_SUBSCRIPTION_ID_META_KEY",
    "McpEventType",
    "McpServerEvent",
    "McpSubscription",
    "McpSubscriptionEnd",
    "McpSubscriptionFilter",
    "event_from_wire",
    "subscription_id_from",
]
