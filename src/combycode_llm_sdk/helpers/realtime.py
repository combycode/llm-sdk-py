"""`Realtime` -- a live session in the shape the rest of this library uses.

    with Realtime(model="openai/gpt-realtime", api_key=key) as session:
        session.send("Name the capital of France.")
        for event in session:
            if event.type == "text":
                print(event.text, end="")
            elif event.type == "turn_end":
                break

A context manager because it holds a socket, and an iterator because that is
what `LLM.stream()` already is. Which provider is behind it changes what arrives
-- Gemini Live models are audio-native and REFUSE a text-only session, OpenAI
answers in text -- but not how it is read.

Usage is metered through the ordinary cost pipeline: every `usage` event is also
emitted as `onCompletion`, so a live session shows up in `engine.cost` next to
every other call rather than being invisible until the invoice.

Transposed from `unified-library-ts/src/helpers/realtime.ts`, where the same
thing is `createRealtime` and the events arrive through `session.on(...)`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

from ..events import Event, UsageEvent
from ..realtime.connection import RealtimeConnection
from ..realtime.providers import REALTIME_ADAPTERS
from ..realtime.session import BaseSession
from ..realtime.types import SessionConfig, WsRequest, modalities_of


class Realtime:
    """A live session over a WebSocket."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        provider: str | None = None,
        modalities: Sequence[str] | None = None,
        voice: str | None = None,
        instructions: str | None = None,
        engine: Any = None,
        base_url: str | None = None,
        connect: Any = None,
        timeout: float | None = None,
    ) -> None:
        from .client_resolver import is_namespaced_model_id, parse_model_id
        from .engine import default_engine

        self._engine = engine if engine is not None else default_engine()
        if model and is_namespaced_model_id(model):
            provider_name, bare_model = parse_model_id(model)
        else:
            provider_name, bare_model = provider or "", model
        if not provider_name:
            raise ValueError(
                'Realtime: name the provider, either as provider= or as '
                '"provider/model" in model=.'
            )

        factory = REALTIME_ADAPTERS.get(provider_name)
        if factory is None:
            raise ValueError(
                f"Realtime: {provider_name!r} hosts no live socket API. "
                f"Available: {', '.join(sorted(REALTIME_ADAPTERS))}."
            )

        key = api_key or (
            self._engine.api_keys.get(provider_name) if self._engine is not None else None
        )
        if not key:
            raise ValueError(f'Realtime: no API key for provider "{provider_name}".')

        # The same catalog translation every other helper does. Without it a live
        # session would be the one path where our slug reached the provider
        # unconverted.
        send_model = bare_model
        catalog = getattr(self._engine, "catalog", None)
        if catalog is not None:
            send_model = catalog.resolve_model_id(provider_name, bare_model)

        self.provider = provider_name
        self.model = send_model
        self._adapter = factory(api_key=key, base_url=base_url)
        self._config = SessionConfig(
            model=send_model,
            modalities=modalities_of(modalities),
            voice=voice,
            instructions=instructions,
        )
        self._connect = connect
        self._timeout = timeout
        self._connection: RealtimeConnection | None = None
        self._session: BaseSession | None = None

    # -- the socket ----------------------------------------------------------

    def open(self) -> Self:
        """Open the socket and complete the provider's handshake."""
        if self._session is not None:
            return self
        request = self._adapter.build_connect_request(self._config)
        connection = self._open_connection(request)
        # The session subscribes BEFORE the socket opens: the handshake is sent
        # from the open callback, and a session wired up afterwards would miss
        # it and then wait forever for a `setupComplete` it never asked for.
        self._session = self._adapter.connect(self._config, lambda _req: connection)
        self._connection = connection
        connection.open()
        return self

    def _open_connection(self, request: WsRequest) -> RealtimeConnection:
        """The connection, from the engine when there is one.

        Going through the engine is what puts the frame hooks on the same bus as
        every other call, so a live session is visible to the same telemetry.
        """
        engine_connect = getattr(self._engine, "realtime_connection", None)
        if engine_connect is not None and self._connect is None:
            connection: RealtimeConnection = engine_connect(request, timeout=self._timeout)
            return connection
        hooks = getattr(self._engine, "hooks", None)
        kwargs: dict[str, Any] = {"connect": self._connect}
        if self._timeout is not None:
            kwargs["timeout"] = self._timeout
        return RealtimeConnection(request, hooks, **kwargs)

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
        self._session = None
        self._connection = None

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- the conversation ----------------------------------------------------

    def send(
        self,
        text: str | None = None,
        *,
        audio: bytes | None = None,
        turn_complete: bool = True,
    ) -> None:
        """Send a turn. Buffered if the handshake has not finished."""
        self._live().send(text, audio=audio, turn_complete=turn_complete)

    def __iter__(self) -> Any:
        """Events until the socket closes.

        A `usage` event is forwarded to the cost pipeline on its way past, which
        is why this wraps the session's iterator rather than being it.
        """
        for event in self._live():
            if isinstance(event, UsageEvent):
                self._record(event)
            yield event

    def _live(self) -> BaseSession:
        if self._session is None:
            raise RuntimeError(
                "Realtime: the session is not open. Use it as a context manager, "
                "or call open() first."
            )
        return self._session

    def _record(self, event: UsageEvent) -> None:
        """Meter a live turn like any other call.

        Emitted as `onCompletion` rather than recorded directly, so the
        CostCollector prices it from the catalog exactly as it prices a
        completion -- one ledger, not two.
        """
        hooks = getattr(self._engine, "hooks", None)
        if hooks is None:
            return
        hooks.emit_sync(
            "onCompletion",
            {
                "provider": self.provider,
                "model": self.model,
                "response": {
                    "id": "",
                    "model": self.model,
                    "content": [],
                    "finishReason": "stop",
                    "usage": _usage_wire(event),
                    "text": "",
                    "toolCalls": [],
                    "thinking": None,
                    "media": [],
                    "latencyMs": 0,
                },
                "request": {
                    "estimatedInputTokens": 0,
                    "inputChars": 0,
                    "messageCount": 0,
                    "hasTools": False,
                },
                "ctx": {},
            },
        )

    def __repr__(self) -> str:
        state = "open" if self._session is not None else "closed"
        return f"<Realtime {self.provider}/{self.model} {state}>"


def _usage_wire(event: UsageEvent) -> dict[str, Any]:
    """The camelCase shape the cost pipeline reads."""
    usage = event.usage
    wire: dict[str, Any] = {
        "inputTokens": usage.input_tokens,
        "outputTokens": usage.output_tokens,
        "totalTokens": usage.total_tokens,
        "cachedTokens": usage.cached_tokens,
    }
    if usage.audio_input_tokens is not None:
        wire["audioInputTokens"] = usage.audio_input_tokens
    if usage.audio_output_tokens is not None:
        wire["audioOutputTokens"] = usage.audio_output_tokens
    return wire


__all__ = ["Event", "Realtime"]
