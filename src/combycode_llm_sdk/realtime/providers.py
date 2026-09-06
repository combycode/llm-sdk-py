"""The two live protocols, and how little they have in common.

**OpenAI** authenticates in the WebSocket SUBPROTOCOL list -- not a header, not
a query parameter -- because a browser cannot set a header on a WebSocket
handshake. It names the model in the URL, is ready the moment the socket opens,
and signals "answer me now" by sending a SECOND frame.

**Gemini Live** authenticates with a query-string key (the same reason, and the
only place in this library where a key rides in a URL), names the model inside
the setup frame instead of the URL, is not ready until the server answers
`setupComplete`, and carries turn completion as a FIELD on its single frame. Its
JSON arrives in BINARY frames where OpenAI's arrives as text.

Everything above is data in `wire/specs/realtime/*.json`, shared byte-for-byte
with the TypeScript. What is left here is reading the answers.

Transposed from `unified-library-ts/src/llm/providers/{openai,google}/realtime.ts`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..events import AudioEvent, Event, RealtimeErrorEvent, TextEvent, TurnEndEvent, UsageEvent
from ..llm.wire_transforms import make_registry
from ..results import Usage
from ..util.base64 import base64_to_bytes
from ..wire.interpreter import build_connection, build_frames
from ..wire.service_specs import service_spec
from .session import BaseSession
from .types import EngineConnect, Frame, SessionConfig, Turn, WsRequest

#: Both providers stream PCM16 at 24 kHz.
AUDIO_PCM16_SAMPLE_RATE_HZ = 24_000

#: Realtime rules need no adapter handles: the frames are data plus the base64
#: audio encoder, which the shared registry already carries.
_REGISTRY = make_registry({})


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


# ── OpenAI ──────────────────────────────────────────────────────────────────


def openai_usage(raw: Mapping[str, Any]) -> Usage:
    """Realtime usage, with audio split from text.

    Kept apart because they price differently: billing a spoken answer at the
    text rate is not a rounding error, it is the wrong number. When the provider
    reports only a total, the audio tokens are subtracted rather than guessed.
    """
    into = raw.get("input_token_details") or {}
    out_of = raw.get("output_token_details") or {}
    audio_in = _int(into.get("audio_tokens"))
    audio_out = _int(out_of.get("audio_tokens"))
    text_in = into.get("text_tokens")
    text_out = out_of.get("text_tokens")
    return Usage(
        input_tokens=max(0, _int(text_in) if text_in is not None
                         else _int(raw.get("input_tokens")) - audio_in),
        output_tokens=max(0, _int(text_out) if text_out is not None
                          else _int(raw.get("output_tokens")) - audio_out),
        total_tokens=_int(raw.get("total_tokens")),
        cached_tokens=_int(into.get("cached_tokens")),
        # None rather than 0 where there was no audio: `Usage` treats a zero
        # as "billed for silence", which prices a text turn at audio rates.
        audio_input_tokens=audio_in or None,
        audio_output_tokens=audio_out or None,
    )


class OpenAIRealtimeAdapter:
    """`wss://api.openai.com/v1/realtime`."""

    name = "openai"
    spec_id = "openai/realtime"
    default_base_url = "https://api.openai.com"

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url or self.default_base_url

    def _config(self) -> dict[str, Any]:
        return {"baseURL": self._base_url, "apiKey": self._api_key}

    def build_connect_request(self, config: SessionConfig) -> WsRequest:
        """The socket descriptor, built without opening anything."""
        built = build_connection(
            service_spec(self.spec_id), "connect", config.as_input(), _REGISTRY, self._config()
        )
        return WsRequest(
            url=built.url,
            protocols=tuple(built.protocols) if built.protocols else None,
            provider=self.name,
            model=config.model,
        )

    def build_open_frame(self, config: SessionConfig) -> Any:
        return build_frames(
            service_spec(self.spec_id), "open", config.as_input(), _REGISTRY, self._config()
        )[0]

    def build_turn_frames(self, turn: Turn, turn_complete: bool = True) -> list[Any]:
        return build_frames(
            service_spec(self.spec_id),
            "send",
            turn.as_input(turn_complete),
            _REGISTRY,
            self._config(),
        )

    def connect(self, config: SessionConfig, connect: EngineConnect) -> OpenAIRealtimeSession:
        return OpenAIRealtimeSession(connect(self.build_connect_request(config)), config, self)


class OpenAIRealtimeSession(BaseSession):
    """A typed event stream over text frames."""

    def __init__(self, connection: Any, config: SessionConfig, adapter: OpenAIRealtimeAdapter):
        self._adapter = adapter
        super().__init__(connection, config)

    def on_open(self) -> None:
        self.send_json(self._adapter.build_open_frame(self.config))
        # OpenAI accepts conversation items as soon as the socket is up.
        self.mark_ready()

    def build_turn_frames(self, turn: Turn, turn_complete: bool) -> list[Any]:
        return self._adapter.build_turn_frames(turn, turn_complete)

    def on_frame(self, frame: Frame) -> None:
        message = self.json_of(frame)
        if not isinstance(message, Mapping):
            return
        kind = message.get("type")
        if kind == "response.output_text.delta":
            delta = message.get("delta")
            if delta:
                self.emit(TextEvent(type="text", text=str(delta)))
        elif kind == "response.output_audio.delta":
            delta = message.get("delta")
            if delta:
                self.emit(
                    AudioEvent(
                        type="audio",
                        audio=base64_to_bytes(str(delta)),
                        mime_type="audio/pcm",
                        sample_rate=AUDIO_PCM16_SAMPLE_RATE_HZ,
                    )
                )
        elif kind == "response.done":
            usage = (message.get("response") or {}).get("usage")
            if isinstance(usage, Mapping):
                self.emit(UsageEvent(type="usage", usage=openai_usage(usage)))
            self.emit(TurnEndEvent(type="turn_end"))
        elif kind == "error":
            error = message.get("error")
            message_text = (
                str(error.get("message")) if isinstance(error, Mapping) and error.get("message")
                else "realtime error"
            )
            self.emit(RealtimeErrorEvent(type="error", message=message_text))


# ── Google ──────────────────────────────────────────────────────────────────


GOOGLE_WS_BASE = "wss://generativelanguage.googleapis.com"
GOOGLE_API_VERSION = "v1beta"


def google_usage(raw: Mapping[str, Any]) -> Usage:
    output = raw.get("responseTokenCount")
    if output is None:
        output = raw.get("candidatesTokenCount")
    return Usage(
        input_tokens=_int(raw.get("promptTokenCount")),
        output_tokens=_int(output),
        total_tokens=_int(raw.get("totalTokenCount")),
        cached_tokens=_int(raw.get("cachedContentTokenCount")),
    )


class GoogleRealtimeAdapter:
    """Gemini Live, bidi over a WebSocket."""

    name = "google"
    spec_id = "google/realtime"

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._api_key = api_key
        base = base_url or GOOGLE_WS_BASE
        # ws:// by the time the spec sees it, because the spec joins a host onto
        # a path and has no business knowing about schemes.
        if base.startswith("http"):
            base = "ws" + base[len("http") :]
        self._base = base.rstrip("/")

    def _config(self) -> dict[str, Any]:
        # `wsBase` and `apiVersion`, not `baseURL`: the host is already ws:// and
        # the version is part of the RPC name rather than a path prefix, so
        # neither is the plain `baseURL` the HTTP specs read.
        return {
            "wsBase": self._base,
            "apiVersion": GOOGLE_API_VERSION,
            "apiKey": self._api_key,
        }

    def build_connect_request(self, config: SessionConfig) -> WsRequest:
        built = build_connection(
            service_spec(self.spec_id), "connect", config.as_input(), _REGISTRY, self._config()
        )
        return WsRequest(
            url=built.url,
            protocols=tuple(built.protocols) if built.protocols else None,
            provider=self.name,
            model=config.model,
        )

    def build_open_frame(self, config: SessionConfig) -> Any:
        return build_frames(
            service_spec(self.spec_id), "open", config.as_input(), _REGISTRY, self._config()
        )[0]

    def build_turn_frames(self, turn: Turn, turn_complete: bool = True) -> list[Any]:
        return build_frames(
            service_spec(self.spec_id),
            "send",
            turn.as_input(turn_complete),
            _REGISTRY,
            self._config(),
        )

    def connect(self, config: SessionConfig, connect: EngineConnect) -> GoogleRealtimeSession:
        return GoogleRealtimeSession(connect(self.build_connect_request(config)), config, self)


class GoogleRealtimeSession(BaseSession):
    """Turn-based bidi, with the JSON inside binary frames."""

    def __init__(self, connection: Any, config: SessionConfig, adapter: GoogleRealtimeAdapter):
        self._adapter = adapter
        super().__init__(connection, config)

    def on_open(self) -> None:
        # Setup goes out now; readiness waits for the server's `setupComplete`.
        self.send_json(self._adapter.build_open_frame(self.config))

    def build_turn_frames(self, turn: Turn, turn_complete: bool) -> list[Any]:
        return self._adapter.build_turn_frames(turn, turn_complete)

    def on_frame(self, frame: Frame) -> None:
        message = self.json_of(frame)
        if not isinstance(message, Mapping):
            return
        if "setupComplete" in message:
            self.mark_ready()
            return
        usage = message.get("usageMetadata")
        if isinstance(usage, Mapping):
            self.emit(UsageEvent(type="usage", usage=google_usage(usage)))
        content = message.get("serverContent")
        if not isinstance(content, Mapping):
            return
        turn_parts = (content.get("modelTurn") or {}).get("parts") or []
        for part in turn_parts:
            if not isinstance(part, Mapping):
                continue
            self.emit(self._part_event(part))
        if content.get("turnComplete"):
            self.emit(TurnEndEvent(type="turn_end"))

    @staticmethod
    def _part_event(part: Mapping[str, Any]) -> Event | None:
        text = part.get("text")
        if text:
            return TextEvent(type="text", text=str(text))
        inline = part.get("inlineData")
        if isinstance(inline, Mapping) and inline.get("data"):
            return AudioEvent(
                type="audio",
                audio=base64_to_bytes(str(inline["data"])),
                # e.g. "audio/pcm;rate=24000" -- reported as the provider sent
                # it, because a caller writing a WAV header needs the parameters.
                mime_type=str(inline.get("mimeType") or "audio/pcm"),
                sample_rate=AUDIO_PCM16_SAMPLE_RATE_HZ,
            )
        return None


#: Provider name -> its realtime adapter. Anthropic, xAI and OpenRouter host no
#: live socket API at all; they are absent rather than mapped to something else.
REALTIME_ADAPTERS: Mapping[str, type[Any]] = {
    "openai": OpenAIRealtimeAdapter,
    "google": GoogleRealtimeAdapter,
}


__all__ = [
    "AUDIO_PCM16_SAMPLE_RATE_HZ",
    "GOOGLE_API_VERSION",
    "GOOGLE_WS_BASE",
    "REALTIME_ADAPTERS",
    "GoogleRealtimeAdapter",
    "GoogleRealtimeSession",
    "OpenAIRealtimeAdapter",
    "OpenAIRealtimeSession",
    "google_usage",
    "openai_usage",
]
