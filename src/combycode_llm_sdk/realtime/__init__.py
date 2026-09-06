"""Live sessions: a socket instead of a request.

The caller-facing name is `Realtime`, in `helpers/realtime.py`. What lives here
is what it is built from -- the socket and its reader thread (`connection.py`),
the readiness and iteration both providers share (`session.py`), and the two
protocols themselves (`providers.py`).
"""

from __future__ import annotations

from .connection import (
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    POLL_SECONDS,
    RealtimeConnection,
    default_connect,
    frame_of,
)
from .providers import (
    AUDIO_PCM16_SAMPLE_RATE_HZ,
    REALTIME_ADAPTERS,
    GoogleRealtimeAdapter,
    GoogleRealtimeSession,
    OpenAIRealtimeAdapter,
    OpenAIRealtimeSession,
    google_usage,
    openai_usage,
)
from .session import BaseSession
from .types import (
    Connection,
    EngineConnect,
    Frame,
    Modality,
    ProviderAdapter,
    SessionConfig,
    Turn,
    WsConnect,
    WsRequest,
    WsSession,
    modalities_of,
)

__all__ = [
    "AUDIO_PCM16_SAMPLE_RATE_HZ",
    "DEFAULT_CONNECT_TIMEOUT_SECONDS",
    "POLL_SECONDS",
    "REALTIME_ADAPTERS",
    "BaseSession",
    "Connection",
    "EngineConnect",
    "Frame",
    "GoogleRealtimeAdapter",
    "GoogleRealtimeSession",
    "Modality",
    "OpenAIRealtimeAdapter",
    "OpenAIRealtimeSession",
    "ProviderAdapter",
    "RealtimeConnection",
    "SessionConfig",
    "Turn",
    "WsConnect",
    "WsRequest",
    "WsSession",
    "default_connect",
    "frame_of",
    "google_usage",
    "modalities_of",
    "openai_usage",
]
