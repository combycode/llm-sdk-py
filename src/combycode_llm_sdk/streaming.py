"""Parse a provider's SSE stream into unified events, without a client.

Python-native, and named by the API contract: `streamed_citations.py` replays a
recorded Anthropic stream through it to assert what a provider actually sends.

    for event in parse_stream("messages", lines):
        if event.type == "citation":
            ...

This is the same parser `LLM.stream` uses -- the same spec, the same registry --
reached without auth, a transport, or a model. That is the point: a recorded
stream can be replayed against the real parser rather than a stand-in, so a test
that passes says something about the shipped code.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .events import Event, to_event
from .llm.providers.anthropic.stream_registry import ANTHROPIC_STREAM_REGISTRY
from .llm.providers.google.interactions_stream_registry import (
    GOOGLE_INTERACTIONS_STREAM_REGISTRY,
)
from .llm.providers.google.stream_registry import GOOGLE_STREAM_REGISTRY
from .llm.providers.openai.responses_stream_registry import OPENAI_RESPONSES_STREAM_REGISTRY
from .llm.providers.openai.stream_registry import OPENAI_STREAM_REGISTRY
from .llm.providers.openrouter.stream_registry import OPENROUTER_STREAM_REGISTRY
from .llm.providers.xai.stream_registry import XAI_STREAM_REGISTRY
from .network.sse import parse_sse_message
from .wire.interpreter import Registry
from .wire.stream_interpreter import create_stream_builder
from .wire.stream_specs import get_stream_spec

#: The api name a caller names -> its spec and registry.
#:
#: Keyed by the API rather than the provider because that is what decides the
#: wire: `messages` is Anthropic's, `responses` is OpenAI's AND xAI's, and a
#: provider can speak two (Google's `generate` and `interactions`). The four
#: provider-qualified aliases are there for the two cases where the api alone is
#: ambiguous about the registry.
_PARSERS: dict[str, tuple[str, Registry]] = {
    "messages": ("anthropic/messages.stream", ANTHROPIC_STREAM_REGISTRY),
    "completions": ("openai/completions.stream", OPENAI_STREAM_REGISTRY),
    "responses": ("openai/responses.stream", OPENAI_RESPONSES_STREAM_REGISTRY),
    "generate": ("google/generate.stream", GOOGLE_STREAM_REGISTRY),
    "interactions": ("google/interactions.stream", GOOGLE_INTERACTIONS_STREAM_REGISTRY),
    "anthropic/messages": ("anthropic/messages.stream", ANTHROPIC_STREAM_REGISTRY),
    "openai/completions": ("openai/completions.stream", OPENAI_STREAM_REGISTRY),
    "openai/responses": ("openai/responses.stream", OPENAI_RESPONSES_STREAM_REGISTRY),
    "google/generate": ("google/generate.stream", GOOGLE_STREAM_REGISTRY),
    "google/interactions": ("google/interactions.stream", GOOGLE_INTERACTIONS_STREAM_REGISTRY),
    # These two are the reason the map is not derived from the api alone: both
    # inherit another provider's spec but supply their own registry, and reaching
    # them by api would silently use the wrong one.
    "openrouter/completions": ("openrouter/completions.stream", OPENROUTER_STREAM_REGISTRY),
    "xai/responses": ("xai/responses.stream", XAI_STREAM_REGISTRY),
}


def parse_stream(api: str, lines: Iterable[str | bytes]) -> Iterator[Event]:
    """Unified events from raw SSE lines.

    `api` names the wire: `messages`, `completions`, `responses`, `generate`,
    `interactions`, or a `provider/api` pair when the api alone is ambiguous
    (`openrouter/completions`, `xai/responses`).

    `lines` are SSE lines as the provider sent them -- `data: {...}` and the
    blank lines between messages. A recording that dropped the blank separators
    still parses: each `data:` line is treated as a complete message, which is
    what every provider actually sends.
    """
    entry = _PARSERS.get(api)
    if entry is None:
        raise ValueError(f'parse_stream: unknown api "{api}" -- expected one of {_apis()}')
    spec_id, registry = entry
    parse = create_stream_builder(get_stream_spec(spec_id), registry)

    for raw in lines:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        frame = parse_sse_message(text)
        if frame is None:
            # A blank separator, a comment, or the `[DONE]` sentinel: none of
            # them is an event.
            continue
        for event in parse(frame):
            yield to_event(event)


def _apis() -> list[str]:
    return sorted(_PARSERS)


#: The api names `parse_stream` accepts.
STREAM_APIS: tuple[str, ...] = tuple(sorted(_PARSERS))

__all__ = ["STREAM_APIS", "parse_stream"]
