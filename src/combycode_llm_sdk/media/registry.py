"""Which adapter serves which provider.

Its own module so `output.py` can import it without importing every adapter
eagerly -- and so a caller bringing their own adapter has one obvious place to
look for the contract it must satisfy.
"""

from __future__ import annotations

from typing import Any

from .google import GoogleMediaAdapter
from .openai import OpenAIMediaAdapter
from .openrouter import OpenRouterMediaAdapter
from .xai import XAIMediaAdapter

ADAPTERS = {
    "openai": OpenAIMediaAdapter,
    "google": GoogleMediaAdapter,
    "xai": XAIMediaAdapter,
    "openrouter": OpenRouterMediaAdapter,
}


def media_adapter(provider: str, api_key: str, base_url: str | None = None) -> Any:
    """The media adapter for one provider, or a refusal naming the ones there are.

    Anthropic generates no media of any kind -- no images, no speech, no video --
    so there is nothing to fall back to and nothing worth guessing.
    """
    factory = ADAPTERS.get(provider)
    if factory is None:
        raise ValueError(
            f"media: {provider!r} generates no media. "
            f"Available: {', '.join(sorted(ADAPTERS))}."
        )
    return factory(api_key, base_url)


__all__ = ["ADAPTERS", "media_adapter"]
