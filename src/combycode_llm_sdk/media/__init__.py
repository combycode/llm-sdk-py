"""Media generation: images, speech and video over separate provider endpoints.

`MediaOutput` is the entry point; the adapters build requests from the wire
specs and the stores decide where the bytes end up. Media that arrives INLINE in
a completion is not this subsystem's business -- the chat adapters handle that.

Transposed from `unified-library-ts/src/plugins/media/` and the four
`llm/providers/*/media.ts` adapters.
"""

from __future__ import annotations

from .google import GoogleMediaAdapter
from .openai import OpenAIMediaAdapter
from .openrouter import OpenRouterMediaAdapter
from .output import MediaOutput
from .registry import ADAPTERS, media_adapter
from .stores import FileMediaStore, MemoryMediaStore, ext_for_mime
from .types import (
    MEDIA_OUTPUT_DEFAULTS,
    MediaCapabilities,
    MediaMeta,
    MediaResult,
    MediaStore,
    MediaType,
    RawMediaResult,
    VideoStatus,
)
from .xai import XAIMediaAdapter

__all__ = [
    "ADAPTERS",
    "MEDIA_OUTPUT_DEFAULTS",
    "FileMediaStore",
    "GoogleMediaAdapter",
    "MediaCapabilities",
    "MediaMeta",
    "MediaOutput",
    "MediaResult",
    "MediaStore",
    "MediaType",
    "MemoryMediaStore",
    "OpenAIMediaAdapter",
    "OpenRouterMediaAdapter",
    "RawMediaResult",
    "VideoStatus",
    "XAIMediaAdapter",
    "ext_for_mime",
    "media_adapter",
]
