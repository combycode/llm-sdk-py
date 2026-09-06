"""Files, uploaded once and referenced after.

    registry = FilesRegistry(hooks=engine.hooks, fetch=engine.fetch)
    registry.register_provider("openai", OpenAIFileAdapter(api_key=key))

    complete(model="openai/gpt-5.4-nano", prompt="Summarise it", messages=[...])

Installing the registry is the whole opt-in. From then on a `path`, `buffer` or
`file` source in any message is resolved on its way out -- uploaded and
referenced when that is cheaper, inlined when it is not, and refused with a
reason when the provider will not take it at all.

See `attachment.py` for the per-provider upload state, `strategy.py` for the
decision, `registry.py` for the hook, and `providers.py` for the four stores.
"""

from __future__ import annotations

from .attachment import (
    Base64Content,
    BytesContent,
    FileAttachment,
    FileAttachmentSnapshot,
    FileContent,
    FileUploadState,
    PathContent,
    UploadStatus,
    UrlContent,
)
from .provider_adapter import FileProviderAdapter, FileUploadResult, RemoteFileInfo
from .providers import (
    FILE_ADAPTERS,
    AnthropicFileAdapter,
    GoogleFileAdapter,
    OpenAIFileAdapter,
    XaiFileAdapter,
    google_file_name,
)
from .registry import FILE_PART_TYPES, RESOLVABLE_SOURCES, FilesRegistry
from .strategy import (
    DEFAULT_INLINE_THRESHOLD_BYTES,
    Action,
    DefaultFileStrategy,
    FileDecision,
    FileStrategy,
    FileStrategyContext,
)

__all__ = [
    "DEFAULT_INLINE_THRESHOLD_BYTES",
    "FILE_ADAPTERS",
    "FILE_PART_TYPES",
    "RESOLVABLE_SOURCES",
    "Action",
    "AnthropicFileAdapter",
    "Base64Content",
    "BytesContent",
    "DefaultFileStrategy",
    "FileAttachment",
    "FileAttachmentSnapshot",
    "FileContent",
    "FileDecision",
    "FileProviderAdapter",
    "FileStrategy",
    "FileStrategyContext",
    "FileUploadResult",
    "FileUploadState",
    "FilesRegistry",
    "GoogleFileAdapter",
    "OpenAIFileAdapter",
    "PathContent",
    "RemoteFileInfo",
    "UploadStatus",
    "UrlContent",
    "XaiFileAdapter",
    "google_file_name",
]
