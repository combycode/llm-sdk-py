"""Inline it, upload it, link to it, or refuse it.

Uploading is not free -- it is a round trip before the round trip, and at some
providers a stored object with a lifetime and a bill. Inlining is not free
either: base64 costs a third more bytes on every single request that carries it.
The crossover is a size, and it is the only interesting decision this subsystem
makes, so it lives behind an interface a caller can replace wholesale.

Transposed from `unified-library-ts/src/plugins/files/strategy.ts`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .attachment import FileAttachment, UrlContent

#: Below this, base64 in the request body beats a round trip. 50 kB is roughly
#: where the extra 33% of an inline encoding stops being cheaper than a second
#: HTTP call, and it is a constructor argument because the right answer depends
#: on how often the same file is sent.
DEFAULT_INLINE_THRESHOLD_BYTES = 50_000

#: What a provider is assumed to accept when no adapter is registered for it.
#: Generous on purpose: the guess must not be the thing that refuses a file.
FALLBACK_MAX_FILE_SIZE = 500_000_000

#: Providers that fetch a URL themselves rather than making the caller download
#: and re-upload it.
URL_CAPABLE_PROVIDERS = ("openai", "xai")

Action = Literal["upload", "reupload", "inline", "url", "skip"]


@dataclass(frozen=True)
class FileStrategyContext:
    """Everything the decision may consider."""

    file: FileAttachment
    provider: str
    model: str
    is_uploaded: bool
    is_expired: bool
    provider_max_size: int
    provider_supports_type: bool
    model_info: Any | None = None


@dataclass(frozen=True)
class FileDecision:
    """What to do, and why -- the reason reaches a warning or a log line."""

    action: Action
    reason: str


class FileStrategy(Protocol):
    def decide(self, ctx: FileStrategyContext) -> FileDecision: ...


class DefaultFileStrategy:
    """Upload the large, inline the small, link when the provider will fetch."""

    def __init__(self, inline_threshold: int = DEFAULT_INLINE_THRESHOLD_BYTES) -> None:
        self.inline_threshold = inline_threshold

    def decide(self, ctx: FileStrategyContext) -> FileDecision:
        # The two refusals come first: neither is worth a round trip to confirm.
        if not ctx.provider_supports_type:
            return FileDecision(
                "skip", f"{ctx.provider} does not accept {ctx.file.mime_type}"
            )
        if ctx.file.size_bytes > ctx.provider_max_size:
            return FileDecision(
                "skip",
                f"file is {ctx.file.size_bytes}B, over {ctx.provider}'s "
                f"{ctx.provider_max_size}B limit",
            )

        # Already there: the cheapest possible answer, and the reason the upload
        # state is tracked per provider at all. `is_expired` is checked as well
        # as `is_uploaded` even though the registry never reports both -- this is
        # a public interface, and a caller assembling a context by hand can.
        if ctx.is_uploaded and not ctx.is_expired:
            return FileDecision("upload", "already uploaded, reusing the reference")
        if ctx.is_expired:
            return FileDecision("reupload", "the previous upload expired")

        if isinstance(ctx.file.content, UrlContent) and ctx.provider in URL_CAPABLE_PROVIDERS:
            return FileDecision("url", f"{ctx.provider} fetches a URL itself")

        if ctx.file.size_bytes < self.inline_threshold:
            return FileDecision(
                "inline",
                f"small file ({ctx.file.size_bytes}B under the "
                f"{self.inline_threshold}B threshold)",
            )
        return FileDecision("upload", "over the inline threshold")


__all__ = [
    "DEFAULT_INLINE_THRESHOLD_BYTES",
    "FALLBACK_MAX_FILE_SIZE",
    "URL_CAPABLE_PROVIDERS",
    "Action",
    "DefaultFileStrategy",
    "FileDecision",
    "FileStrategy",
    "FileStrategyContext",
]
