"""What a provider's file store can do, and what it costs to use it.

The four adapters differ in more than their URLs: Google auto-deletes after 48
hours and OpenAI does not, xAI accepts five MIME types and Google accepts
anything, and the size ceilings span two orders of magnitude. Those are the facts
a strategy needs before it can decide whether to upload at all, so they are
declared on the adapter rather than looked up in a table somewhere else.

No adapter holds a fetch. Every call takes one, so all file HTTP rides the
engine's queue, retry policy and hooks -- an upload is not an exception to
`engine.fetch`, it is the largest request the library sends.

Transposed from `unified-library-ts/src/plugins/files/provider-adapter.ts`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .attachment import FileAttachment


@dataclass(frozen=True)
class FileUploadResult:
    """Where the file now lives, and until when."""

    remote_id: str
    #: Epoch milliseconds, or None when the provider keeps it indefinitely.
    expires_at: float | None = None


@dataclass(frozen=True)
class RemoteFileInfo:
    """One entry of the provider's own file listing."""

    remote_id: str
    filename: str
    size_bytes: int
    #: Epoch milliseconds.
    created_at: float
    expires_at: float | None = None


@dataclass(frozen=True)
class FileUploadOptions:
    """What the caller asks for at upload time.

    `lifetime_seconds` exists because files do NOT clean themselves up: OpenAI
    states that everything but `purpose=batch` persists until manually deleted,
    so an agent attaching a document per turn grows an unbounded pile on the
    customer account.

    Three of the four providers accept a lifetime and one does not, which is the
    part worth knowing before using it: Google `expiration_time` is marked
    "Output only" -- Google decides, and asking changes nothing. An adapter that
    cannot honour the request says so through `warn` rather than dropping it
    silently, because a unified option that quietly does nothing on one provider
    is how a caller ends up believing in a cleanup that never happens.
    """

    #: Seconds from upload until the provider deletes the file. Sent as given;
    #: each provider clamps to its own range and reports its own error.
    lifetime_seconds: int | None = None
    #: Raised when an adapter cannot honour an option. Supplied by the registry.
    warn: Callable[[str, Mapping[str, Any] | None], None] | None = None


@runtime_checkable
class FileProviderAdapter(Protocol):
    """One provider's file endpoints."""

    # Declared read-only, as properties, because they are facts about the
    # provider rather than state anyone sets. Mutable protocol attributes are
    # INVARIANT, which would force every adapter to annotate `supported_types`
    # as `tuple[str, ...] | None` even where it is always a tuple -- a type that
    # lies about the adapter to satisfy the checker. Read-only members are
    # covariant, so each one declares what it actually has.

    @property
    def name(self) -> str: ...

    @property
    def expires_after_ms(self) -> float | None:
        """How long an upload survives in milliseconds, None for indefinitely.

        Google is the only provider here that reaps: 48 hours, silently.
        """

    @property
    def max_file_size(self) -> int:
        """The provider's hard ceiling in bytes.

        A file over it is refused by the strategy rather than sent and rejected,
        because the rejection arrives only after the whole body has gone up.
        """

    @property
    def supported_types(self) -> tuple[str, ...] | None:
        """MIME types the store accepts, or None for "anything"."""

    def upload(
        self, file: FileAttachment, fetch: Any, opts: FileUploadOptions | None = None
    ) -> FileUploadResult: ...

    def delete(self, remote_id: str, fetch: Any) -> None: ...

    def get_info(self, remote_id: str, fetch: Any) -> RemoteFileInfo | None: ...

    def list(self, fetch: Any) -> list[RemoteFileInfo]: ...


__all__ = ["FileProviderAdapter", "FileUploadResult", "RemoteFileInfo"]
