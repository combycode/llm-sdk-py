"""One file, and what each provider currently knows about it.

The same PDF sent to Anthropic and to OpenAI is one document with two remote
identities, and they expire independently. Holding the upload state per provider
is what makes `attachments=[the_same_file]` work across a multi-provider run
without uploading it twice or, worse, sending Anthropic an id that OpenAI issued.

**Timestamps here are epoch MILLISECONDS**, not the seconds this library uses for
durations elsewhere. `export()` produces a snapshot the TypeScript must be able
to read and write, and it stores milliseconds; a registry persisted by one side
and loaded by the other has to agree on the unit.

Transposed from `unified-library-ts/src/plugins/files/attachment.ts`. The
TypeScript's `blob` content kind and `fromBlob` have no counterpart: `Blob` is a
browser type whose Python equivalent is `bytes`, which `BytesContent` already
covers. `from_path` and `from_bytes` are the constructors a Python caller
actually reaches for.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..util.base64 import base64_to_bytes, bytes_to_base64


#: Every file operation is measured on this clock. Epoch milliseconds -- see the
#: module note; the snapshot format is shared with the TypeScript.
def now_ms() -> float:
    return time.time() * 1000


@dataclass(frozen=True)
class BytesContent:
    """The bytes, already in hand."""

    mime_type: str
    data: bytes
    type: Literal["buffer"] = "buffer"


@dataclass(frozen=True)
class PathContent:
    """A file on disk, read only when something actually needs it.

    Deferred on purpose: a registry may hold a hundred attachments and send
    three, and reading the other ninety-seven to find that out would be the
    slowest possible way to learn nothing.
    """

    mime_type: str
    path: str
    type: Literal["path"] = "path"


@dataclass(frozen=True)
class UrlContent:
    """A URL the provider fetches itself. Never read by this library."""

    url: str
    mime_type: str | None = None
    type: Literal["url"] = "url"


@dataclass(frozen=True)
class Base64Content:
    """Already encoded -- passed through rather than round-tripped."""

    mime_type: str
    data: str
    type: Literal["base64"] = "base64"


FileContent = BytesContent | PathContent | UrlContent | Base64Content

UploadStatus = Literal["pending", "uploaded", "expired", "deleted", "error"]


@dataclass
class FileUploadState:
    """What one provider knows about one file."""

    provider: str
    status: UploadStatus = "pending"
    remote_id: str | None = None
    uploaded_at: float | None = None
    expires_at: float | None = None
    error: str | None = None

    def as_row(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "status": self.status,
            "remoteId": self.remote_id,
            "uploadedAt": self.uploaded_at,
            "expiresAt": self.expires_at,
            "error": self.error,
        }


@dataclass(frozen=True)
class FileAttachmentSnapshot:
    """A registry entry as it survives a restart."""

    id: str
    filename: str
    mime_type: str
    size_bytes: int
    uploads: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0

    def as_row(self) -> dict[str, Any]:
        """The camelCase form the TypeScript reads."""
        return {
            "id": self.id,
            "filename": self.filename,
            "mimeType": self.mime_type,
            "sizeBytes": self.size_bytes,
            "uploads": [dict(u) for u in self.uploads],
            "metadata": dict(self.metadata),
            "createdAt": self.created_at,
        }


class FileAttachment:
    """A file plus its per-provider upload state."""

    def __init__(
        self,
        *,
        filename: str,
        mime_type: str,
        size_bytes: int,
        content: FileContent,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.id = id or str(uuid.uuid4())
        self.filename = filename
        self.mime_type = mime_type
        self.size_bytes = size_bytes
        self.content = content
        self.created_at = now_ms()
        self.metadata: dict[str, Any] = dict(metadata or {})
        #: provider -> what that provider knows. Never shared between providers:
        #: a remote id is meaningless anywhere but where it was issued.
        self.uploads: dict[str, FileUploadState] = {}

    # -- construction --------------------------------------------------------

    @staticmethod
    def from_path(
        path: str | Path,
        *,
        mime_type: str,
        filename: str | None = None,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FileAttachment:
        """A file on disk. Its size is read now; its bytes are not."""
        p = Path(path)
        return FileAttachment(
            id=id,
            filename=filename or p.name,
            mime_type=mime_type,
            size_bytes=p.stat().st_size,
            content=PathContent(mime_type=mime_type, path=str(p)),
            metadata=metadata,
        )

    @staticmethod
    def from_bytes(
        data: bytes,
        *,
        filename: str,
        mime_type: str,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FileAttachment:
        """Bytes already in memory -- an upload, a generated report, a download."""
        return FileAttachment(
            id=id,
            filename=filename,
            mime_type=mime_type,
            size_bytes=len(data),
            content=BytesContent(mime_type=mime_type, data=data),
            metadata=metadata,
        )

    # -- per-provider state --------------------------------------------------

    def is_available(self, provider: str) -> bool:
        """Is there a usable remote id for this provider right now?

        Checking expiry HERE, and marking the state expired as a side effect, is
        what stops a stale id being sent: the provider answers 404 for a file the
        caller believes it uploaded, which reads like a bug in this library.
        """
        state = self.uploads.get(provider)
        if state is None or state.status != "uploaded":
            return False
        if state.expires_at is not None and now_ms() > state.expires_at:
            state.status = "expired"
            return False
        return True

    def needs_upload(self, provider: str) -> bool:
        return not self.is_available(provider)

    def get_ref(self, provider: str) -> str | None:
        if not self.is_available(provider):
            return None
        state = self.uploads.get(provider)
        return state.remote_id if state else None

    def set_uploaded(self, provider: str, remote_id: str, expires_at: float | None) -> None:
        self.uploads[provider] = FileUploadState(
            provider=provider,
            status="uploaded",
            remote_id=remote_id,
            uploaded_at=now_ms(),
            expires_at=expires_at,
        )

    def set_error(self, provider: str, error: str) -> None:
        self.uploads[provider] = FileUploadState(
            provider=provider, status="error", error=error
        )

    def set_deleted(self, provider: str) -> None:
        state = self.uploads.get(provider)
        if state is not None:
            state.status = "deleted"

    # -- reading the bytes ---------------------------------------------------

    def to_bytes(self) -> bytes:
        """The raw bytes, read now."""
        content = self.content
        if isinstance(content, BytesContent):
            return content.data
        if isinstance(content, Base64Content):
            return base64_to_bytes(content.data)
        if isinstance(content, PathContent):
            return Path(content.path).read_bytes()
        raise ValueError(
            f"{self.filename}: cannot load URL content without fetching it. "
            "A url attachment is for providers that fetch it themselves."
        )

    def to_base64(self) -> str:
        """The bytes, base64-encoded -- what an inline content part carries."""
        content = self.content
        if isinstance(content, Base64Content):
            return content.data
        return bytes_to_base64(self.to_bytes())

    # -- persistence ---------------------------------------------------------

    def export(self) -> FileAttachmentSnapshot:
        return FileAttachmentSnapshot(
            id=self.id,
            filename=self.filename,
            mime_type=self.mime_type,
            size_bytes=self.size_bytes,
            uploads=[state.as_row() for state in self.uploads.values()],
            metadata=dict(self.metadata),
            created_at=self.created_at,
        )

    def __repr__(self) -> str:
        where = ", ".join(sorted(self.uploads)) or "nowhere"
        return f"<FileAttachment {self.filename!r} {self.size_bytes}B uploaded to {where}>"


__all__ = [
    "Base64Content",
    "BytesContent",
    "FileAttachment",
    "FileAttachmentSnapshot",
    "FileContent",
    "FileUploadState",
    "PathContent",
    "UploadStatus",
    "UrlContent",
    "now_ms",
]
