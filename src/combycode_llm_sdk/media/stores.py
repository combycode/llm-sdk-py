"""Two places to put the bytes.

`MemoryMediaStore` is the default and loses everything on exit, which is right
for a test and for a caller who is about to write the bytes somewhere else
themselves. `FileMediaStore` writes the asset and a metadata sidecar to a
directory, so a generated image survives the process that made it.

The metadata is stored as JSON beside the bytes rather than inside them: a PNG
has nowhere to put a prompt, and a caller listing a directory should be able to
read what each file is without decoding it.

Transposed from `unified-library-ts/src/plugins/media/memory-store.ts` and
`file-store.ts`.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path

from .types import MediaMeta, MediaType

#: The extension each media type is written with. A generated PNG called `.bin`
#: is one no image viewer will open, and the caller is usually a person.
MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpeg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "audio/mp3": ".mp3",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/pcm": ".pcm",
    "audio/opus": ".opus",
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}


def ext_for_mime(mime_type: str) -> str:
    """The file extension for a media type, guessed from the subtype if unknown."""
    known = MIME_TO_EXT.get(mime_type)
    if known:
        return known
    subtype = mime_type.split("/")[-1] if "/" in mime_type else ""
    return f".{subtype or 'bin'}"


class MemoryMediaStore:
    """Assets held in this process, and gone when it ends.

    Locked, because media generation is the one place this library hands work
    to a thread pool: several images from one request are saved concurrently.
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[bytes, MediaMeta]] = {}
        self._lock = threading.Lock()

    def save(self, media_id: str, data: bytes, meta: MediaMeta) -> None:
        with self._lock:
            self._entries[media_id] = (data, meta)

    def load(self, media_id: str) -> tuple[bytes, MediaMeta] | None:
        with self._lock:
            return self._entries.get(media_id)

    def get_meta(self, media_id: str) -> MediaMeta | None:
        entry = self.load(media_id)
        return entry[1] if entry else None

    def delete(self, media_id: str) -> None:
        with self._lock:
            self._entries.pop(media_id, None)

    def list(
        self, media_type: MediaType | None = None, provider: str | None = None
    ) -> Sequence[str]:
        with self._lock:
            items = list(self._entries.items())
        return [
            media_id
            for media_id, (_, meta) in items
            if (media_type is None or meta.type == media_type)
            and (provider is None or meta.provider == provider)
        ]

    def has(self, media_id: str) -> bool:
        with self._lock:
            return media_id in self._entries

    def __repr__(self) -> str:
        return f"<MemoryMediaStore {len(self._entries)} asset(s)>"


class FileMediaStore:
    """Assets on disk: the bytes, and a `.meta.json` sidecar beside them."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _data_path(self, media_id: str, mime_type: str) -> Path:
        return self.dir / f"{media_id}{ext_for_mime(mime_type)}"

    def _meta_path(self, media_id: str) -> Path:
        return self.dir / f"{media_id}.meta.json"

    def save(self, media_id: str, data: bytes, meta: MediaMeta) -> None:
        self._data_path(media_id, meta.mime_type).write_bytes(data)
        self._meta_path(media_id).write_text(
            json.dumps(meta.as_row(), indent=2), encoding="utf-8"
        )

    def get_meta(self, media_id: str) -> MediaMeta | None:
        path = self._meta_path(media_id)
        if not path.exists():
            return None
        try:
            return MediaMeta.of(json.loads(path.read_text(encoding="utf-8")))
        except ValueError:
            # A truncated sidecar -- a crash mid-write, say. Reported as absent
            # rather than raised: the caller asked whether this asset is here.
            return None

    def load(self, media_id: str) -> tuple[bytes, MediaMeta] | None:
        meta = self.get_meta(media_id)
        if meta is None:
            return None
        path = self._data_path(media_id, meta.mime_type)
        if not path.exists():
            return None
        return path.read_bytes(), meta

    def delete(self, media_id: str) -> None:
        meta = self.get_meta(media_id)
        if meta is not None:
            self._data_path(media_id, meta.mime_type).unlink(missing_ok=True)
        self._meta_path(media_id).unlink(missing_ok=True)

    def list(
        self, media_type: MediaType | None = None, provider: str | None = None
    ) -> Sequence[str]:
        out: list[str] = []
        for path in sorted(self.dir.glob("*.meta.json")):
            media_id = path.name[: -len(".meta.json")]
            meta = self.get_meta(media_id)
            if meta is None:
                continue
            if media_type is not None and meta.type != media_type:
                continue
            if provider is not None and meta.provider != provider:
                continue
            out.append(media_id)
        return out

    def has(self, media_id: str) -> bool:
        return self._meta_path(media_id).exists()

    def __repr__(self) -> str:
        return f"<FileMediaStore {self.dir}>"


__all__ = ["MIME_TO_EXT", "FileMediaStore", "MemoryMediaStore", "ext_for_mime"]
