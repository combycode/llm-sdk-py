"""The plugin that turns a file on disk into whatever this provider wants.

A caller writes the same message either way:

    {"type": "document", "source": {"type": "path", "path": "report.pdf",
                                    "mimeType": "application/pdf"}}

With no registry installed that part goes out as it stands and the adapter
inlines it. With one installed, the registry intercepts `onMessageResolve`, asks
the strategy what to do, and rewrites the part in place -- into a `provider_ref`
when it uploaded, a `base64` when it inlined, a `url` when the provider will
fetch it, or a line of text saying why it was skipped. The caller's code does not
change, which is the entire point: WHERE the bytes go is an operational question,
not something a prompt should have to know.

The handler is deliberately SYNCHRONOUS. `LLMClient.complete` emits this hook
with `emit_sync`, which starts coroutines without awaiting them -- an async
handler here would return before uploading anything and the request would go out
with an unresolved path source. The cost is that an upload blocks, including
under the async client. That is the honest trade for one implementation serving
both cores.

Transposed from `unified-library-ts/src/plugins/files/registry.ts`.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, Self

from .attachment import (
    Base64Content,
    BytesContent,
    FileAttachment,
    FileContent,
    PathContent,
    UrlContent,
)
from .provider_adapter import FileProviderAdapter, RemoteFileInfo
from .strategy import (
    FALLBACK_MAX_FILE_SIZE,
    DefaultFileStrategy,
    FileDecision,
    FileStrategy,
    FileStrategyContext,
)

#: The content-part kinds that can carry a file. A `text` part never does.
FILE_PART_TYPES = ("image", "document", "audio", "video")

#: The source kinds this registry claims. A `base64` or `url` source is already
#: in a form the provider takes, so it is left exactly as the caller wrote it.
RESOLVABLE_SOURCES = ("path", "buffer", "file")

#: Base64 carries 4 characters for every 3 bytes.
_BASE64_RATIO = 3 / 4

#: Named here because the class below defines a method called `list`, which
#: shadows the builtin inside its own annotations.
AttachmentList = list[FileAttachment]
RemoteFileList = list[RemoteFileInfo]


class FilesRegistry:
    """Files this process knows about, and where each one currently lives."""

    def __init__(
        self,
        hooks: Any,
        fetch: Any,
        catalog: Any = None,
        strategy: FileStrategy | None = None,
    ) -> None:
        self.hooks = hooks
        #: Every adapter call takes this, so file HTTP rides the same queue,
        #: retry policy and hooks as a completion. Never a side-fetch.
        self.fetch = fetch
        self.catalog = catalog
        self.strategy: FileStrategy = strategy or DefaultFileStrategy()
        self._files: dict[str, FileAttachment] = {}
        self._providers: dict[str, FileProviderAdapter] = {}
        self._unsubscribe = hooks.on("onMessageResolve", self._resolve_messages)

    def close(self) -> None:
        """Stop intercepting. Idempotent -- closing twice is not an error."""
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- providers -----------------------------------------------------------

    def register_provider(self, name: str, adapter: FileProviderAdapter) -> None:
        self._providers[name] = adapter

    def provider(self, name: str) -> FileProviderAdapter | None:
        return self._providers.get(name)

    # -- files ---------------------------------------------------------------

    def add(
        self,
        *,
        filename: str,
        mime_type: str,
        content: FileContent,
        size_bytes: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FileAttachment:
        file = FileAttachment(
            filename=filename,
            mime_type=mime_type,
            size_bytes=size_bytes if size_bytes is not None else self._estimate_size(content),
            content=content,
            metadata=metadata,
        )
        self._files[file.id] = file
        return file

    def get(self, file_id: str) -> FileAttachment | None:
        return self._files.get(file_id)

    def list(self) -> AttachmentList:
        return list(self._files.values())

    def remove(self, file_id: str) -> None:
        self._files.pop(file_id, None)

    # -- uploads -------------------------------------------------------------

    def upload(self, file_id: str, provider: str) -> str:
        file = self._files.get(file_id)
        if file is None:
            raise KeyError(f"file not in this registry: {file_id}")
        adapter = self._providers.get(provider)
        if adapter is None:
            raise KeyError(
                f"no file adapter registered for {provider!r}. "
                f"Registered: {', '.join(sorted(self._providers)) or 'none'}"
            )

        try:
            result = adapter.upload(file, self.fetch)
        except Exception as exc:
            # Recorded before it is re-raised: a failed upload that leaves no
            # trace looks like one that never happened, and the next call would
            # try to inline a file the strategy had already ruled out.
            file.set_error(provider, str(exc))
            raise

        file.set_uploaded(provider, result.remote_id, result.expires_at)
        self._warn(
            "file_uploaded",
            f"uploaded {file.filename} to {provider} as {result.remote_id}",
            {"fileId": file_id, "provider": provider, "remoteId": result.remote_id},
        )
        return result.remote_id

    def delete_remote(self, file_id: str, provider: str) -> None:
        """Delete the provider's copy. Silent when there is nothing to delete."""
        file = self._files.get(file_id)
        if file is None:
            return
        ref = file.get_ref(provider)
        adapter = self._providers.get(provider)
        if ref is None or adapter is None:
            return
        adapter.delete(ref, self.fetch)
        file.set_deleted(provider)

    def list_remote(self, provider: str) -> RemoteFileList:
        adapter = self._providers.get(provider)
        return adapter.list(self.fetch) if adapter is not None else []

    # -- the hook ------------------------------------------------------------

    def _resolve_messages(self, ctx: Any) -> None:
        provider = str(ctx.get("provider") or "")
        model = str(ctx.get("model") or "")
        adapter = self._providers.get(provider)

        for message in ctx.get("messages") or []:
            content = message.get("content") if isinstance(message, Mapping) else None
            # A string content has no parts to resolve, and is the common case.
            if not isinstance(content, list):
                continue
            for index, part in enumerate(content):
                if not self._carries_a_file(part):
                    continue
                resolved = self._resolve_part(part, provider, model, adapter)
                if resolved is not None:
                    content[index] = resolved

    def _carries_a_file(self, part: Any) -> bool:
        if not isinstance(part, Mapping) or part.get("type") not in FILE_PART_TYPES:
            return False
        source = part.get("source")
        return isinstance(source, Mapping) and source.get("type") in RESOLVABLE_SOURCES

    def _resolve_part(
        self, part: Mapping[str, Any], provider: str, model: str, adapter: Any
    ) -> dict[str, Any] | None:
        source = part["source"]
        part_type = str(part["type"])
        file = self._file_for(source)
        if file is None:
            return None

        state = file.uploads.get(provider)
        decision = self.strategy.decide(
            FileStrategyContext(
                file=file,
                provider=provider,
                model=model,
                is_uploaded=file.is_available(provider),
                is_expired=state is not None and state.status == "expired",
                provider_max_size=getattr(adapter, "max_file_size", None)
                or FALLBACK_MAX_FILE_SIZE,
                provider_supports_type=self._supports(adapter, file.mime_type),
                model_info=self.catalog.get(provider, model) if self.catalog else None,
            )
        )
        return self._carry_out(file, decision, provider, part_type, adapter)

    def _file_for(self, source: Mapping[str, Any]) -> FileAttachment | None:
        """The attachment a source names, registering it on the spot if new."""
        kind = source.get("type")
        if kind == "file":
            file_id = str(source.get("fileId") or "")
            file = self._files.get(file_id)
            if file is None:
                self._warn("file_not_found", f"file {file_id} is not in this registry")
            return file
        if kind == "path":
            path = Path(str(source.get("path") or ""))
            mime_type = str(source.get("mimeType") or "application/octet-stream")
            return self.add(
                filename=path.name or "file",
                mime_type=mime_type,
                content=PathContent(mime_type=mime_type, path=str(path)),
                size_bytes=path.stat().st_size,
            )
        if kind == "buffer":
            data = source.get("data")
            data = bytes(data) if isinstance(data, (bytes, bytearray, memoryview)) else b""
            mime_type = str(source.get("mimeType") or "application/octet-stream")
            return self.add(
                filename=str(source.get("filename") or "buffer-file"),
                mime_type=mime_type,
                content=BytesContent(mime_type=mime_type, data=data),
                size_bytes=len(data),
            )
        return None

    @staticmethod
    def _supports(adapter: Any, mime_type: str) -> bool:
        """Does this provider's store accept this type?

        Matched on the top-level type (`image/`, `text/`) rather than exactly,
        which is what the TypeScript does: a store that lists `image/png` takes
        `image/webp` too, and listing every subtype would be a table that goes
        stale the week a provider adds one.
        """
        if adapter is None:
            return True
        supported = getattr(adapter, "supported_types", None)
        if not supported:
            return True
        head = mime_type.split("/")[0]
        return any(t.split("/")[0] == head for t in supported)

    # -- carrying out a decision ---------------------------------------------

    def _carry_out(
        self,
        file: FileAttachment,
        decision: FileDecision,
        provider: str,
        part_type: str,
        adapter: Any,
    ) -> dict[str, Any]:
        if decision.action in ("upload", "reupload"):
            # No adapter means nothing can be uploaded; inlining still sends the
            # file, which is what the caller asked for.
            if adapter is None:
                return self._inline_part(file, part_type)
            if file.needs_upload(provider):
                self.upload(file.id, provider)
            ref = file.get_ref(provider)
            if ref is None:
                return self._inline_part(file, part_type)
            return {
                "type": part_type,
                "source": {"type": "provider_ref", "mimeType": file.mime_type, "refId": ref},
            }

        if decision.action == "url":
            if isinstance(file.content, UrlContent):
                return {"type": part_type, "source": {"type": "url", "url": file.content.url}}
            return self._inline_part(file, part_type)

        if decision.action == "skip":
            self._warn(
                "file_skipped",
                decision.reason,
                {"fileId": file.id, "provider": provider},
            )
            # A text part rather than a removal: the model is told the document
            # is missing, instead of being asked about one it cannot see.
            return {
                "type": "text",
                "text": f"[file {file.filename} not sent: {decision.reason}]",
            }

        return self._inline_part(file, part_type)

    def _inline_part(self, file: FileAttachment, part_type: str) -> dict[str, Any]:
        return {
            "type": part_type,
            "source": {
                "type": "base64",
                "mimeType": file.mime_type,
                "data": file.to_base64(),
            },
        }

    # -- helpers -------------------------------------------------------------

    def _warn(self, code: str, message: str, details: Mapping[str, Any] | None = None) -> None:
        ctx: MutableMapping[str, Any] = {
            "source": "files",
            "code": code,
            "message": message,
        }
        if details:
            ctx["details"] = dict(details)
        self.hooks.emit_sync("onWarning", ctx)

    @staticmethod
    def _estimate_size(content: FileContent) -> int:
        """Bytes, without reading anything that is not already in memory.

        A path or a URL reports 0 rather than being opened: the size is only
        needed to choose between inlining and uploading, and a registry holding a
        hundred paths must not stat all of them to add one.
        """
        if isinstance(content, BytesContent):
            return len(content.data)
        if isinstance(content, Base64Content):
            return int(len(content.data) * _BASE64_RATIO)
        return 0

    def __repr__(self) -> str:
        return (
            f"<FilesRegistry {len(self._files)} file(s), "
            f"providers: {', '.join(sorted(self._providers)) or 'none'}>"
        )


__all__ = ["FILE_PART_TYPES", "RESOLVABLE_SOURCES", "FilesRegistry"]
