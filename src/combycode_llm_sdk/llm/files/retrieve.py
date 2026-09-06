"""Retrieve the bytes of a hosted-tool output file (e.g. code-execution files).

Transposed from `unified-library-ts/src/llm/files/retrieve.ts`.

A `FileOutput` may carry inline base64 `data` (Google), a `url` (OpenAI
code-interpreter images), or an `id` fetched from the provider's files API
(Anthropic, OpenAI container files). These helpers resolve all three into bytes
-- buffered (`retrieve_file`) or streamed (`stream_file`, for large files piped
straight to a sink).

All HTTP flows through the injected `EngineFetch` (auth, queue, cost, traces).

**Two browser types have no Python equivalent and are replaced deliberately.**
`Blob` becomes `bytes`: everything a caller does with the Blob -- read it, write
it, know its size and type -- is served by the bytes plus the `mimeType` /
`size` / `name` this already returns alongside. `ReadableStream<Uint8Array>`
becomes an async iterator of `bytes`, which is what the engine yields and what
`async for` consumes. Nothing about WHICH request is sent changes.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Mapping
from typing import Any
from urllib.parse import unquote

from ...runtime import is_browser
from ...util.base64 import base64_to_bytes
from ...util.http import header
from ...wire.interpreter import build_from_spec
from ...wire.utility_specs import utility_spec
from ..providers.anthropic.constants import ANTHROPIC_API_VERSION
from ..wire_transforms import make_registry

#: `interface RetrieveContext` (retrieve.ts:21) --
#: `{provider, apiKey, fetch, baseURL?}`.
RetrieveContext = dict[str, Any]

#: `interface RetrievedFile` (retrieve.ts:30) --
#: `{bytes, name?, mimeType, size}`. `bytes` stands in for the TypeScript's
#: `blob`; the name is different because the thing is different, and a `blob`
#: key holding raw bytes would be a lie in both directions.
RetrievedFile = dict[str, Any]

#: `interface FileStream` (retrieve.ts:42) --
#: `{stream, name?, mimeType?, size?}`. `stream` is an async iterator of bytes.
FileStream = dict[str, Any]

_DEFAULT_BASE: dict[str, str] = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
    "xai": "https://api.x.ai",
    "google": "https://generativelanguage.googleapis.com",
    "openrouter": "https://openrouter.ai/api",
}

_OCTET_STREAM = "application/octet-stream"

_EXT_MIME: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "svg": "image/svg+xml",
    "bmp": "image/bmp",
    "csv": "text/csv",
    "tsv": "text/tab-separated-values",
    "txt": "text/plain",
    "json": "application/json",
    "xml": "application/xml",
    "html": "text/html",
    "md": "text/markdown",
    "pdf": "application/pdf",
    "zip": "application/zip",
    "parquet": "application/vnd.apache.parquet",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def _content_spec_for(ctx: RetrieveContext, file: Mapping[str, Any]) -> str:
    """Which spec builds the content request for a file fetched by id."""
    provider = ctx["provider"]
    if provider == "anthropic":
        return "anthropic/files.content"
    if provider == "google":
        return "google/files.content"
    if provider in ("openai", "xai", "openrouter"):
        # Code-execution output files live inside a container.
        ref: Mapping[str, Any] = file.get("ref") or {}
        return (
            "openai/files.content.container"
            if ref.get("containerId")
            else "openai/files.content"
        )
    raise ValueError(f'retrieve_file: no file-content endpoint for provider "{provider}"')


def _build_file_request(
    spec_id: str,
    ctx: RetrieveContext,
    input_: Mapping[str, Any],
    response_type: str,
) -> dict[str, Any]:
    """Build one file request from its spec.

    Two facts the spec cannot work out for itself are passed IN: whether this is
    a browser (an Anthropic download needs a CORS opt-in header there and nowhere
    else) and whether the url is on the provider's OWN host. The second is
    decided at the call site on purpose -- sending a credential to whatever host
    a response named is the risk this guard exists for, and that decision should
    be readable as code.
    """
    built = build_from_spec(
        utility_spec(spec_id),
        {**input_, "browser": is_browser(), "provider": ctx["provider"]},
        make_registry({}),
        ctx["provider"],
        None,
        {
            "baseURL": ctx.get("baseURL") or _DEFAULT_BASE[ctx["provider"]],
            "apiKey": ctx["apiKey"],
            "apiVersion": ANTHROPIC_API_VERSION,
        },
    )
    # The body is dropped, not forwarded: every one of these specs is a GET, and
    # `no_body`/`body` are the builder's own bookkeeping.
    return {
        "url": built.url,
        "method": built.method,
        "headers": built.headers or {},
        "provider": ctx["provider"],
        "model": "files",
        "responseType": response_type,
    }


def _file_request(
    ctx: RetrieveContext, file: Mapping[str, Any], response_type: str
) -> dict[str, Any]:
    """WHICH request fetches this file -- the whole decision, with no I/O in it.

    Shared by both cores so the url-vs-id choice, and the same-host guard that
    decides whether a credential travels, cannot differ between them.
    """
    if file.get("url"):
        base = ctx.get("baseURL") or _DEFAULT_BASE[ctx["provider"]]
        return _build_file_request(
            "files/download.byUrl",
            ctx,
            {"url": file["url"], "sameHost": str(file["url"]).startswith(base)},
            response_type,
        )
    if file.get("id"):
        ref: Mapping[str, Any] = file.get("ref") or {}
        return _build_file_request(
            _content_spec_for(ctx, file),
            ctx,
            {"id": file["id"], "containerId": ref.get("containerId")},
            response_type,
        )
    raise ValueError("retrieve_file: FileOutput has neither `data`, `url`, nor `id`")


def _mime_from_name(name: str | None) -> str | None:
    """Best-effort MIME from a filename extension (fallback when no Content-Type)."""
    if not name or "." not in name:
        return None
    return _EXT_MIME.get(name.rsplit(".", 1)[-1].lower())


def _filename_from_disposition(disposition: str | None) -> str | None:
    """Filename from a Content-Disposition header.

    Prefers the RFC 5987 `filename*=utf-8''<pct-encoded>` form (the real UTF-8
    name) over `filename="..."`.
    """
    if not disposition:
        return None
    extended = re.search(r"filename\*=[^']*''([^;]+)", disposition, re.IGNORECASE)
    if extended:
        try:
            return unquote(extended.group(1).strip())
        except (ValueError, UnicodeDecodeError):
            pass  # fall through to the plain filename
    plain = re.search(r'filename="?([^";]+)"?', disposition, re.IGNORECASE)
    return plain.group(1).strip() if plain else None


def _content_type(headers: Mapping[str, str]) -> str | None:
    value = header(headers, "content-type")
    return value.split(";")[0].strip() if value else None


def _inline_file(file: Mapping[str, Any]) -> RetrievedFile | None:
    """Inline base64 needs no request at all (Google returns files this way)."""
    if not file.get("data"):
        return None
    data = base64_to_bytes(file["data"])
    mime = file.get("mimeType") or _OCTET_STREAM
    return {"bytes": data, "name": file.get("name"), "mimeType": mime, "size": len(data)}


def _retrieved(res: Mapping[str, Any], file: Mapping[str, Any]) -> RetrievedFile:
    """The response -> `{bytes, name, mimeType, size}`.

    The header sources win over the FileOutput's own fields, because they came
    from the server that just sent the bytes.
    """
    headers: Mapping[str, str] = res.get("headers") or {}
    name = _filename_from_disposition(header(headers, "content-disposition")) or file.get("name")
    mime_type = (
        _content_type(headers) or file.get("mimeType") or _mime_from_name(name) or _OCTET_STREAM
    )
    data = bytes(res.get("body") or b"")
    return {"bytes": data, "name": name, "mimeType": mime_type, "size": len(data)}


def retrieve_file(file: Mapping[str, Any], ctx: RetrieveContext) -> RetrievedFile:
    """Fetch the whole file as bytes (buffered) plus its name/type/size."""
    inline = _inline_file(file)
    if inline is not None:
        return inline
    return _retrieved(ctx["fetch"](_file_request(ctx, file, "arraybuffer")), file)


async def aretrieve_file(file: Mapping[str, Any], ctx: RetrieveContext) -> RetrievedFile:
    """The async twin of `retrieve_file`."""
    inline = _inline_file(file)
    if inline is not None:
        return inline
    return _retrieved(await ctx["fetch"](_file_request(ctx, file, "arraybuffer")), file)


def _streamed(res: Mapping[str, Any], file: Mapping[str, Any]) -> FileStream:
    headers: Mapping[str, str] = res.get("headers") or {}
    length = header(headers, "content-length")
    name = _filename_from_disposition(header(headers, "content-disposition")) or file.get("name")
    return {
        "stream": res.get("body"),
        "name": name,
        "mimeType": _content_type(headers) or file.get("mimeType") or _mime_from_name(name),
        "size": int(length) if length else None,
    }


def stream_file(file: Mapping[str, Any], ctx: RetrieveContext) -> FileStream:
    """Stream the file's bytes (un-buffered) plus best-effort name/type/size.

    Iterate the stream straight into a file, an object store, or an HTTP
    response -- nothing is buffered. The stream is an iterator of `bytes`; the
    async twin yields the async flavour of the same.
    """
    if file.get("data"):
        data = base64_to_bytes(file["data"])
        return {
            "stream": iter((data,)),
            "name": file.get("name"),
            "mimeType": file.get("mimeType"),
            "size": len(data),
        }
    return _streamed(ctx["fetch"](_file_request(ctx, file, "stream")), file)


async def astream_file(file: Mapping[str, Any], ctx: RetrieveContext) -> FileStream:
    """The async twin of `stream_file`."""
    if file.get("data"):
        data = base64_to_bytes(file["data"])
        return {
            "stream": _single_chunk(data),
            "name": file.get("name"),
            "mimeType": file.get("mimeType"),
            "size": len(data),
        }
    return _streamed(await ctx["fetch"](_file_request(ctx, file, "stream")), file)


async def _single_chunk(data: bytes) -> AsyncIterator[bytes]:
    """Inline bytes as a one-chunk stream, so both branches return the same shape."""
    yield data


__all__ = [
    "FileStream",
    "RetrieveContext",
    "RetrievedFile",
    "aretrieve_file",
    "astream_file",
    "retrieve_file",
    "stream_file",
]
