"""A conversation as something you can read, archive, or attach to a bug.

Transposed from `unified-library-ts/src/helpers/conversation-export.ts` and
`conversation-zip.ts`.

Two renderings of the same document, differing only in where the media goes:

  `conversation_to_markdown` inlines it as data-URLs, so one file is the whole
  record and nothing can go missing. A conversation with three screenshots in it
  becomes a multi-megabyte string, which is the price of that.

  `conversation_to_zip` pulls each blob out to `media/media-001.png` and links
  it relatively. The markdown stays readable and the archive stays openable by
  anything.

**On the ZIP.** The TypeScript hand-writes the container in STORE mode to keep
its zero-dependency promise -- JavaScript has no archive in its standard
library. Python does, so `zipfile` is that same decision rather than a different
one: still nothing installed. What carries over verbatim is the determinism.
Every entry is stamped 1980-01-01, the epoch DOS timestamps start at, because
an archive that differs byte for byte between two runs of the same input cannot
be diffed, cached, or checksummed -- and an export is exactly the thing someone
will want to compare against yesterday's.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..util.base64 import base64_to_bytes, bytes_to_base64
from ..wire.interpreter import js_json

#: The DOS epoch, which is what a ZipInfo already defaults to -- stated anyway
#: so the intent survives a future default, and because the thing it guards
#: against is one character away: `writestr("name", data)` with a plain string
#: stamps the WALL CLOCK, and only `writestr(ZipInfo(...), data)` does not.
#: Measured, not assumed: a string name produced a 2026 timestamp here.
FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

DEFAULT_MARKDOWN_NAME = "conversation.md"
DEFAULT_MEDIA_DIR = "media"

#: Media parts, and the label each renders under.
_MEDIA_LABELS = {
    "image": "image",
    "audio": "audio",
    "video": "video",
    "document": "document",
}

_GENERATED = ("image_output", "audio_output", "video_output")

_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/ogg": "ogg",
    "video/mp4": "mp4",
    "video/webm": "webm",
    "application/pdf": "pdf",
}

_DATA_URL = re.compile(r"^data:(.*?);base64,(.*)$", re.DOTALL)


def ext_of(mime: str) -> str:
    """A file extension for a mime type, falling back to its subtype.

    Sanitised rather than trusted: the mime comes off a provider response, and
    it ends up in a path inside an archive somebody will extract.
    """
    key = mime.lower().split(";")[0].strip()
    if key in _EXT:
        return _EXT[key]
    subtype = key.split("/", 1)[1] if "/" in key else key
    return re.sub(r"[^a-z0-9]", "", subtype) or "bin"


def _source_url(source: Mapping[str, Any]) -> str:
    """A data source as something a markdown link can point at."""
    kind = source.get("type")
    if kind == "url":
        return str(source.get("url") or "")
    if kind == "base64":
        return f"data:{source.get('mimeType')};base64,{source.get('data')}"
    if kind == "buffer":
        return f"data:{source.get('mimeType')};base64,{bytes_to_base64(source.get('data') or b'')}"
    if kind == "file":
        return f"file:{source.get('fileId')}"
    if kind == "provider_ref":
        return f"ref:{source.get('refId')}"
    if kind == "path":
        return str(source.get("path") or "")
    return ""


def _bytes_of(source: Mapping[str, Any]) -> tuple[bytes, str] | str | None:
    """Raw bytes and mime when the media is embedded, else an external URL.

    A `data:` URL counts as embedded even though it arrived as a url: it carries
    the whole blob, and leaving it in the markdown is the multi-megabyte line
    the archive exists to avoid.
    """
    kind = source.get("type")
    if kind == "base64":
        return base64_to_bytes(str(source.get("data") or "")), str(source.get("mimeType") or "")
    if kind == "buffer":
        data = source.get("data") or b""
        return bytes(data), str(source.get("mimeType") or "")
    if kind == "url":
        url = str(source.get("url") or "")
        match = _DATA_URL.match(url)
        if match:
            return base64_to_bytes(match.group(2)), match.group(1)
        return url
    if kind == "path":
        return str(source.get("path") or "")
    if kind == "file":
        return f"file:{source.get('fileId')}"
    if kind == "provider_ref":
        return f"ref:{source.get('refId')}"
    return None


def _render_part(part: Mapping[str, Any], link: Callable[[Mapping[str, Any], str], str]) -> str:
    kind = str(part.get("type") or "")
    if kind == "text":
        return str(part.get("text") or "")
    if kind in _MEDIA_LABELS:
        return link(part, _MEDIA_LABELS[kind])
    if kind in _GENERATED:
        return f"[generated {kind.replace('_output', '')}: {part.get('mediaId')}]"
    if kind == "tool_call":
        # Indented here and compact for tool_result, exactly as the
        # TypeScript renders each: a call is something a person reads in a
        # transcript, a result is a payload.
        args = json.dumps(part.get("arguments") or {}, indent=2)
        return f"```tool_call {part.get('name')}\n{args}\n```"
    if kind == "tool_result":
        body = part.get("content")
        text = body if isinstance(body, str) else js_json(body)
        return f"```tool_result\n{text}\n```"
    return ""


def _render(
    messages: Sequence[Mapping[str, Any]],
    title: str | None,
    link: Callable[[Mapping[str, Any], str], str],
) -> str:
    out: list[str] = []
    if title:
        out.append(f"# {title}\n")
    for message in messages:
        out.append(f"## {message.get('role')}\n")
        content = message.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, Sequence):
            out.append(
                "\n\n".join(_render_part(p, link) for p in content if isinstance(p, Mapping))
            )
        else:
            out.append("")
        out.append("")
    return "\n".join(out).strip() + "\n"


def conversation_to_markdown(
    messages: Sequence[Mapping[str, Any]],
    *,
    inline_media: bool = True,
    title: str | None = None,
) -> str:
    """The conversation as one self-contained Markdown document.

    `inline_media=False` replaces each blob with a `[image]` placeholder, for
    when the transcript is what matters and the attachments are noise.
    """

    def link(part: Mapping[str, Any], label: str) -> str:
        if not inline_media:
            return f"[{label}]"
        url = _source_url(part.get("source") or {})
        return f"![image]({url})" if label == "image" else f"[{label}]({url})"

    return _render(messages, title, link)


@dataclass(frozen=True)
class ConversationZip:
    """What `conversation_to_zip` produced."""

    #: The archive itself.
    bytes: bytes
    #: The same markdown that is inside it, for previewing without unpacking.
    markdown: str
    #: How many blobs were pulled out into files.
    media_count: int


def conversation_to_zip(
    messages: Sequence[Mapping[str, Any]],
    *,
    title: str | None = None,
    markdown_name: str = DEFAULT_MARKDOWN_NAME,
    media_dir: str = DEFAULT_MEDIA_DIR,
) -> ConversationZip:
    """The conversation as an archive: one Markdown file, media beside it."""
    extracted: list[tuple[str, bytes]] = []

    def link(part: Mapping[str, Any], label: str) -> str:
        got = _bytes_of(part.get("source") or {})
        if got is None:
            return f"[{label}]"
        if isinstance(got, str):
            href = got
        else:
            payload, mime = got
            name = f"{media_dir}/media-{len(extracted) + 1:03d}.{ext_of(mime)}"
            extracted.append((name, payload))
            href = name
        if not href:
            return f"[{label}]"
        return f"![image]({href})" if label == "image" else f"[{label}]({href})"

    markdown = _render(messages, title, link)

    buffer = io.BytesIO()
    # STORE, not DEFLATE: media blobs are already compressed, so deflating them
    # costs time and saves nothing -- the same choice the TypeScript makes.
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        # The markdown first, so it is what an extractor lists at the top.
        for name, payload in [
            (markdown_name, markdown.encode("utf-8")),
            *extracted,
        ]:
            # A ZipInfo, never a bare name: the string form of writestr is
            # what reaches for the clock.
            info = zipfile.ZipInfo(name, date_time=FIXED_TIMESTAMP)
            archive.writestr(info, payload)

    return ConversationZip(
        bytes=buffer.getvalue(), markdown=markdown, media_count=len(extracted)
    )


__all__ = [
    "DEFAULT_MARKDOWN_NAME",
    "DEFAULT_MEDIA_DIR",
    "FIXED_TIMESTAMP",
    "ConversationZip",
    "conversation_to_markdown",
    "conversation_to_zip",
    "ext_of",
]
