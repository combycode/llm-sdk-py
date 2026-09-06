"""Turn a spec's multipart DESCRIPTOR into a real multipart body.

A spec can say that a request is multipart and which fields it carries, but not
what the bytes are -- those come from a `FileAttachment` the caller holds. So the
interpreter emits `{ name, kind: 'file' | 'value', value? }` and this fills in the
one part it cannot: the file itself.

Kept out of `wire/` deliberately. The wire layer has no outbound imports and no
notion of an attachment; this is the seam where spec data meets the caller's
bytes, which makes it llm-layer glue rather than part of the interpreter.

Transposed from `unified-library-ts/src/llm/wire-multipart.ts`.

ONE TRANSPOSITION, and it is the whole reason to read this docstring. TypeScript
returns a `FormData`, a browser type Python has no equivalent of. What it returns
here is the ORDERED list of parts, in exactly the tuple shape the HTTP client
takes for a multipart body: `(name, str)` for a value, and
`(name, (filename, bytes, mime_type))` for the file. Order is preserved for the
same reason TypeScript preserves it -- multipart is an ordered format and some
servers care -- and the encoding stays where the encoding already lives, in the
transport, rather than being duplicated here.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..wire.interpreter import MultipartField, js_string


@dataclass(frozen=True, slots=True)
class MultipartFile:
    """`interface MultipartFile`."""

    #: The bytes, already read.
    data: bytes
    filename: str
    mime_type: str


#: `(name, value)` for a value part; `(name, (filename, data, mime_type))` for the
#: file part.
MultipartPart = tuple[str, Any]


def to_form_data(
    fields: Sequence[MultipartField], file: MultipartFile
) -> list[MultipartPart]:
    """Build the multipart body a multipart spec describes.

    Field ORDER follows the spec, because multipart is an ordered format and some
    servers care. A `file` field with no file supplied is an error rather than an
    omission: a silently fileless upload would be accepted by the type checker and
    rejected by the provider, which is the exact failure mode the specs exist to
    remove.
    """
    form: list[MultipartPart] = []
    for f in fields:
        if f.kind == "file":
            form.append((f.name, (file.filename, file.data, file.mime_type)))
        else:
            # `String(f.value ?? '')` -- nullish, so `false` and `0` still render.
            form.append((f.name, js_string("" if f.value is None else f.value)))
    return form


def encode_multipart(parts: Sequence[MultipartPart]) -> tuple[bytes, str]:
    """The parts as body bytes, and the content-type header that describes them.

    Encoded here rather than left to the transport: the boundary appears in BOTH
    the body and the header, so whoever builds one must build the other. A
    transport handed the parts and asked to encode them would be a second place
    that has to agree about ordering and about how a filename is escaped.
    """
    boundary = f"----combycode{uuid.uuid4().hex}"
    marker = f"--{boundary}".encode()
    crlf = b"\r\n"
    chunks: list[bytes] = []

    for name, value in parts:
        chunks.append(marker + crlf)
        if isinstance(value, tuple):
            filename, data, mime_type = value
            disposition = (
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"'
            )
            chunks.append(disposition.encode("utf-8") + crlf)
            chunks.append(f"Content-Type: {mime_type}".encode() + crlf + crlf)
            chunks.append(bytes(data))
        else:
            disposition = f'Content-Disposition: form-data; name="{name}"'
            chunks.append(disposition.encode("utf-8") + crlf + crlf)
            chunks.append(str(value).encode("utf-8"))
        chunks.append(crlf)

    chunks.append(marker + b"--" + crlf)
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


__all__ = ["MultipartFile", "MultipartPart", "encode_multipart", "to_form_data"]
