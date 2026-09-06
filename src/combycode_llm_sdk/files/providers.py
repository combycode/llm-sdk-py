"""The four file stores, built from the shared wire specs.

Three of them are one multipart POST and three plain calls. Google is not: its
Files API is a RESUMABLE upload, so a file goes up in two requests and the second
one addresses a URL the first returns in a response header -- which no spec can
describe, because the spec would have to know the answer before asking. That
asymmetry is the whole reason this file is not four copies of one class.

Every request is built by `build_from_spec` from the same JSON the TypeScript
reads, and sent through the injected fetch. Nothing here opens a socket.

Transposed from `unified-library-ts/src/llm/providers/*/files.ts`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from ..llm.providers.anthropic.constants import ANTHROPIC_API_VERSION
from ..llm.wire_multipart import MultipartFile, encode_multipart, to_form_data
from ..llm.wire_transforms import make_registry
from ..util.http import header
from ..wire.interpreter import build_from_spec
from ..wire.service_specs import service_spec
from .attachment import FileAttachment, now_ms
from .provider_adapter import FileUploadResult, RemoteFileInfo

#: Google reaps uploads after 48 hours whether or not anyone is still using them.
FORTY_EIGHT_HOURS_MS = 48 * 60 * 60 * 1000

#: Epoch SECONDS on the wire (OpenAI, xAI) -> epoch milliseconds in here.
_SECONDS_TO_MS = 1000


def _status(response: Any) -> int:
    value = response.get("status") if isinstance(response, Mapping) else None
    return value if isinstance(value, int) else 0


def _body(response: Any) -> Mapping[str, Any]:
    raw = (response.get("body") if isinstance(response, Mapping) else None) or {}
    if isinstance(raw, (str, bytes)):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return raw if isinstance(raw, Mapping) else {}


def _rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _epoch_ms(value: Any) -> float | None:
    """Epoch seconds from the wire, as milliseconds. None stays None.

    Zero is a value, not an absence -- `if not value` would turn the epoch into
    "never expires", so the test is explicitly against None.
    """
    if value is None:
        return None
    try:
        return float(value) * _SECONDS_TO_MS
    except (TypeError, ValueError):
        return None


def _iso_ms(value: Any) -> float:
    """An RFC 3339 timestamp (Anthropic, Google) as epoch milliseconds.

    An unparseable stamp yields 0 rather than raising: a listing is not worth
    failing over a date field nobody reads.
    """
    if not isinstance(value, str) or not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp() * 1000
    except ValueError:
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class _SpecFileAdapter:
    """The shape three of the four providers share exactly."""

    #: Set by each subclass.
    name = ""
    expires_after_ms: float | None = None
    max_file_size = 0
    supported_types: tuple[str, ...] | None = None
    default_base_url = ""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url or self.default_base_url
        self._registry = make_registry({})

    # -- request building ----------------------------------------------------

    def _config(self) -> dict[str, Any]:
        return {"baseURL": self._base_url, "apiKey": self._api_key}

    def _request(
        self, spec_id: str, payload: Mapping[str, Any], file: MultipartFile | None = None
    ) -> dict[str, Any]:
        """One request from its spec, plus the routing fields the wire never sees."""
        built = build_from_spec(
            service_spec(spec_id),
            dict(payload),
            self._registry,
            self.name,
            None,
            self._config(),
        )
        request: dict[str, Any] = {
            "url": built.url,
            "method": built.method or "POST",
            "headers": dict(built.headers or {}),
            "provider": self.name,
            # Named, not derived. These endpoints are model-agnostic, and a
            # queue keyed by a model that does not exist would be nobody's.
            "model": "files",
            "responseType": "json",
        }
        if built.multipart and file is not None:
            encoded, content_type = encode_multipart(to_form_data(built.multipart, file))
            request["body"] = encoded
            request["rawBody"] = True
            request["headers"]["content-type"] = content_type
        elif not built.no_body:
            request["body"] = built.body
        return request

    def build_upload_request(self, file: FileAttachment, data: bytes) -> dict[str, Any]:
        return self._request(
            f"{self.name}/files.upload",
            {},
            MultipartFile(data=data, filename=file.filename, mime_type=file.mime_type),
        )

    def build_delete_request(self, remote_id: str) -> dict[str, Any]:
        return self._request(f"{self.name}/files.delete", {"remoteId": remote_id})

    def build_get_info_request(self, remote_id: str) -> dict[str, Any]:
        return self._request(f"{self.name}/files.getInfo", {"remoteId": remote_id})

    def build_list_request(self) -> dict[str, Any]:
        return self._request(f"{self.name}/files.list", {})

    # -- operations ----------------------------------------------------------

    def upload(self, file: FileAttachment, fetch: Any) -> FileUploadResult:
        response = fetch(self.build_upload_request(file, file.to_bytes()))
        if _status(response) >= 400:
            raise RuntimeError(
                f"{self.name} file upload failed ({_status(response)}): {_body(response)}"
            )
        return self._upload_result(_body(response))

    def _upload_result(self, body: Mapping[str, Any]) -> FileUploadResult:
        return FileUploadResult(remote_id=str(body.get("id") or ""), expires_at=None)

    def delete(self, remote_id: str, fetch: Any) -> None:
        fetch(self.build_delete_request(remote_id))

    def get_info(self, remote_id: str, fetch: Any) -> RemoteFileInfo | None:
        response = fetch(self.build_get_info_request(remote_id))
        if _status(response) >= 400:
            return None
        return self._info(_body(response))

    def list(self, fetch: Any) -> list[RemoteFileInfo]:
        response = fetch(self.build_list_request())
        if _status(response) >= 400:
            return []
        return [self._info(row) for row in _rows(_body(response).get("data"))]

    def _info(self, row: Mapping[str, Any]) -> RemoteFileInfo:
        raise NotImplementedError


class OpenAIFileAdapter(_SpecFileAdapter):
    """`POST /v1/files`. Kept until deleted."""

    name = "openai"
    expires_after_ms: float | None = None
    max_file_size = 50_000_000
    supported_types: tuple[str, ...] | None = None
    default_base_url = "https://api.openai.com"

    def _upload_result(self, body: Mapping[str, Any]) -> FileUploadResult:
        return FileUploadResult(
            remote_id=str(body.get("id") or ""),
            expires_at=_epoch_ms(body.get("expires_at")),
        )

    def _info(self, row: Mapping[str, Any]) -> RemoteFileInfo:
        return RemoteFileInfo(
            remote_id=str(row.get("id") or ""),
            filename=str(row.get("filename") or ""),
            size_bytes=_int(row.get("bytes")),
            created_at=_epoch_ms(row.get("created_at")) or 0.0,
            expires_at=_epoch_ms(row.get("expires_at")),
        )


class XaiFileAdapter(_SpecFileAdapter):
    """`POST /v1/files`, five MIME types, no expiry."""

    name = "xai"
    expires_after_ms: float | None = None
    max_file_size = 48_000_000
    supported_types = (
        "text/plain",
        "text/markdown",
        "text/csv",
        "application/json",
        "application/pdf",
    )
    default_base_url = "https://api.x.ai"

    def _info(self, row: Mapping[str, Any]) -> RemoteFileInfo:
        return RemoteFileInfo(
            remote_id=str(row.get("id") or ""),
            filename=str(row.get("filename") or ""),
            size_bytes=_int(row.get("bytes")),
            created_at=_epoch_ms(row.get("created_at")) or 0.0,
        )


class AnthropicFileAdapter(_SpecFileAdapter):
    """`POST /v1/files` (beta). Timestamps arrive as RFC 3339, not epoch."""

    name = "anthropic"
    expires_after_ms: float | None = None
    max_file_size = 500_000_000
    supported_types = (
        "application/pdf",
        "text/plain",
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
    )
    default_base_url = "https://api.anthropic.com"

    def _config(self) -> dict[str, Any]:
        return {
            "baseURL": self._base_url,
            "apiKey": self._api_key,
            "apiVersion": ANTHROPIC_API_VERSION,
        }

    def _info(self, row: Mapping[str, Any]) -> RemoteFileInfo:
        return RemoteFileInfo(
            remote_id=str(row.get("id") or ""),
            filename=str(row.get("filename") or ""),
            size_bytes=_int(row.get("size_bytes")),
            created_at=_iso_ms(row.get("created_at")),
        )


def google_file_name(remote_id: str) -> str:
    """Reduce any form of Google file id to the bare name the REST path wants.

    Three forms reach this, and only two used to work:

    - `https://.../v1beta/files/abc` -- the `uri` this adapter hands back from
      `upload()` and `list()`, matched on `/files/`
    - `abc` -- a bare name, passed through
    - `files/abc` -- Google's CANONICAL resource name, the `name` field its own
      API returns

    The third fell through the `/files/` test (no leading slash) and produced
    `/v1beta/files/files/abc`, a 404. It never broke this library's own round
    trip, because upload and list return the `uri`; it broke the moment a caller
    passed the id Google itself had given them.
    """
    if "/files/" in remote_id:
        return remote_id.split("/files/")[-1] or remote_id
    if remote_id.startswith("files/"):
        return remote_id[len("files/") :]
    return remote_id


class GoogleFileAdapter(_SpecFileAdapter):
    """Resumable upload, and a 48-hour reaper.

    Two requests per upload: one that reserves the slot and answers with a URL in
    a header, one that sends the bytes to that URL. The second is still built
    from a spec -- the URL is an INPUT to it, the same way a batch id is -- so
    the only thing this class does that the others do not is read a header.
    """

    name = "google"
    expires_after_ms: float | None = FORTY_EIGHT_HOURS_MS
    max_file_size = 2_000_000_000
    supported_types: tuple[str, ...] | None = None
    default_base_url = "https://generativelanguage.googleapis.com"

    def build_start_upload_request(
        self, file: FileAttachment, byte_length: int
    ) -> dict[str, Any]:
        return self._request(
            "google/files.startUpload",
            {
                "filename": file.filename,
                "mimeType": file.mime_type,
                "byteLength": byte_length,
            },
        )

    def build_finish_upload_request(
        self, upload_url: str, file: FileAttachment, data: bytes
    ) -> dict[str, Any]:
        request = self._request(
            "google/files.finishUpload",
            {"uploadUrl": upload_url, "mimeType": file.mime_type},
        )
        request["body"] = data
        request["rawBody"] = True
        return request

    def build_delete_request(self, remote_id: str) -> dict[str, Any]:
        return self._request("google/files.delete", {"name": google_file_name(remote_id)})

    def build_get_info_request(self, remote_id: str) -> dict[str, Any]:
        return self._request("google/files.getInfo", {"name": google_file_name(remote_id)})

    def build_list_request(self) -> dict[str, Any]:
        return self._request("google/files.list", {})

    def upload(self, file: FileAttachment, fetch: Any) -> FileUploadResult:
        data = file.to_bytes()

        started = fetch(self.build_start_upload_request(file, len(data)))
        if _status(started) >= 400:
            raise RuntimeError(
                f"google file upload start failed ({_status(started)}): {_body(started)}"
            )

        # Header names are case-insensitive, and guessing three casings still
        # misses whichever fourth one a server or runtime decides to send.
        headers = started.get("headers") if isinstance(started, Mapping) else None
        upload_url = header(headers or {}, "x-goog-upload-url")
        if not upload_url:
            raise RuntimeError(
                "google file upload: the start response carried no "
                "x-goog-upload-url header, so there is nowhere to send the bytes"
            )

        finished = fetch(self.build_finish_upload_request(upload_url, file, data))
        if _status(finished) >= 400:
            raise RuntimeError(
                f"google file upload failed ({_status(finished)}): {_body(finished)}"
            )

        body = _body(finished)
        described = body.get("file")
        described = described if isinstance(described, Mapping) else body
        expiration = described.get("expirationTime")
        return FileUploadResult(
            remote_id=str(described.get("uri") or ""),
            # No expiry in the answer still means 48 hours -- Google reaps on
            # its own schedule, and recording "unknown" as "never" is how a
            # stale reference gets sent two days later.
            expires_at=_iso_ms(expiration)
            if expiration
            else now_ms() + FORTY_EIGHT_HOURS_MS,
        )

    def list(self, fetch: Any) -> list[RemoteFileInfo]:
        response = fetch(self.build_list_request())
        if _status(response) >= 400:
            return []
        return [self._info(row) for row in _rows(_body(response).get("files"))]

    def _info(self, row: Mapping[str, Any]) -> RemoteFileInfo:
        expiration = row.get("expirationTime")
        return RemoteFileInfo(
            remote_id=str(row.get("uri") or ""),
            filename=str(row.get("displayName") or ""),
            # `sizeBytes` is a STRING here, unlike everywhere else.
            size_bytes=_int(row.get("sizeBytes")),
            created_at=_iso_ms(row.get("createTime")),
            expires_at=_iso_ms(expiration) if expiration else None,
        )


#: Provider name -> its adapter class, for `FilesRegistry.for_provider`.
FILE_ADAPTERS: Mapping[str, type[_SpecFileAdapter]] = {
    "anthropic": AnthropicFileAdapter,
    "google": GoogleFileAdapter,
    "openai": OpenAIFileAdapter,
    "xai": XaiFileAdapter,
}


__all__ = [
    "FILE_ADAPTERS",
    "FORTY_EIGHT_HOURS_MS",
    "AnthropicFileAdapter",
    "GoogleFileAdapter",
    "OpenAIFileAdapter",
    "XaiFileAdapter",
    "google_file_name",
]
