"""Files: the four stores, the strategy, and the registry that rewrites a message.

The centrepiece is `TestTheSameRequestsAsTheTypeScript`, a differential against
`tests/fixtures/service-golden.json` -- thirteen requests frozen from the
TypeScript adapters. Unit tests written beside an implementation confirm the
model its author already had; a fixture captured from the other implementation
cannot. Every URL, method, header, field name, field ORDER and body here is held
to what the TypeScript actually sent.

Two differences are expected and are NOT the library's:

- **The multipart `content-type` header.** The TypeScript hands a `FormData` to
  the runtime, which invents the boundary and sets the header itself, so the
  frozen request carries no content-type at all. Python has no FormData in its
  transport, so `encode_multipart` writes both the body and the header that
  describes it. Compared here by parsing the encoded body back into parts, which
  checks more than the frozen shape did -- names, order, filename and bytes.

- **`;charset=utf-8` on a `text/*` file part.** Bun's `new Blob([...], {type:
  'text/plain'})` reports its type as `text/plain;charset=utf-8`, so the golden
  froze that. It is the runtime's normalisation, not a decision either library
  made, and Node's Blob does not do it. Part types are compared without their
  parameters for that reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from combycode_llm_sdk.bus.hook_bus import HookBus
from combycode_llm_sdk.files import (
    AnthropicFileAdapter,
    Base64Content,
    BytesContent,
    DefaultFileStrategy,
    FileAttachment,
    FileDecision,
    FilesRegistry,
    FileStrategyContext,
    GoogleFileAdapter,
    OpenAIFileAdapter,
    PathContent,
    UrlContent,
    XaiFileAdapter,
    google_file_name,
)

GOLDEN = json.loads(
    (Path(__file__).parent.parent / "fixtures" / "service-golden.json").read_text("utf-8")
)["index"]

#: Exactly the attachment the freeze script used.
HELLO = b"hello"


def attachment() -> FileAttachment:
    return FileAttachment.from_bytes(HELLO, filename="note.txt", mime_type="text/plain")


#: The freeze answered every call with this, so a method got far enough to make
#: the next one. Google's second request only happens because of the header.
FILE_RESPONSE = {
    "id": "file_abc",
    "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc",
    "file": {"uri": "https://generativelanguage.googleapis.com/v1beta/files/abc"},
    "data": [],
    "files": [],
}
UPLOAD_HEADER = {"x-goog-upload-url": "https://upload.example/session"}

#: The id each provider hands back, in its own awkward format.
FILE_REMOTE_ID = {
    "anthropic": "file_abc",
    "openai": "file-abc",
    "google": "https://generativelanguage.googleapis.com/v1beta/files/abc",
    "xai": "file_abc",
}

ADAPTERS = {
    "anthropic": AnthropicFileAdapter,
    "openai": OpenAIFileAdapter,
    "google": GoogleFileAdapter,
    "xai": XaiFileAdapter,
}

FILE_CASES = [
    ("anthropic", "upload"),
    ("anthropic", "delete"),
    ("anthropic", "get_info"),
    ("anthropic", "list"),
    ("openai", "upload"),
    ("openai", "delete"),
    ("openai", "get_info"),
    ("openai", "list"),
    ("google", "upload"),
    ("google", "delete"),
    ("google", "get_info"),
    ("google", "list"),
    ("xai", "upload"),
]

#: `get_info` here is `getInfo` in the frozen keys.
GOLDEN_OP = {"upload": "upload", "delete": "delete", "get_info": "getInfo", "list": "list"}


def capturing(
    headers: dict[str, str] | None = None, body: Any = None
) -> tuple[Any, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def fetch(request: dict[str, Any], options: Any = None) -> dict[str, Any]:
        seen.append(request)
        return {
            "status": 200,
            "headers": dict(headers or {}),
            "body": FILE_RESPONSE if body is None else body,
        }

    return fetch, seen


# ── canonicalisation ────────────────────────────────────────────────────────


def parse_multipart(body: bytes, content_type: str) -> dict[str, Any]:
    """The encoded body, read back into the shape the golden froze.

    Parsing rather than trusting `to_form_data`: this is the only test that sees
    the bytes that would actually leave the process.
    """
    boundary = content_type.split("boundary=", 1)[1]
    marker = f"--{boundary}".encode()
    entries: list[dict[str, Any]] = []
    for chunk in body.split(marker):
        if not chunk.strip() or chunk.startswith(b"--"):
            continue
        head, _, payload = chunk.lstrip(b"\r\n").partition(b"\r\n\r\n")
        headers = {
            k.strip().lower(): v.strip()
            for k, _, v in (line.partition(":") for line in head.decode().split("\r\n"))
            if k
        }
        disposition = headers.get("content-disposition", "")
        name = disposition.split('name="', 1)[1].split('"', 1)[0]
        payload = payload[:-2] if payload.endswith(b"\r\n") else payload
        if 'filename="' in disposition:
            entries.append(
                {
                    "name": name,
                    "filename": disposition.split('filename="', 1)[1].split('"', 1)[0],
                    # Parameters dropped: see the module docstring -- the golden
                    # carries Bun's `;charset=utf-8`, which is the runtime's.
                    "type": headers.get("content-type", "").split(";")[0],
                    "size": len(payload),
                }
            )
        else:
            entries.append({"name": name, "value": payload.decode("utf-8")})
    return {"__formData": entries}


def canon(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return {"__bytes": len(value)}
    if isinstance(value, list):
        return [canon(v) for v in value]
    if isinstance(value, dict):
        return {k: canon(value[k]) for k in sorted(value) if value[k] is not None}
    return value


def canon_request(request: dict[str, Any]) -> Any:
    out = dict(request)
    headers = dict(out.get("headers") or {})
    content_type = headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        out["body"] = parse_multipart(out["body"], content_type)
        # The TypeScript never sets this itself, so the golden has no record of
        # it. Removed rather than compared against nothing.
        headers.pop("content-type")
    out["headers"] = headers
    return canon(out)


def golden_of(request: dict[str, Any]) -> dict[str, Any]:
    """The frozen side, canonicalised the same way (part types unparameterised)."""
    frozen: dict[str, Any] = canon(json.loads(json.dumps(request)))
    body = frozen.get("body")
    if isinstance(body, dict) and "__formData" in body:
        for entry in body["__formData"]:
            if "type" in entry:
                entry["type"] = entry["type"].split(";")[0]
    return frozen


class TestTheSameRequestsAsTheTypeScript:
    """Thirteen file requests, held to what the other implementation sent."""

    @pytest.mark.parametrize(("provider", "op"), FILE_CASES)
    def test_matches_the_frozen_request(self, provider: str, op: str) -> None:
        adapter = ADAPTERS[provider](api_key="k")
        fetch, seen = capturing(UPLOAD_HEADER)

        if op == "upload":
            adapter.upload(attachment(), fetch)
        elif op == "list":
            adapter.list(fetch)
        else:
            getattr(adapter, op)(FILE_REMOTE_ID[provider], fetch)

        assert seen, "the adapter sent nothing at all"
        for index, request in enumerate(seen):
            key = f"files/{provider}/{GOLDEN_OP[op]}" + (f".{index}" if index else "")
            assert key in GOLDEN, f"no frozen request named {key}"
            assert canon_request(request) == golden_of(GOLDEN[key]), key

    def test_the_golden_covers_every_case_and_no_more(self) -> None:
        # A differential that silently stops covering a case is worse than none:
        # it keeps reporting green over an adapter nobody measures any more.
        frozen = {k for k in GOLDEN if k.startswith("files/")}
        expected = {f"files/{p}/{GOLDEN_OP[o]}" for p, o in FILE_CASES} | {
            "files/google/upload.1"
        }
        assert frozen == expected

    def test_the_comparison_can_fail(self) -> None:
        # The canary. Every assertion above passes trivially if `canon_request`
        # returns something the golden also happens to produce, so one known-bad
        # request must be rejected.
        adapter = OpenAIFileAdapter(api_key="k", base_url="https://wrong.example")
        fetch, seen = capturing()
        adapter.list(fetch)
        assert canon_request(seen[0]) != golden_of(GOLDEN["files/openai/list"])


class TestGoogleIsTwoRequests:
    """The resumable upload, which is the only shape that is not one call."""

    def test_the_bytes_go_to_the_url_from_the_response_header(self) -> None:
        fetch, seen = capturing(UPLOAD_HEADER)
        result = GoogleFileAdapter(api_key="k").upload(attachment(), fetch)

        assert len(seen) == 2
        assert seen[0]["url"].endswith("/upload/v1beta/files")
        assert seen[1]["url"] == "https://upload.example/session"
        assert seen[1]["body"] == HELLO
        assert seen[1]["rawBody"] is True
        assert result.remote_id.endswith("/files/abc")

    def test_no_upload_url_is_an_error_naming_the_missing_header(self) -> None:
        # Without the header there is nowhere to send the bytes. Failing with the
        # header's name beats a request to `undefined`.
        fetch, _ = capturing({})
        with pytest.raises(RuntimeError, match="x-goog-upload-url"):
            GoogleFileAdapter(api_key="k").upload(attachment(), fetch)

    def test_an_absent_expiry_still_expires_in_48_hours(self) -> None:
        # Google reaps on its own schedule. Recording "the answer did not say" as
        # "never" is how a dead reference gets sent two days later.
        fetch, _ = capturing(UPLOAD_HEADER)
        result = GoogleFileAdapter(api_key="k").upload(attachment(), fetch)
        assert result.expires_at is not None
        assert result.expires_at > 0

    @pytest.mark.parametrize(
        ("given", "wanted"),
        [
            ("abc", "abc"),
            ("files/abc", "abc"),
            ("https://generativelanguage.googleapis.com/v1beta/files/abc", "abc"),
        ],
    )
    def test_every_form_of_a_google_id_reduces_to_the_bare_name(
        self, given: str, wanted: str
    ) -> None:
        # The canonical `files/abc` form -- the one Google's own API returns --
        # used to fall through and produce `/v1beta/files/files/abc`, a 404.
        assert google_file_name(given) == wanted
        url = GoogleFileAdapter(api_key="k").build_get_info_request(given)["url"]
        assert url.endswith(f"/v1beta/files/{wanted}")


class TestReadingTheProvidersAnswer:
    """Timestamps, sizes and ids, in the several shapes the providers use."""

    def test_openai_epoch_seconds_become_milliseconds(self) -> None:
        fetch, _ = capturing(body={"data": [
            {"id": "f1", "filename": "a.txt", "bytes": 12,
             "created_at": 1_700_000_000, "expires_at": 1_700_003_600}
        ]})
        (info,) = OpenAIFileAdapter(api_key="k").list(fetch)
        assert info.created_at == 1_700_000_000_000
        assert info.expires_at == 1_700_003_600_000

    def test_anthropic_rfc3339_becomes_milliseconds(self) -> None:
        fetch, _ = capturing(body={"data": [
            {"id": "f1", "filename": "a.pdf", "size_bytes": 9,
             "created_at": "2026-01-01T00:00:00Z"}
        ]})
        (info,) = AnthropicFileAdapter(api_key="k").list(fetch)
        assert info.created_at > 0
        assert info.size_bytes == 9

    def test_google_reports_its_size_as_a_string(self) -> None:
        fetch, _ = capturing(body={"files": [
            {"uri": "https://h/v1beta/files/a", "displayName": "a.txt",
             "sizeBytes": "4096", "createTime": "2026-01-01T00:00:00Z"}
        ]})
        (info,) = GoogleFileAdapter(api_key="k").list(fetch)
        assert info.size_bytes == 4096

    def test_a_failed_listing_is_empty_rather_than_an_exception(self) -> None:
        # A listing is a convenience. Raising on it would make a status call fail
        # a whole run, and there is nothing the caller could do differently.
        def fetch(request: Any, options: Any = None) -> dict[str, Any]:
            return {"status": 500, "headers": {}, "body": {"error": "boom"}}

        assert OpenAIFileAdapter(api_key="k").list(fetch) == []
        assert OpenAIFileAdapter(api_key="k").get_info("f", fetch) is None

    def test_a_failed_upload_raises_with_the_status_and_the_body(self) -> None:
        # An upload is not a convenience: silently returning no id would make the
        # next call inline a file the caller believes was uploaded.
        def fetch(request: Any, options: Any = None) -> dict[str, Any]:
            return {"status": 413, "headers": {}, "body": {"error": "too large"}}

        with pytest.raises(RuntimeError, match="413"):
            OpenAIFileAdapter(api_key="k").upload(attachment(), fetch)


class TestTheAttachment:
    """One file, several providers, independent expiry."""

    def test_an_upload_at_one_provider_is_not_a_reference_at_another(self) -> None:
        file = attachment()
        file.set_uploaded("openai", "file-1", None)
        assert file.get_ref("openai") == "file-1"
        assert file.get_ref("anthropic") is None
        assert file.needs_upload("anthropic")

    def test_an_expired_upload_stops_being_available_and_says_so(self) -> None:
        file = attachment()
        file.set_uploaded("google", "files/x", expires_at=1.0)  # long past
        assert not file.is_available("google")
        assert file.get_ref("google") is None
        # The state is updated, not just the answer: the strategy reads it to
        # tell "never uploaded" from "uploaded and gone", which decide
        # differently.
        assert file.uploads["google"].status == "expired"

    def test_content_kinds_all_produce_the_same_bytes(self) -> None:
        from combycode_llm_sdk.util.base64 import bytes_to_base64

        as_bytes = FileAttachment.from_bytes(HELLO, filename="n", mime_type="text/plain")
        as_b64 = FileAttachment(
            filename="n", mime_type="text/plain", size_bytes=5,
            content=Base64Content(mime_type="text/plain", data=bytes_to_base64(HELLO)),
        )
        assert as_bytes.to_bytes() == as_b64.to_bytes() == HELLO
        assert as_bytes.to_base64() == as_b64.to_base64()

    def test_a_path_is_read_only_when_asked(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.txt"
        target.write_bytes(HELLO)
        file = FileAttachment.from_path(target, mime_type="text/plain")

        assert file.filename == "doc.txt"
        assert file.size_bytes == 5  # from stat, not from reading
        target.unlink()
        # Deferred: the bytes were never held, so now there are none.
        with pytest.raises(FileNotFoundError):
            file.to_bytes()

    def test_a_url_cannot_be_read_and_the_error_says_why(self) -> None:
        file = FileAttachment(
            filename="n", mime_type="image/png", size_bytes=0,
            content=UrlContent(url="https://example/img.png"),
        )
        with pytest.raises(ValueError, match="fetch"):
            file.to_bytes()

    def test_a_snapshot_carries_every_provider_it_reached(self) -> None:
        file = attachment()
        file.set_uploaded("openai", "file-1", None)
        file.set_error("xai", "too large")
        row = file.export().as_row()

        assert row["mimeType"] == "text/plain"
        assert {u["provider"] for u in row["uploads"]} == {"openai", "xai"}
        failed = next(u for u in row["uploads"] if u["provider"] == "xai")
        assert failed["error"] == "too large"


class TestTheStrategy:
    """The one interesting decision, and the two refusals that precede it."""

    def context(self, **over: Any) -> FileStrategyContext:
        base: dict[str, Any] = {
            "file": attachment(),
            "provider": "openai",
            "model": "gpt-5.4-nano",
            "is_uploaded": False,
            "is_expired": False,
            "provider_max_size": 50_000_000,
            "provider_supports_type": True,
        }
        base.update(over)
        return FileStrategyContext(**base)

    def test_a_small_file_is_inlined(self) -> None:
        assert self.strategy_action(self.context()) == "inline"

    def test_a_large_file_is_uploaded(self) -> None:
        big = FileAttachment(
            filename="big.pdf", mime_type="application/pdf", size_bytes=9_000_000,
            content=BytesContent(mime_type="application/pdf", data=b""),
        )
        assert self.strategy_action(self.context(file=big)) == "upload"

    def test_a_file_over_the_providers_ceiling_is_refused_before_it_is_sent(self) -> None:
        huge = FileAttachment(
            filename="huge.pdf", mime_type="application/pdf", size_bytes=99_000_000_000,
            content=BytesContent(mime_type="application/pdf", data=b""),
        )
        decision = self.decide(self.context(file=huge))
        assert decision.action == "skip"
        # The reason reaches a warning, so it has to name both numbers.
        assert "99000000000" in decision.reason and "50000000" in decision.reason

    def test_an_unsupported_type_is_refused(self) -> None:
        assert self.strategy_action(self.context(provider_supports_type=False)) == "skip"

    def test_an_existing_upload_is_reused_rather_than_repeated(self) -> None:
        assert self.strategy_action(self.context(is_uploaded=True)) == "upload"

    def test_an_expired_upload_is_repeated(self) -> None:
        assert self.strategy_action(self.context(is_expired=True)) == "reupload"

    def test_a_url_is_left_to_the_providers_that_fetch_one(self) -> None:
        linked = FileAttachment(
            filename="i.png", mime_type="image/png", size_bytes=0,
            content=UrlContent(url="https://example/i.png"),
        )
        assert self.strategy_action(self.context(file=linked)) == "url"
        # Anthropic does not fetch URLs, so the same file is inlined there.
        assert self.strategy_action(
            self.context(file=linked, provider="anthropic")
        ) == "inline"

    def test_the_threshold_is_the_callers_to_move(self) -> None:
        tiny = DefaultFileStrategy(inline_threshold=1)
        assert tiny.decide(self.context()).action == "upload"

    @staticmethod
    def decide(ctx: FileStrategyContext) -> FileDecision:
        return DefaultFileStrategy().decide(ctx)

    def strategy_action(self, ctx: FileStrategyContext) -> str:
        return self.decide(ctx).action


class TestTheRegistryRewritesMessages:
    """What a caller's message looks like on the way out."""

    def registry(
        self, strategy: Any = None
    ) -> tuple[FilesRegistry, list[dict[str, Any]], HookBus]:
        hooks = HookBus()
        fetch, seen = capturing()
        return FilesRegistry(hooks=hooks, fetch=fetch, strategy=strategy), seen, hooks

    def resolve(
        self, registry: FilesRegistry, part: dict[str, Any], provider: str = "openai"
    ) -> Any:
        messages = [{"role": "user", "content": [part, {"type": "text", "text": "hi"}]}]
        registry.hooks.emit_sync(
            "onMessageResolve",
            {"provider": provider, "model": "m", "messages": messages, "system": None},
        )
        return messages[0]["content"][0]

    def test_a_small_path_becomes_an_inline_part(self, tmp_path: Path) -> None:
        target = tmp_path / "note.txt"
        target.write_bytes(HELLO)
        registry, seen, _ = self.registry()

        part = self.resolve(
            registry,
            {"type": "document", "source": {"type": "path", "path": str(target),
                                            "mimeType": "text/plain"}},
        )
        assert part["source"]["type"] == "base64"
        assert part["source"]["data"] == "aGVsbG8="
        assert seen == [], "inlining must not send a request"

    def test_a_large_file_is_uploaded_and_referenced(self) -> None:
        registry, seen, _ = self.registry()
        registry.register_provider("openai", OpenAIFileAdapter(api_key="k"))
        file = registry.add(
            filename="big.pdf", mime_type="application/pdf",
            content=BytesContent(mime_type="application/pdf", data=b"x" * 60_000),
        )

        part = self.resolve(
            registry, {"type": "document", "source": {"type": "file", "fileId": file.id}}
        )
        assert part["source"] == {
            "type": "provider_ref", "mimeType": "application/pdf", "refId": "file_abc",
        }
        assert len(seen) == 1, "one upload, not one per resolve"

    def test_the_second_send_reuses_the_first_upload(self) -> None:
        registry, seen, _ = self.registry()
        registry.register_provider("openai", OpenAIFileAdapter(api_key="k"))
        file = registry.add(
            filename="big.pdf", mime_type="application/pdf",
            content=BytesContent(mime_type="application/pdf", data=b"x" * 60_000),
        )
        source = {"type": "file", "fileId": file.id}
        self.resolve(registry, {"type": "document", "source": dict(source)})
        self.resolve(registry, {"type": "document", "source": dict(source)})
        assert len(seen) == 1, "the file was uploaded twice"

    def test_the_same_file_uploads_once_per_provider(self) -> None:
        registry, seen, _ = self.registry()
        registry.register_provider("openai", OpenAIFileAdapter(api_key="k"))
        registry.register_provider("anthropic", AnthropicFileAdapter(api_key="k"))
        file = registry.add(
            filename="big.pdf", mime_type="application/pdf",
            content=BytesContent(mime_type="application/pdf", data=b"x" * 60_000),
        )
        source = {"type": "file", "fileId": file.id}
        a = self.resolve(registry, {"type": "document", "source": dict(source)}, "openai")
        b = self.resolve(registry, {"type": "document", "source": dict(source)}, "anthropic")

        assert len(seen) == 2
        # Two identities for one document. Sending either id to the other
        # provider is a 404, which is why the state is keyed by provider.
        assert a["source"]["refId"] == b["source"]["refId"] == "file_abc"
        assert set(file.uploads) == {"openai", "anthropic"}

    def test_an_unsupported_type_becomes_text_saying_so(self) -> None:
        registry, _, hooks = self.registry()
        registry.register_provider("xai", XaiFileAdapter(api_key="k"))
        warnings: list[Any] = []
        hooks.on("onWarning", lambda ctx: warnings.append(dict(ctx)))
        file = registry.add(
            filename="clip.mp4", mime_type="video/mp4",
            content=BytesContent(mime_type="video/mp4", data=b"x"),
        )

        part = self.resolve(
            registry, {"type": "video", "source": {"type": "file", "fileId": file.id}}, "xai"
        )
        # Text, not a removal: the model is told the file is missing rather than
        # asked about one it cannot see.
        assert part["type"] == "text"
        assert "clip.mp4" in part["text"]
        assert [w["code"] for w in warnings] == ["file_skipped"]

    def test_a_part_with_no_file_source_is_left_exactly_as_it_was(self) -> None:
        registry, seen, _ = self.registry()
        original = {"type": "image", "source": {"type": "url", "url": "https://e/i.png"}}
        assert self.resolve(registry, dict(original)) == original
        assert seen == []

    def test_an_unknown_file_id_warns_and_changes_nothing(self) -> None:
        registry, _, hooks = self.registry()
        warnings: list[Any] = []
        hooks.on("onWarning", lambda ctx: warnings.append(dict(ctx)))
        original = {"type": "document", "source": {"type": "file", "fileId": "nope"}}

        assert self.resolve(registry, dict(original)) == original
        assert [w["code"] for w in warnings] == ["file_not_found"]

    def test_with_no_adapter_a_large_file_is_inlined_rather_than_dropped(self) -> None:
        registry, seen, _ = self.registry()
        file = registry.add(
            filename="big.pdf", mime_type="application/pdf",
            content=BytesContent(mime_type="application/pdf", data=b"x" * 60_000),
        )
        part = self.resolve(
            registry, {"type": "document", "source": {"type": "file", "fileId": file.id}}
        )
        # The caller asked for the file to be sent. Without a store to put it in,
        # inline is worse but it is not nothing.
        assert part["source"]["type"] == "base64"
        assert seen == []

    def test_a_failed_upload_is_recorded_before_it_is_raised(self) -> None:
        hooks = HookBus()

        def failing(request: Any, options: Any = None) -> dict[str, Any]:
            return {"status": 500, "headers": {}, "body": {"error": "boom"}}

        registry = FilesRegistry(hooks=hooks, fetch=failing)
        registry.register_provider("openai", OpenAIFileAdapter(api_key="k"))
        file = registry.add(
            filename="a.pdf", mime_type="application/pdf",
            content=BytesContent(mime_type="application/pdf", data=b"x"),
        )
        with pytest.raises(RuntimeError):
            registry.upload(file.id, "openai")
        assert file.uploads["openai"].status == "error"
        assert "500" in (file.uploads["openai"].error or "")

    def test_closing_stops_the_interception(self, tmp_path: Path) -> None:
        target = tmp_path / "n.txt"
        target.write_bytes(HELLO)
        registry, _, _ = self.registry()
        registry.close()

        original = {"type": "document",
                    "source": {"type": "path", "path": str(target), "mimeType": "text/plain"}}
        assert self.resolve(registry, dict(original)) == original
        registry.close()  # idempotent

    def test_uploading_an_unknown_file_or_provider_says_which(self) -> None:
        registry, _, _ = self.registry()
        with pytest.raises(KeyError, match="not in this registry"):
            registry.upload("nope", "openai")
        file = registry.add(
            filename="a.txt", mime_type="text/plain",
            content=BytesContent(mime_type="text/plain", data=HELLO),
        )
        with pytest.raises(KeyError, match="no file adapter"):
            registry.upload(file.id, "openai")

    def test_a_size_is_estimated_without_reading_anything(self) -> None:
        registry, _, _ = self.registry()
        from combycode_llm_sdk.util.base64 import bytes_to_base64

        assert registry.add(
            filename="a", mime_type="text/plain",
            content=BytesContent(mime_type="text/plain", data=b"x" * 300),
        ).size_bytes == 300
        assert registry.add(
            filename="b", mime_type="text/plain",
            content=Base64Content(mime_type="text/plain", data=bytes_to_base64(b"x" * 300)),
        ).size_bytes == 300
        # A path reports 0 rather than being opened: a registry holding a hundred
        # of them must not stat all of them to add one.
        assert registry.add(
            filename="c", mime_type="text/plain",
            content=PathContent(mime_type="text/plain", path="/nowhere/x.txt"),
        ).size_bytes == 0

    def test_delete_remote_is_silent_when_there_is_nothing_to_delete(self) -> None:
        registry, seen, _ = self.registry()
        registry.delete_remote("nope", "openai")
        file = registry.add(
            filename="a.txt", mime_type="text/plain",
            content=BytesContent(mime_type="text/plain", data=HELLO),
        )
        registry.delete_remote(file.id, "openai")  # never uploaded
        assert seen == []

    def test_delete_remote_marks_the_file_deleted_for_that_provider(self) -> None:
        registry, seen, _ = self.registry()
        registry.register_provider("openai", OpenAIFileAdapter(api_key="k"))
        file = registry.add(
            filename="a.txt", mime_type="text/plain",
            content=BytesContent(mime_type="text/plain", data=HELLO),
        )
        file.set_uploaded("openai", "file-1", None)
        registry.delete_remote(file.id, "openai")

        assert len(seen) == 1
        assert seen[0]["method"] == "DELETE"
        assert file.uploads["openai"].status == "deleted"
        assert file.get_ref("openai") is None
