"""Two decisions: refusing what cannot be called, and asking a provider to
delete an uploaded file for us.

The refusal reads the naming, but what it tracks is the difference between "we
cannot reach it" and "it is gone". `imagen-4` is our slug, so the catalog's
answer applies; `imagen-4.0-generate-001` is Google's own id, and typing it is a
deliberate choice -- most likely a deployment where the endpoint answers. We
measured it gone for US; we cannot measure it gone for everyone.

The lifetime shapes were measured on 2026-09-29, and two of three would have
been wrong if guessed: OpenAI needs BRACKET fields (a JSON string is refused
400) and xAI needs its field BEFORE the file part (after it, the server replies
"expires_after must appear before the file field"). Google cannot take one at
all -- its expiration_time is "Output only".
"""

from __future__ import annotations

import base64
import re
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from combycode_llm_sdk.bus.hook_bus import HookBus
from combycode_llm_sdk.catalog.catalog import ModelCatalog, resolve_catalog
from combycode_llm_sdk.files.attachment import BytesContent, FileAttachment
from combycode_llm_sdk.files.provider_adapter import FileUploadOptions, FileUploadResult
from combycode_llm_sdk.files.providers import (
    AnthropicFileAdapter,
    GoogleFileAdapter,
    OpenAIFileAdapter,
    XaiFileAdapter,
)
from combycode_llm_sdk.files.registry import FilesRegistry
from combycode_llm_sdk.media import MediaOutput, MemoryMediaStore


def _file() -> FileAttachment:
    return FileAttachment.from_bytes(b"x", filename="a.txt", mime_type="text/plain")


def _collect(seen: list[str]) -> Callable[..., None]:
    def warn(message: str, _details: Mapping[str, Any] | None = None) -> None:
        seen.append(message)

    return warn


def _field_names(req: Any) -> list[str]:
    """The part names actually on the wire, in the order they are sent.

    The builder encodes the multipart body itself, so this reads the bytes
    rather than a list of parts -- which is the stricter check anyway: xAI
    rejects `expires_after` that arrives after the file part, and only the
    encoded order can show that.
    """
    body = req["body"]
    assert req.get("rawBody") is True
    text = bytes(body).decode("latin-1")
    return re.findall(r'(?<!file)name="([^"]+)"', text)


class TestRefuseCall:
    catalog = ModelCatalog()

    @classmethod
    def setup_class(cls) -> None:
        cls.catalog.load_provider_defaults()

    @pytest.mark.parametrize("model", ["imagen-4", "imagen-4-fast", "imagen-4-ultra"])
    def test_refuses_our_slug(self, model: str) -> None:
        refusal = self.catalog.refuse_call("google", model)
        assert refusal is not None
        assert "Enterprise" in refusal

    def test_names_the_provider_id_as_the_way_through(self) -> None:
        refusal = self.catalog.refuse_call("google", "imagen-4")
        assert refusal is not None and "imagen-4.0-generate-001" in refusal

    @pytest.mark.parametrize(
        "model",
        [
            "imagen-4.0-generate-001",
            "imagen-4.0-fast-generate-001",
            "imagen-4.0-ultra-generate-001",
        ],
    )
    def test_lets_the_provider_id_through(self, model: str) -> None:
        assert self.catalog.refuse_call("google", model) is None

    def test_refuses_sora_with_no_way_through(self) -> None:
        # Slug and provider id are the same string, and the endpoint is gone for
        # everyone -- there is nothing to force.
        refusal = self.catalog.refuse_call("openai", "sora-2")
        assert refusal is not None
        assert "2026-09-24" in refusal
        assert "name the provider" not in refusal

    def test_says_nothing_about_a_model_that_works(self) -> None:
        assert self.catalog.refuse_call("google", "gemini-3.1-flash-image") is None

    def test_the_measurement_survives_the_override(self) -> None:
        # Knowledge and policy are separate: the provider id overrides the
        # decision, never the fact.
        assert self.catalog.unavailable_reason("google", "imagen-4.0-generate-001") is not None
        assert self.catalog.refuse_call("google", "imagen-4.0-generate-001") is None


class TestUploadLifetime:
    def test_anthropic_sends_a_plain_scalar(self) -> None:
        a = AnthropicFileAdapter("k")
        with_it = _field_names(a.build_upload_request(_file(), b"x", FileUploadOptions(3600)))
        without = _field_names(a.build_upload_request(_file(), b"x"))
        assert "expires_in_seconds" in with_it
        assert "expires_in_seconds" not in without

    def test_openai_sends_bracket_fields(self) -> None:
        o = OpenAIFileAdapter("k")
        names = _field_names(o.build_upload_request(_file(), b"x", FileUploadOptions(3600)))
        assert "expires_after[anchor]" in names
        assert "expires_after[seconds]" in names

    def test_nobody_sends_a_lifetime_that_was_not_asked_for(self) -> None:
        # The anchor field is guarded on `defined`, and a None reaching the
        # payload would satisfy that guard: it is a present key, not an absent
        # one. Without this the default upload carries expires_after[anchor].
        for adapter in (AnthropicFileAdapter("k"), OpenAIFileAdapter("k"), XaiFileAdapter("k")):
            names = _field_names(adapter.build_upload_request(_file(), b"x"))
            assert not [n for n in names if n.startswith("expires")], adapter.name

    def test_xai_sends_its_field_before_the_file(self) -> None:
        x = XaiFileAdapter("k")
        names = _field_names(x.build_upload_request(_file(), b"x", FileUploadOptions(3600)))
        assert names.index("expires_after") < names.index("file")

    def test_google_warns_rather_than_dropping_it(self) -> None:
        seen: list[str] = []
        g = GoogleFileAdapter("k")

        def fetch(_req: Any) -> dict[str, Any]:
            return {"status": 500, "headers": {}, "body": {}}

        with pytest.raises(RuntimeError):
            g.upload(
                _file(),
                fetch,
                FileUploadOptions(3600, warn=_collect(seen)),
            )
        assert len(seen) == 1
        assert "3600" in seen[0]
        assert "expiration_time" in seen[0]

    def test_google_says_nothing_when_no_lifetime_is_asked_for(self) -> None:
        seen: list[str] = []
        g = GoogleFileAdapter("k")

        def fetch(_req: Any) -> dict[str, Any]:
            return {"status": 500, "headers": {}, "body": {}}

        with pytest.raises(RuntimeError):
            g.upload(_file(), fetch, FileUploadOptions(warn=_collect(seen)))
        assert seen == []


class TestTheRegistryThreadsIt:
    """The seam between the configured option and the wire.

    This is the half that went missing on the way across: the adapters took the
    option and nothing handed it to them, so it was unreachable from the public
    API while every adapter test still passed.
    """

    @staticmethod
    def _spy(seen: dict[str, Any]) -> Any:
        class Spy:
            name = "spy"
            expires_after_ms = None
            max_file_size = 100_000_000
            supported_types = None

            def upload(
                self, file: Any, fetch: Any, opts: FileUploadOptions | None = None
            ) -> FileUploadResult:
                seen["opts"] = opts
                return FileUploadResult(remote_id="remote_1")

            def delete(self, remote_id: str, fetch: Any) -> None: ...

            def get_info(self, remote_id: str, fetch: Any) -> None:
                return None

            def list(self, fetch: Any) -> list[Any]:
                return []

        return Spy()

    @staticmethod
    def _noop_fetch(request: Any, options: Any = None) -> dict[str, Any]:
        return {"status": 200, "headers": {}, "body": {}}

    def _upload(self, seen: dict[str, Any], **config: Any) -> tuple[str, list[Any]]:
        hooks = HookBus()
        warnings: list[Any] = []
        hooks.on("onWarning", lambda w: warnings.append(dict(w)))
        registry = FilesRegistry(hooks=hooks, fetch=self._noop_fetch, **config)
        registry.register_provider("spy", self._spy(seen))
        file = registry.add(
            filename="a.txt",
            mime_type="text/plain",
            content=BytesContent(mime_type="text/plain", data=b"hi"),
        )
        registry.upload(file.id, "spy")
        return file.id, warnings

    def test_passes_what_the_caller_configured(self) -> None:
        seen: dict[str, Any] = {}
        self._upload(seen, upload_lifetime_seconds=3600)
        assert seen["opts"].lifetime_seconds == 3600

    def test_leaves_it_unset_when_nothing_was_configured(self) -> None:
        seen: dict[str, Any] = {}
        self._upload(seen)
        assert seen["opts"].lifetime_seconds is None

    def test_an_adapter_warn_reaches_on_warning(self) -> None:
        seen: dict[str, Any] = {}
        file_id, warnings = self._upload(seen, upload_lifetime_seconds=60)
        seen["opts"].warn("cannot honour it", {"requestedLifetimeSeconds": 60})
        adjusted = [w for w in warnings if w.get("code") == "request_adjusted"]
        assert len(adjusted) == 1
        assert adjusted[0]["message"] == "cannot honour it"
        assert adjusted[0]["details"]["fileId"] == file_id
        assert adjusted[0]["details"]["provider"] == "spy"


class TestMediaAsksBeforeItSpends:
    """The seam: that `generate_image` actually consults the refusal.

    The catalog tests above would all stay green if the call were removed from
    the media surface, which is the thing a caller actually reaches.
    """

    @staticmethod
    def _rig(model: str) -> tuple[Any, list[dict[str, Any]]]:
        seen: list[dict[str, Any]] = []

        def fetch(request: dict[str, Any], _options: Any = None) -> dict[str, Any]:
            seen.append(request)
            png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"x" * 40).decode()
            return {
                "status": 200,
                "headers": {},
                "body": {
                    "candidates": [
                        {"content": {"parts": [{"inlineData": {"mimeType": "image/png",
                                                               "data": png}}]}}
                    ]
                },
            }

        media = MediaOutput(
            model=model,
            api_key="k",
            store=MemoryMediaStore(),
            fetch=fetch,
            catalog=resolve_catalog(None),
        )
        return media, seen

    def test_it_refuses_our_slug_without_sending_anything(self) -> None:
        media, seen = self._rig("google/imagen-4")
        with pytest.raises(Exception, match="Enterprise"):
            media.generate_image(prompt="x")
        assert seen == []

    def test_it_goes_ahead_on_the_provider_own_id(self) -> None:
        media, seen = self._rig("google/imagen-4.0-generate-001")
        media.generate_image(prompt="x")
        assert len(seen) == 1

    def test_it_leaves_a_model_with_nothing_against_it_alone(self) -> None:
        media, seen = self._rig("google/gemini-3.1-flash-image")
        media.generate_image(prompt="x")
        assert len(seen) == 1
