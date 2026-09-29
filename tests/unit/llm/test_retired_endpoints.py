"""Two endpoints that are gone, settled by a live check rather than by types.

    models/imagen-4.0-generate-001:predict -> 404, "not supported for predict"
    /v1/videos (GET and POST)              -> 404

The Imagen one mattered most: it was the DEFAULT image model, so the default
Google image path 404'd. The Sora one is quieter -- `sora-2` and `sora-2-pro`
are STILL listed by /v1/models, so a catalog built from ListModels keeps
reporting a model no endpoint serves.
"""

from __future__ import annotations

from typing import Any

import pytest

from combycode_llm_sdk.media.google import DEFAULT_IMAGE_MODEL, GoogleMediaAdapter
from combycode_llm_sdk.media.openai import OpenAIMediaAdapter
from combycode_llm_sdk.network.errors import UnsupportedError

KEY = "test-key"  # the media adapters take the key positionally, not a config dict


class TestGoogleImageDefault:
    def test_the_default_is_a_model_that_answers(self) -> None:
        assert DEFAULT_IMAGE_MODEL == "gemini-3.1-flash-image"

    def test_it_builds_through_generate_content_not_predict(self) -> None:
        req = GoogleMediaAdapter(KEY).build_generate_image_request("a cube", DEFAULT_IMAGE_MODEL)
        url = req["url"] if isinstance(req, dict) else req.url
        assert "gemini-3.1-flash-image" in url
        assert "generateContent" in url
        assert "predict" not in url

    def test_a_named_imagen_model_still_builds_the_predict_envelope(self) -> None:
        # Refusing it was tried in the TypeScript tree and reverted: it breaks the
        # frozen media corpus and the spec/adapter parity check, and an Enterprise
        # deployment can still reach that endpoint.
        req = GoogleMediaAdapter(KEY).build_generate_image_request("a cube", "imagen-4.0-generate-001")
        url = req["url"] if isinstance(req, dict) else req.url
        assert "predict" in url


class TestSoraShutdown:
    @staticmethod
    def _fetch(status: int, body: dict[str, Any]) -> Any:
        def fetch(_req: Any) -> dict[str, Any]:
            return {"status": status, "headers": {}, "body": body}

        return fetch

    def test_a_404_reports_the_shutdown(self) -> None:
        adapter = OpenAIMediaAdapter(KEY)
        with pytest.raises(UnsupportedError) as excinfo:
            adapter.submit_video("a cube", "sora-2", self._fetch(404, {}))
        assert "2026-09-24" in str(excinfo.value)
        assert "sora-2" in str(excinfo.value)

    def test_it_does_not_return_an_empty_id(self) -> None:
        # The old behaviour: a 404 produced "", which reads as a submitted job
        # with no id and fails later somewhere that cannot explain itself.
        adapter = OpenAIMediaAdapter(KEY)
        with pytest.raises(UnsupportedError):
            adapter.submit_video("x", "sora-2", self._fetch(404, {}))

    def test_a_working_endpoint_still_submits(self) -> None:
        adapter = OpenAIMediaAdapter(KEY)
        assert adapter.submit_video("x", "sora-2", self._fetch(200, {"id": "vid_1"})) == "vid_1"
