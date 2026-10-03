"""A credential that spans Workspaces has to say which one it is acting in.

Transposed from
`unified-library-ts/tests/unit/llm/anthropic-workspace.test.ts`.

Anthropic accounts the Workspace for spend, rate limits and retention, and
``anthropic-workspace-id`` is what selects it. A key scoped to one Workspace may
omit the header. A key that can act on SEVERAL and omits it does not fail -- it
charges the default Workspace, which is the worst failure mode available:
silent, and only visible on a bill.

So it is sent on every Anthropic request the library makes, not only
completions. Uploading a file or submitting a batch to the wrong Workspace is
the same mistake as billing a turn there, and a header that covered only
``/v1/messages`` would look like the feature while leaving the hole.

Two ways in, because the two surfaces are shaped differently:

* completions: ``providerOptions.workspaceId`` per request, and an
  ``AnthropicAdapter`` built with one as the client-wide default. The request
  wins.
* files, batches, counting, model listing: a ``workspace_id`` on their own
  config, since none of them is a completion and none has providerOptions.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")


from combycode_llm_sdk.batch.adapters import AnthropicBatchAdapter
from combycode_llm_sdk.files.attachment import BytesContent, FileAttachment
from combycode_llm_sdk.files.providers import AnthropicFileAdapter
from combycode_llm_sdk.llm.providers.anthropic.messages import AnthropicAdapter

WS = "wrkspc_011CZkZaBF1tNoB5wlCeusgy"
HEADER = "anthropic-workspace-id"


def _req(provider_options: dict[str, Any] | None = None) -> dict[str, Any]:
    """The minimum a spec build needs from a normalized request."""
    req: dict[str, Any] = {
        "model": "claude-haiku-4.5",
        "messages": [{"role": "user", "content": "hi"}],
    }
    if provider_options is not None:
        req["providerOptions"] = provider_options
    return req


def headers_of(
    adapter: AnthropicAdapter, provider_options: dict[str, Any] | None = None
) -> dict[str, str]:
    """How the client composes them: client-wide first, per-request over the top."""
    built = adapter.build_request(_req(provider_options))
    return {**adapter.auth_headers(), **(built.headers or {})}


class TestCompletions:
    def test_it_sends_nothing_when_no_workspace_was_named(self) -> None:
        # The ordinary single-Workspace credential. Sending the header empty is
        # a different request from not sending it, and only the second says
        # "this client was given no Workspace".
        assert HEADER not in headers_of(AnthropicAdapter({"apiKey": "k"}))

    def test_it_sends_the_client_wide_default_on_every_request(self) -> None:
        adapter = AnthropicAdapter({"apiKey": "k", "workspaceId": WS})
        assert headers_of(adapter)[HEADER] == WS

    def test_it_sends_a_per_request_workspace_with_no_client_default(self) -> None:
        adapter = AnthropicAdapter({"apiKey": "k"})
        assert headers_of(adapter, {"workspaceId": WS})[HEADER] == WS

    def test_the_request_overrides_the_client_default(self) -> None:
        adapter = AnthropicAdapter({"apiKey": "k", "workspaceId": WS})
        assert headers_of(adapter, {"workspaceId": "wrkspc_other"})[HEADER] == "wrkspc_other"

    def test_the_client_default_survives_an_unrelated_provider_option(self) -> None:
        adapter = AnthropicAdapter({"apiKey": "k", "workspaceId": WS})
        assert headers_of(adapter, {"userProfileId": "u_1"})[HEADER] == WS

    def test_an_empty_string_sends_no_header(self) -> None:
        assert HEADER not in headers_of(AnthropicAdapter({"apiKey": "k"}), {"workspaceId": ""})
        assert HEADER not in headers_of(AnthropicAdapter({"apiKey": "k", "workspaceId": ""}))

    def test_it_does_not_disturb_the_other_anthropic_headers(self) -> None:
        h = headers_of(AnthropicAdapter({"apiKey": "k", "workspaceId": WS}))
        assert h["x-api-key"] == "k"
        assert isinstance(h["anthropic-version"], str)


class TestTheSurfacesThatAreNotCompletions:
    def test_a_file_upload_carries_it(self) -> None:
        adapter = AnthropicFileAdapter("k", workspace_id=WS)
        built = adapter.build_upload_request(
            FileAttachment(
                filename="x.bin",
                mime_type="application/octet-stream",
                size_bytes=3,
                content=BytesContent(
                    mime_type="application/octet-stream", data=b"abc"
                ),
            ),
            b"\x01\x02\x03",
        )
        assert built["headers"][HEADER] == WS

    def test_so_does_listing_and_deleting(self) -> None:
        adapter = AnthropicFileAdapter("k", workspace_id=WS)
        assert adapter.build_list_request()["headers"][HEADER] == WS
        assert adapter.build_delete_request("file_1")["headers"][HEADER] == WS

    def test_none_of_them_carries_it_when_none_was_configured(self) -> None:
        adapter = AnthropicFileAdapter("k")
        assert HEADER not in adapter.build_list_request()["headers"]

    def test_a_batch_carries_it_on_the_submission(self) -> None:
        adapter = AnthropicBatchAdapter("k", workspace_id=WS)
        assert adapter.submit_request([])["headers"][HEADER] == WS

    def test_a_batch_omits_it_when_none_was_configured(self) -> None:
        adapter = AnthropicBatchAdapter("k")
        assert HEADER not in adapter.submit_request([])["headers"]


class TestFetchingBackWhatATurnProduced:
    """`files.content` stands alone rather than extending `files.base`.

    So it did NOT inherit the header the other file calls got: the download
    went to the default Workspace while every other call went to the right one.
    """

    def retrieve(self, workspace_id: str | None = None) -> list[dict[str, str]]:
        from combycode_llm_sdk.llm.client import LLMClient

        seen: list[dict[str, str]] = []

        def fetch(req: dict[str, Any]) -> dict[str, Any]:
            seen.append(dict(req.get("headers") or {}))
            return {
                "status": 200,
                "headers": {"content-type": "application/octet-stream"},
                "body": b"\x01\x02\x03",
            }

        config: dict[str, Any] = {"apiKey": "k"}
        if workspace_id is not None:
            config["workspaceId"] = workspace_id
        client = LLMClient(
            {
                "provider": "anthropic",
                "model": "claude-haiku-4.5",
                "apiKey": "k",
                "fetch": fetch,
                "adapter": AnthropicAdapter(config),
            }
        )
        client.retrieve_file({"id": "file_1", "source": "code_execution"})
        return seen

    def test_it_carries_the_client_workspace(self) -> None:
        seen = self.retrieve(WS)
        # Asserted unconditionally: a guard would pass cheerfully when the
        # request was never sent at all.
        assert len(seen) == 1
        assert seen[0][HEADER] == WS

    def test_it_sends_none_when_the_client_has_none(self) -> None:
        seen = self.retrieve()
        assert len(seen) == 1
        assert HEADER not in seen[0]
