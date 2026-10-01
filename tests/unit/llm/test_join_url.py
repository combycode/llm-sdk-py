"""Joining a base URL to a path without destroying the base's query string.

Transposed from `unified-library-ts/tests/unit/llm/join-url.test.ts`.

`base + path` is correct for every base this library SHIPS, because each is only a
host. It is wrong for the one shape callers configure by hand -- an Azure-style
endpoint carrying `?api-version=...`:

    "https://x.openai.azure.com/openai?api-version=2026-05-01" + "/v1/responses"
    -> ".../openai?api-version=2026-05-01/v1/responses"

The path has become part of the `api-version` VALUE. The request reaches the base
path with a nonsense version, and the error that comes back is about the version
-- so the only clue points at the wrong thing.

The realtime half of the same fault was worse than a bad join: the OpenAI realtime
adapter ACCEPTED a `base_url`, handed it to the spec, and the spec never read it. A
caller who configured one got silence and the default host. The Google adapter has
always honoured its own.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from combycode_llm_sdk.llm.join_url import join_url, ws_url
from combycode_llm_sdk.realtime.providers import OpenAIRealtimeAdapter
from combycode_llm_sdk.realtime.types import SessionConfig

AZURE = "https://x.openai.azure.com/openai?api-version=2026-05-01"


class TestJoinUrl:
    def test_the_path_lands_before_the_query(self) -> None:
        assert (
            join_url(AZURE, "/v1/responses")
            == "https://x.openai.azure.com/openai/v1/responses?api-version=2026-05-01"
        )

    def test_it_behaves_like_concatenation_for_a_plain_host(self) -> None:
        # The regression that matters: every shipped base is this shape, so this
        # path carries all existing traffic.
        assert join_url("https://api.openai.com", "/v1/responses") == (
            "https://api.openai.com/v1/responses"
        )
        assert join_url("https://api.anthropic.com", "/v1/messages") == (
            "https://api.anthropic.com/v1/messages"
        )

    def test_a_doubled_slash_collapses(self) -> None:
        # The two halves come from different places -- our adapter's constant and
        # the caller's config -- so neither can know what the other ended with, and
        # `//` is a different path to a strict router.
        assert join_url("https://host/", "/v1/x") == "https://host/v1/x"
        assert join_url("https://host", "/v1/x") == "https://host/v1/x"
        assert join_url("https://host/", "v1/x") == "https://host/v1/x"

    def test_a_multi_parameter_query_stays_whole(self) -> None:
        assert join_url("https://h/p?a=1&b=2", "/x") == "https://h/p/x?a=1&b=2"

    def test_an_empty_path_returns_base_plus_query(self) -> None:
        assert join_url("https://h/p?a=1", "") == "https://h/p?a=1"

    def test_a_fragment_is_dropped_rather_than_carried(self) -> None:
        # `#x` is never sent to a server, and keeping it mid-URL would move it
        # somewhere it means even less.
        assert join_url("https://h/p?a=1#frag", "/x") == "https://h/p/x?a=1"


class TestWsUrl:
    def test_it_switches_the_scheme_and_merges_the_parameters(self) -> None:
        # The three faults at once: scheme, path-before-query, and `model` MERGING
        # rather than starting a second `?`.
        assert ws_url(AZURE, "/realtime", {"model": "gpt-realtime"}) == (
            "wss://x.openai.azure.com/openai/realtime?api-version=2026-05-01&model=gpt-realtime"
        )

    def test_it_produces_exactly_the_ga_url_from_the_default_host(self) -> None:
        # What the frozen service corpus pins. If this changed, every realtime
        # session would move.
        assert ws_url("https://api.openai.com", "/v1/realtime", {"model": "gpt-realtime"}) == (
            "wss://api.openai.com/v1/realtime?model=gpt-realtime"
        )

    def test_each_scheme_maps_to_its_websocket_twin(self) -> None:
        # `https` -> `wss` and `http` -> `ws`: a plain `ws` from an `https` base
        # would be an unencrypted socket to an encrypted endpoint.
        assert ws_url("https://h", "/x") == "wss://h/x"
        assert ws_url("http://h", "/x") == "ws://h/x"
        assert ws_url("wss://h", "/x", {"a": "1"}) == "wss://h/x?a=1"
        assert ws_url("ws://h", "/x") == "ws://h/x"

    def test_a_none_parameter_is_dropped_not_sent_empty(self) -> None:
        # A provider that validates its query rejects `model=` differently from an
        # absent `model`, and absent is what "not specified" means.
        assert ws_url("https://h", "/x", {"model": None}) == "wss://h/x"

    def test_it_overrides_a_parameter_the_base_already_carried(self) -> None:
        assert ws_url("https://h?model=old", "/x", {"model": "new"}) == "wss://h/x?model=new"


class TestTheOpenAIRealtimeAdapterHonoursItsBaseUrl:
    @staticmethod
    def connect(base_url: str | None = None) -> str:
        adapter = OpenAIRealtimeAdapter(api_key="k", base_url=base_url)
        return adapter.build_connect_request(SessionConfig(model="gpt-realtime")).url

    def test_it_still_produces_the_ga_url_by_default(self) -> None:
        assert self.connect() == "wss://api.openai.com/v1/realtime?model=gpt-realtime"

    def test_it_uses_a_configured_host_instead_of_ignoring_it(self) -> None:
        # Before this, `base_url` was accepted, passed to the spec, and never read
        # -- the caller got no error and the default host.
        assert self.connect("https://gateway.internal") == (
            "wss://gateway.internal/v1/realtime?model=gpt-realtime"
        )

    def test_it_survives_a_host_that_carries_its_own_query(self) -> None:
        assert self.connect(AZURE) == (
            "wss://x.openai.azure.com/openai/v1/realtime?api-version=2026-05-01&model=gpt-realtime"
        )
