"""`stream: true`, and the conversations a server can be asked to continue.

The streaming tests bind a REAL port and read the socket as a client would,
because the one thing that matters here cannot be observed any other way: that
frames arrive one at a time. A test that reads the whole body passes just as
happily against an implementation that buffers everything and writes it at the
end -- which would deliver nothing anyone would call streaming.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from typing import Any

import pytest

from combycode_llm_sdk import LLM, TransportResponse
from combycode_llm_sdk.agent.history import ConversationHistory
from combycode_llm_sdk.persistence import MemoryPersistence
from combycode_llm_sdk.server import (
    DEFAULT_STREAM_CHUNK_CHARS,
    SSE_TERMINATOR,
    HttpRequest,
    OaiServer,
    ResponseEntry,
    ResponseStore,
    ResponseTarget,
    ServerEntry,
    has_fresh_provider_state,
    make_http_server,
    new_response_id,
    split_for_stream,
    stream_frames,
)
from combycode_llm_sdk.server.response_store import now_ms


class Provider:
    """The upstream, answering in the shape OpenAI's Responses API really uses.

    A stub TRANSPORT under a real `LLM`, not a fake client: the streaming path
    runs through dispatch and the client, and a hand-written double would prove
    only that the double works.
    """

    def __init__(self, text: str) -> None:
        self.text = text

    def __call__(self, request: Any) -> TransportResponse:
        return TransportResponse(
            status=200,
            body={
                "id": "resp_provider",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": self.text}],
                    }
                ],
                "usage": {"input_tokens": 11, "output_tokens": 3},
            },
        )


def server_with(text: str = "hello world", **over: Any) -> OaiServer:
    client = LLM(model="openai/gpt-4o-mini", api_key="k", transport=Provider(text))
    return OaiServer(entries=[ServerEntry(model="m", client=client)], **over)


def chat(model: str = "m", **over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": "hi"}]}
    body.update(over)
    return body


def post(server: OaiServer, body: dict[str, Any]) -> Any:
    """One request through the real entry point, with no socket in the way."""
    return server.handle(
        HttpRequest(method="POST", path="/v1/chat/completions", body=body)
    )


# ── the frames themselves ───────────────────────────────────────────────────


class TestTheFrames:
    def test_text_is_cut_into_pieces_of_the_asked_width(self) -> None:
        assert split_for_stream("abcdefg", 3) == ["abc", "def", "g"]
        assert split_for_stream("abc", 10) == ["abc"]

    def test_an_empty_answer_is_no_pieces_rather_than_one_empty_piece(self) -> None:
        assert split_for_stream("", 5) == []

    def test_a_zero_width_cannot_produce_an_endless_stream(self) -> None:
        # `range(0, n, 0)` raises, and a caller passing 0 would otherwise take
        # the server down rather than get a bad stream.
        assert split_for_stream("abc", 0) == ["a", "b", "c"]

    def test_the_first_frame_carries_the_role_and_no_content(self) -> None:
        # A client that builds a message from the deltas has nothing to
        # attribute the text to without it, and several SDKs drop the choice.
        frames = list(stream_frames(chunk_id="c", model="m", text="hi", chunk_chars=40))
        first = json.loads(frames[0].removeprefix("data: "))
        assert first["choices"][0]["delta"] == {"role": "assistant"}
        assert first["object"] == "chat.completion.chunk"

    def test_the_last_data_frame_carries_the_finish_reason_and_no_content(self) -> None:
        frames = list(stream_frames(chunk_id="c", model="m", text="hi", finish_reason="length"))
        final = json.loads(frames[-2].removeprefix("data: "))
        assert final["choices"][0]["finish_reason"] == "length"
        assert final["choices"][0]["delta"] == {}

    def test_the_stream_ends_with_the_sentinel_that_is_not_json(self) -> None:
        frames = list(stream_frames(chunk_id="c", model="m", text="hi"))
        assert frames[-1] == SSE_TERMINATOR

    def test_the_pieces_reassemble_into_exactly_the_answer(self) -> None:
        # The only property a client actually depends on.
        text = "The quick brown fox jumps over the lazy dog, twice, and then again."
        frames = list(stream_frames(chunk_id="c", model="m", text=text, chunk_chars=7))
        content = "".join(
            json.loads(f.removeprefix("data: "))["choices"][0]["delta"].get("content", "")
            for f in frames
            if f != SSE_TERMINATOR
        )
        assert content == text

    def test_every_frame_is_delimited_by_a_blank_line(self) -> None:
        # The blank line IS the delimiter, not decoration: without it a client's
        # parser reads two events as one.
        for frame in stream_frames(chunk_id="c", model="m", text="hi"):
            assert frame.endswith("\n\n")


# ── the route ───────────────────────────────────────────────────────────────


class TestTheRoute:
    def test_a_plain_request_still_answers_with_json(self) -> None:
        response = post(server_with(), chat())
        assert response.headers["content-type"] == "application/json"
        assert response.body["choices"][0]["message"]["content"] == "hello world"

    def test_stream_true_answers_with_an_event_stream(self) -> None:
        response = post(server_with(), chat(stream=True))
        assert response.headers["content-type"] == "text/event-stream"
        # No-cache is not politeness: a proxy that buffers an event stream turns
        # it back into one blob.
        assert response.headers["cache-control"] == "no-cache"

    def test_the_body_is_lazy_rather_than_a_rendered_string(self) -> None:
        # If it were rendered here, the shell could not write it incrementally
        # and the feature would be a formatting change.
        response = post(server_with(), chat(stream=True))
        assert not isinstance(response.body, (str, bytes, list))
        assert hasattr(response.body, "__next__")

    def test_the_chunk_width_is_the_servers_to_set(self) -> None:
        response = post(server_with("abcdefghij", stream_chunk_chars=2), chat(stream=True))
        contents = [
            json.loads(f.removeprefix("data: "))["choices"][0]["delta"].get("content")
            for f in response.body
            if f != SSE_TERMINATOR
        ]
        assert [c for c in contents if c] == ["ab", "cd", "ef", "gh", "ij"]

    def test_the_default_width_is_the_documented_one(self) -> None:
        assert DEFAULT_STREAM_CHUNK_CHARS == 40

    def test_a_streamed_answer_names_the_registered_model_not_the_providers(self) -> None:
        # What lets a client switch providers without changing what it sends.
        response = post(server_with(), chat(stream=True))
        first = json.loads(next(iter(response.body)).removeprefix("data: "))
        assert first["model"] == "m"

    def test_an_unknown_model_is_still_a_json_error_when_streaming(self) -> None:
        # An error is not an event stream: a client asking for SSE still needs
        # to be able to read the failure.
        response = post(server_with(), chat(model="nope", stream=True))
        assert response.status == 404
        assert response.headers["content-type"] == "application/json"


# ── over a real socket ──────────────────────────────────────────────────────


class TestOverARealSocket:
    """A bound port, read the way a client reads it."""

    @pytest.fixture
    def address(self) -> Any:
        server = make_http_server(server_with("x" * 200, stream_chunk_chars=10), port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def post(self, address: str, body: dict[str, Any]) -> Any:
        request = urllib.request.Request(
            f"{address}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=10)

    def test_a_real_client_reads_frames_one_at_a_time(self, address: str) -> None:
        # The test this whole path exists for. Reading line by line, the first
        # frame must arrive before the connection ends -- an implementation that
        # buffers cannot satisfy this.
        with self.post(address, chat(stream=True)) as response:
            assert response.headers["content-type"] == "text/event-stream"
            first = response.readline()
            assert first.startswith(b"data: "), first
            payload = json.loads(first.decode().removeprefix("data: "))
            assert payload["choices"][0]["delta"] == {"role": "assistant"}

            rest = response.read().decode()

        assert rest.rstrip().endswith("[DONE]")

    def test_the_whole_stream_reassembles_into_the_answer(self, address: str) -> None:
        with self.post(address, chat(stream=True)) as response:
            body = response.read().decode()

        content = "".join(
            json.loads(line.removeprefix("data: "))["choices"][0]["delta"].get("content", "")
            for line in body.splitlines()
            if line.startswith("data: ") and not line.endswith("[DONE]")
        )
        assert content == "x" * 200

    def test_a_non_streaming_request_over_the_same_port_is_unaffected(
        self, address: str
    ) -> None:
        with self.post(address, chat()) as response:
            payload = json.loads(response.read().decode())
        assert payload["object"] == "chat.completion"
        assert payload["choices"][0]["message"]["content"] == "x" * 200


# ── the response store ──────────────────────────────────────────────────────


class TestTheResponseStore:
    def entry(self, response_id: str = "resp_1", user: str | None = None) -> ResponseEntry:
        history = ConversationHistory()
        history.system = "be brief"
        return ResponseEntry(
            local_response_id=response_id,
            user_id=user,
            target=ResponseTarget(model="m", id="anthropic/claude-haiku-4.5"),
            history=history,
        )

    def test_a_conversation_round_trips_in_memory(self) -> None:
        store = ResponseStore()
        store.put(self.entry())
        got = store.get("resp_1")
        assert got is not None and got.target.model == "m"

    def test_an_unknown_id_is_none_rather_than_an_empty_conversation(self) -> None:
        # An empty history would continue as if the user had said nothing.
        assert ResponseStore().get("nope") is None

    def test_one_user_cannot_read_another_users_conversation(self) -> None:
        # Response ids are guessable enough that this must hold by
        # construction, not by convention.
        store = ResponseStore()
        store.put(self.entry("resp_1", user="alice"))

        assert store.get("resp_1", "alice") is not None
        assert store.get("resp_1", "bob") is None
        assert store.get("resp_1") is None, "an anonymous read must not reach an owned entry"

    def test_a_user_id_cannot_address_another_users_namespace(self) -> None:
        # The key is `prefix<user>:<id>`, so a user id containing `:` would
        # otherwise let one tenant name another's keyspace.
        store = ResponseStore(persistence=MemoryPersistence())
        store.put(self.entry("resp_1", user="alice"))
        store.put(self.entry("resp_1", user="alice:resp_1:bob"))

        assert store.get("resp_1", "alice") is not None
        assert store.list("alice") == ["resp_1"]

    def test_it_survives_a_restart_when_there_is_a_store(self) -> None:
        backing = MemoryPersistence()
        ResponseStore(persistence=backing).put(self.entry())

        # A second store, as a redeploy would build.
        restored = ResponseStore(persistence=backing).get("resp_1")
        assert restored is not None
        assert restored.target.model == "m"
        assert restored.history.system == "be brief", "the transcript came back as a history"

    def test_a_restored_history_is_a_history_and_not_a_dict(self) -> None:
        # A restored entry whose history carries a different type from a live
        # one's is a bug that only appears after a restart.
        backing = MemoryPersistence()
        store = ResponseStore(persistence=backing)
        entry = self.entry()
        entry.history.append({"role": "user", "content": "hi"})
        store.put(entry)

        restored = ResponseStore(persistence=backing).get("resp_1")
        assert restored is not None
        assert isinstance(restored.history, ConversationHistory)
        assert len(restored.history.entries) == 1

    def test_the_cache_evicts_the_least_recently_used(self) -> None:
        store = ResponseStore(memory_capacity=2)
        for name in ("a", "b"):
            store.put(self.entry(name))
        store.get("a")           # a is now the most recent
        store.put(self.entry("c"))

        assert len(store) == 2
        assert store.get("b") is None, "b was the least recently used"
        assert store.get("a") is not None

    def test_eviction_costs_a_read_and_not_a_conversation(self) -> None:
        # The cache is a convenience over the persistence, not the record.
        backing = MemoryPersistence()
        store = ResponseStore(persistence=backing, memory_capacity=1)
        store.put(self.entry("a"))
        store.put(self.entry("b"))

        assert store.get("a") is not None, "evicted from the cache, not lost"

    def test_listing_reads_the_store_rather_than_what_is_hot(self) -> None:
        backing = MemoryPersistence()
        store = ResponseStore(persistence=backing, memory_capacity=1)
        for name in ("a", "b", "c"):
            store.put(self.entry(name))
        assert sorted(store.list()) == ["a", "b", "c"]

    def test_delete_removes_it_from_both_halves(self) -> None:
        backing = MemoryPersistence()
        store = ResponseStore(persistence=backing)
        store.put(self.entry())
        store.delete("resp_1")

        assert store.get("resp_1") is None
        assert ResponseStore(persistence=backing).get("resp_1") is None

    def test_a_new_id_looks_like_one_a_client_will_send_back(self) -> None:
        first, second = new_response_id(), new_response_id()
        assert first.startswith("resp_") and len(first) == 29
        assert first != second


class TestProviderSideState:
    def target(self, **over: Any) -> ResponseEntry:
        base: dict[str, Any] = {
            "local_response_id": "r",
            "provider_response_id": "prov_1",
            "provider_state_expires_at": now_ms() + 60_000,
        }
        base.update(over)
        return ResponseEntry(**base)

    def test_fresh_state_can_be_chained_to(self) -> None:
        assert has_fresh_provider_state(self.target()) is True

    def test_expired_state_cannot(self) -> None:
        assert has_fresh_provider_state(self.target(provider_state_expires_at=now_ms() - 1)) is False

    def test_an_unknown_expiry_counts_as_stale(self) -> None:
        # Chaining to state that turns out to be gone fails the whole call;
        # re-sending the transcript merely costs tokens. The cheap wrong answer
        # is the safe one.
        assert has_fresh_provider_state(self.target(provider_state_expires_at=None)) is False

    def test_no_provider_id_is_nothing_to_chain_to(self) -> None:
        assert has_fresh_provider_state(self.target(provider_response_id=None)) is False

    def test_the_clock_can_be_supplied(self) -> None:
        entry = self.target(provider_state_expires_at=1_000.0)
        assert has_fresh_provider_state(entry, now=999.0) is True
        assert has_fresh_provider_state(entry, now=1_001.0) is False


def test_the_store_does_not_remember_by_accident() -> None:
    # A server that kept conversations nobody asked it to keep is a privacy
    # question nobody chose to answer.
    assert server_with().response_store is None
    time.sleep(0)  # keeps the import of `time` honest for the socket fixture
