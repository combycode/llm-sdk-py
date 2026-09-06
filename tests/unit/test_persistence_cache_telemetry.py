"""Where bytes land, what a cached answer belongs to, and what must never leave.

Three small subsystems that share a shape: an interface with more than one
backend, where the failure is two backends that quietly disagree.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, ClassVar

import pytest

from combycode_llm_sdk import (
    Cache,
    Engine,
    FilePersistence,
    MemoryCacheStore,
    MemoryPersistence,
    Persistence,
    PersistentCacheStore,
    TelemetryAdapter,
)
from combycode_llm_sdk.cache import (
    CacheEntry,
    make_storage_key,
    parse_storage_key,
    request_cache_key,
)
from combycode_llm_sdk.persistence import decode_key, encode_key
from combycode_llm_sdk.telemetry import REDACTED, redact_headers, redact_url

KEY = "run:2026-08-30:quarterly-report"


class NotOurStore:
    """An application's own backend, written against nothing of ours."""

    def __init__(self) -> None:
        self._rows: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self._rows.get(key)

    def set(self, key: str, value: Any) -> None:
        self._rows[key] = value

    def delete(self, key: str) -> None:
        self._rows.pop(key, None)

    def list(self, prefix: str | None = None) -> list[str]:
        return [k for k in self._rows if prefix is None or k.startswith(prefix)]

    def has(self, key: str) -> bool:
        return key in self._rows


class TestPersistence:
    def test_the_protocol_is_structural(self) -> None:
        # An application's own store should not have to import this library to
        # be usable by it.
        assert isinstance(NotOurStore(), Persistence)
        assert isinstance(MemoryPersistence(), Persistence)

    def test_something_missing_a_method_is_not_one(self) -> None:
        class Partial:
            def get(self, key: str) -> Any:
                return None

        assert not isinstance(Partial(), Persistence)

    def test_a_key_never_written_reads_as_none(self) -> None:
        # Raising would make the ordinary state of a fresh store an error every
        # caller has to catch before it can start.
        assert MemoryPersistence().get("nope") is None
        with tempfile.TemporaryDirectory() as directory:
            assert FilePersistence(directory).get("nope") is None

    def test_memory_copies_so_a_read_cannot_edit_the_checkpoint(self) -> None:
        # The file backend physically cannot hand back a live reference, and two
        # backends that disagree about this are not interchangeable.
        store = MemoryPersistence()
        store.set("draft", {"steps": ["fetch"]})
        store.get("draft")["steps"].append("extract")
        assert store.get("draft") == {"steps": ["fetch"]}

    def test_memory_copies_on_the_way_in_too(self) -> None:
        store = MemoryPersistence()
        value = {"steps": ["fetch"]}
        store.set("draft", value)
        value["steps"].append("mutated after")
        assert store.get("draft") == {"steps": ["fetch"]}

    def test_a_key_with_colons_round_trips_through_a_filename(self) -> None:
        # A colon is how keys are namespaced and is not legal in a Windows
        # filename.
        with tempfile.TemporaryDirectory() as directory:
            store = FilePersistence(directory)
            store.set(KEY, ["fetch"])
            assert store.list("run:") == [KEY]
            assert store.get(KEY) == ["fetch"]
            store.delete(KEY)
            assert not store.has(KEY)

    def test_the_escaping_is_reversible(self) -> None:
        assert decode_key(encode_key(KEY)) == KEY

    def test_a_prefix_filters_the_listing(self) -> None:
        store = MemoryPersistence()
        store.set("task:a", 1)
        store.set("cache:b", 2)
        assert store.list("task:") == ["task:a"]

    def test_the_two_backends_resume_the_same_way(self) -> None:
        # A resume proved against memory in a test is the same code that resumes
        # from disk in production.
        with tempfile.TemporaryDirectory() as directory:
            for store in (MemoryPersistence(), FilePersistence(directory), NotOurStore()):
                store.set(KEY, ["fetch", "extract"])
                assert list(store.get(KEY)) == ["fetch", "extract"]
                assert store.has(KEY)

    def test_a_write_is_atomic(self, tmp_path: Path) -> None:
        # Written beside and moved into place, so a crash mid-write leaves the
        # previous checkpoint intact rather than half a file.
        store = FilePersistence(tmp_path)
        store.set("a", {"v": 1})
        store.set("a", {"v": 2})
        assert store.get("a") == {"v": 2}
        assert list(tmp_path.glob("*.tmp")) == []


class TestTheCacheKey:
    REQUEST: ClassVar[dict[str, Any]] = {
        "model": "gpt-5.4-nano",
        "input": [{"role": "user", "content": "hi"}],
    }

    def test_two_services_offering_one_model_do_not_share_a_key(self) -> None:
        # The one that costs something: without the route in the key, one caller
        # is handed the other service's completion and nothing says it is wrong.
        assert request_cache_key(self.REQUEST, provider="openai") != request_cache_key(
            self.REQUEST, provider="openrouter"
        )

    def test_nor_one_service_at_two_base_urls(self) -> None:
        assert request_cache_key(self.REQUEST, provider="openai") != request_cache_key(
            self.REQUEST, provider="openai", base_url="https://contoso.openai.azure.com"
        )

    def test_the_same_request_in_another_order_is_the_same_request(self) -> None:
        # A key that disagreed would just miss the cache, silently.
        reordered = {"input": self.REQUEST["input"], "model": self.REQUEST["model"]}
        assert request_cache_key(reordered, provider="openai") == request_cache_key(
            self.REQUEST, provider="openai"
        )

    def test_message_order_still_matters(self) -> None:
        # The order of the messages IS the conversation.
        swapped = {
            "model": "m",
            "input": [{"role": "user", "content": "b"}, {"role": "user", "content": "a"}],
        }
        original = {
            "model": "m",
            "input": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
        }
        assert request_cache_key(swapped, provider="p") != request_cache_key(
            original, provider="p"
        )

    def test_a_storage_key_round_trips(self) -> None:
        assert parse_storage_key(make_storage_key("tenant-7", "abc")) == ("tenant-7", "abc")


class TestTheCache:
    def test_an_entry_comes_back(self) -> None:
        cache = Cache(store=MemoryCacheStore())
        cache.set(None, "k", "Paris.")
        assert cache.get(None, "k") == "Paris."

    def test_another_namespace_does_not_see_it(self) -> None:
        cache = Cache(store=MemoryCacheStore())
        cache.set("tenant-7", "k", "theirs")
        assert cache.get("tenant-9", "k") is None

    def test_an_expired_entry_is_not_served(self) -> None:
        cache = Cache(store=MemoryCacheStore())
        cache.store.set(
            make_storage_key("default", "k"),
            CacheEntry(body="answered in 1970", stored_at=0.0, ttl_ms=60_000),
        )
        assert cache.get(None, "k") is None

    def test_the_read_that_found_it_expired_drops_it(self) -> None:
        # Expiry is checked on READ, so there is no sweeper thread and nothing
        # has to sleep to prove it works.
        cache = Cache(store=MemoryCacheStore())
        cache.store.set(
            make_storage_key("default", "k"),
            CacheEntry(body="stale", stored_at=0.0, ttl_ms=60_000),
        )
        cache.get(None, "k")
        assert cache.store.keys() == []

    def test_an_infinite_ttl_never_expires(self) -> None:
        entry = CacheEntry(body="x", stored_at=0.0, ttl_ms=float("inf"))
        assert entry.expired() is False

    def test_invalidate_is_scoped_to_its_namespace(self) -> None:
        cache = Cache(store=MemoryCacheStore())
        cache.set("tenant-7", "k", "theirs")
        assert cache.invalidate(cache_name="tenant-9") == 0
        assert cache.invalidate(cache_name="tenant-7") == 1
        assert cache.get("tenant-7", "k") is None

    def test_an_entry_on_disk_outlives_the_process(self, tmp_path: Path) -> None:
        Cache(store=PersistentCacheStore(FilePersistence(tmp_path))).set("t", "k", "kept")
        reopened = Cache(store=PersistentCacheStore(FilePersistence(tmp_path)))
        assert reopened.get("t", "k") == "kept"


class TestRedaction:
    def test_a_credential_in_a_url_is_always_redacted(self) -> None:
        # No switch reaches this tier, because a switch is a thing somebody
        # eventually flips.
        cleaned = redact_url("https://api.example.com/v1?key=sk-secret-123&model=x")
        assert "sk-secret-123" not in cleaned
        assert "model=x" in cleaned

    def test_a_credential_in_a_header_is_always_redacted(self) -> None:
        cleaned = redact_headers({"authorization": "Bearer sk-1", "content-type": "json"})
        assert cleaned["authorization"] == REDACTED
        assert cleaned["content-type"] == "json"

    def test_free_text_is_redacted_by_default(self) -> None:
        # A provider's error routinely echoes the request back.
        adapter = TelemetryAdapter()
        engine = Engine(register_as_default=False, plugins=[adapter])
        engine.emit_warning(
            source="llm", code="provider_error", message="bad value: alice@example.com"
        )
        recorded = adapter.events[-1]
        assert "alice@example.com" not in str(recorded)
        # The CODE is structural and must survive.
        assert recorded.code == "provider_error"

    def test_opting_out_actually_opts_out(self) -> None:
        adapter = TelemetryAdapter(redact_free_text=False)
        engine = Engine(register_as_default=False, plugins=[adapter])
        engine.emit_warning(source="llm", code="e", message="alice@example.com")
        assert "alice@example.com" in str(adapter.events[-1])

    def test_credentials_survive_no_switch(self) -> None:
        adapter = TelemetryAdapter(redact_free_text=False)
        engine = Engine(register_as_default=False, plugins=[adapter])
        engine.emit_request(
            url="https://api.example.com/v1?key=sk-secret-123",
            headers={"authorization": "Bearer sk-secret-123"},
        )
        assert "sk-secret-123" not in str(adapter.events[-1])


class TestTraces:
    def test_a_completion_produces_one_llm_span(self) -> None:
        adapter = TelemetryAdapter(traces=True)
        engine = Engine(register_as_default=False, plugins=[adapter])
        seen: list[Any] = []
        adapter.on_span_end(seen.append)
        engine.emit_completion(provider="openai", model="m", input_tokens=10, output_tokens=5)

        spans = [s for s in seen if s.kind == "llm"]
        assert len(spans) == 1
        assert spans[0].duration_ms >= 0
        assert spans[0].attributes["provider"] == "openai"
        assert spans[0].trace_id is not None

    def test_no_spans_without_traces_on(self) -> None:
        # A span per event would make a trace unreadable, so it is opt-in.
        adapter = TelemetryAdapter()
        engine = Engine(register_as_default=False, plugins=[adapter])
        engine.emit_completion(provider="openai", model="m")
        assert adapter.spans == []

    def test_events_are_recorded_whether_or_not_traces_are_on(self) -> None:
        adapter = TelemetryAdapter()
        engine = Engine(register_as_default=False, plugins=[adapter])
        engine.emit_completion(provider="openai", model="m")
        assert adapter.events

    def test_a_hook_with_no_span_kind_records_no_span(self) -> None:
        adapter = TelemetryAdapter(traces=True)
        engine = Engine(register_as_default=False, plugins=[adapter])
        engine.emit_warning(source="x", code="y", message="z")
        assert adapter.spans == []
        assert adapter.events

    def test_the_buffer_is_bounded(self) -> None:
        # An unbounded buffer in a long-lived process is a memory leak that only
        # shows up in production.
        adapter = TelemetryAdapter(max_events=5)
        engine = Engine(register_as_default=False, plugins=[adapter])
        for _ in range(20):
            engine.emit_warning(source="x", code="y", message="z")
        assert len(adapter.events) == 5

    def test_detaching_stops_recording(self) -> None:
        adapter = TelemetryAdapter()
        engine = Engine(register_as_default=False, plugins=[adapter])
        adapter.detach()
        engine.emit_completion(provider="openai", model="m")
        assert adapter.events == []

    def test_an_unknown_field_says_what_the_event_carries(self) -> None:
        adapter = TelemetryAdapter()
        engine = Engine(register_as_default=False, plugins=[adapter])
        engine.emit_warning(source="x", code="y", message="z")
        with pytest.raises(AttributeError, match="carries"):
            adapter.events[-1].banana  # noqa: B018
