"""The registry-backed system prompt, and the surface compaction is built on.

These are the invariants a compaction strategy relies on and cannot check for
itself: that a rewrite reaches the messages actually being sent, that indices
stay consistent afterwards, and that a token count the provider gave us is not
carried forward onto messages it never saw.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")

from combycode_llm_sdk.agent.history import ConversationHistory
from combycode_llm_sdk.agent.layers import (
    LAYER_LEGACY_SYSTEM,
    LEGACY_SYSTEM_WRITE_PRIORITY,
    PRIORITY_AGENTLOOP_SYSTEM,
)
from combycode_llm_sdk.results import Usage


def _filled(n: int = 5) -> ConversationHistory:
    history = ConversationHistory("conv")
    for i in range(n):
        history.append({"role": "user", "content": f"m{i}"})
    return history


class TestTheSystemPrompt:
    def test_the_setter_round_trips(self) -> None:
        history = ConversationHistory()
        history.system = "You are helpful."
        assert history.system == "You are helpful."

    def test_it_composes_every_system_tagged_contributor(self) -> None:
        # The point of the registry: the guard and the memory writer also
        # contribute, and returning only what the caller set would under-report
        # what the model is about to read.
        history = ConversationHistory()
        history.system = "Legacy text."
        history.registry.set(
            "agentloop.system", "Persona.", priority=PRIORITY_AGENTLOOP_SYSTEM, tags=["system"]
        )
        assert history.system == "Persona.\n\nLegacy text."

    def test_a_layer_without_the_system_tag_is_not_part_of_it(self) -> None:
        history = ConversationHistory()
        history.system = "Legacy text."
        history.registry.set("scratch", "not a system layer", priority=1)
        assert history.system == "Legacy text."

    def test_setting_it_empty_removes_the_layer(self) -> None:
        history = ConversationHistory()
        history.system = "gone in a moment"
        history.system = ""
        assert history.registry.get(LAYER_LEGACY_SYSTEM) is None
        assert history.system == ""

    def test_appending_adds_rather_than_replaces(self) -> None:
        history = ConversationHistory()
        history.system = "First."
        history.append_system("Second.")
        assert history.system == "First.\n\nSecond."

    def test_appending_to_nothing_does_not_lead_with_blank_lines(self) -> None:
        history = ConversationHistory()
        history.append_system("Only.")
        assert history.system == "Only."

    def test_it_writes_where_the_writers_actually_write(self) -> None:
        # The TypeScript exports PRIORITY_LEGACY_SYSTEM = 50 but both of its
        # writers hardcode 200. The port keeps the behaviour, because moving
        # the layer would move the cache prefix for every existing caller.
        history = ConversationHistory()
        history.system = "text"
        layer = history.registry.get(LAYER_LEGACY_SYSTEM)
        assert layer is not None
        assert layer.priority == LEGACY_SYSTEM_WRITE_PRIORITY == 200


class TestReadingTheTranscript:
    def test_at_counts_from_the_end_when_negative(self) -> None:
        history = _filled()
        last, first = history.at(-1), history.at(0)
        assert last is not None and last.message["content"] == "m4"
        assert first is not None and first.message["content"] == "m0"

    def test_an_index_past_the_end_is_nothing_rather_than_an_error(self) -> None:
        assert _filled().at(99) is None
        assert _filled().at(-99) is None

    def test_last_and_filter_and_by_role(self) -> None:
        history = _filled()
        history.append({"role": "assistant", "content": "hi"})
        assert [e.message["content"] for e in history.last(2)] == ["m4", "hi"]
        assert len(history.by_role("user")) == 5
        assert len(history.filter(lambda e: e.message["content"].startswith("m"))) == 5

    def test_it_iterates_its_entries(self) -> None:
        assert len(list(_filled())) == 5

    def test_total_usage_sums_only_what_was_recorded(self) -> None:
        history = ConversationHistory()
        history.append({"role": "assistant", "content": "a"}, usage=Usage(input_tokens=10, output_tokens=2))
        history.append({"role": "assistant", "content": "b"}, usage=Usage(input_tokens=5, output_tokens=3))
        history.append({"role": "user", "content": "no usage here"})
        total = history.total_usage()
        assert (total.input_tokens, total.output_tokens) == (15, 5)

    def test_the_total_does_not_invent_audio(self) -> None:
        # None means no audio was sent, which is not zero -- a zero would price
        # silence as if audio had been billed.
        history = ConversationHistory()
        history.append({"role": "assistant", "content": "a"}, usage=Usage(input_tokens=1))
        assert history.total_usage().audio_input_tokens is None


class TestTheProvidersOwnCount:
    def test_it_anchors_the_estimate_instead_of_re_estimating(self) -> None:
        history = _filled()
        history.record_actual_usage(1000)
        assert history.estimated_tokens() == 1000
        history.append({"role": "user", "content": "x" * 40})
        # 1000 for what the provider counted, plus an estimate of only the new
        # message -- not a fresh guess at the whole transcript.
        assert 1000 < history.estimated_tokens() <= 1020

    def test_an_assistant_reply_records_it_without_being_asked(self) -> None:
        history = _filled()
        history.append({"role": "assistant", "content": "hi"}, usage=Usage(input_tokens=777))
        assert history.last_actual_total == 777

    def test_a_reply_with_no_usage_does_not_reset_the_anchor(self) -> None:
        history = _filled()
        history.record_actual_usage(500)
        history.append({"role": "assistant", "content": "hi"})
        assert history.last_actual_total == 500

    def test_clearing_forgets_the_count_too(self) -> None:
        history = _filled()
        history.record_actual_usage(500)
        history.clear()
        assert history.last_actual_total == 0
        assert history.estimated_tokens() == 0


class TestCompaction:
    def test_splice_replaces_a_range_with_one_entry(self) -> None:
        history = _filled()
        removed = history.splice_range(0, 3, {"role": "user", "content": "[summary]"})
        assert len(removed) == 3
        assert [e.message["content"] for e in history] == ["[summary]", "m3", "m4"]

    def test_it_reindexes_so_the_next_range_is_addressable(self) -> None:
        history = _filled()
        history.splice_range(0, 3, {"role": "user", "content": "[summary]"})
        assert [e.index for e in history] == [0, 1, 2]

    def test_the_summary_inherits_the_time_of_what_it_replaces(self) -> None:
        # A summary of yesterday that claims to have been said just now sorts
        # itself in front of the messages it summarises.
        history = _filled()
        original = history.at(2)
        assert original is not None
        latest = original.timestamp
        history.splice_range(0, 3, {"role": "user", "content": "[summary]"})
        head = history.at(0)
        assert head is not None and head.timestamp == latest

    def test_an_impossible_range_changes_nothing(self) -> None:
        history = _filled()
        assert history.splice_range(3, 1, {"role": "user", "content": "x"}) == []
        assert history.splice_range(0, 99, {"role": "user", "content": "x"}) == []
        assert history.length == 5

    def test_a_rewrite_drops_a_count_that_described_other_messages(self) -> None:
        history = _filled()
        history.record_actual_usage(900)
        history.splice_range(0, 3, {"role": "user", "content": "[summary]"})
        assert history.last_actual_total == 0

    def test_a_rewrite_after_the_counted_range_keeps_the_count(self) -> None:
        history = _filled()
        history.record_actual_usage(900)
        history._last_actual_index = 0
        history.splice_range(3, 5, {"role": "user", "content": "[tail]"})
        assert history.last_actual_total == 900

    def test_truncate_keeps_the_most_recent_and_returns_the_rest(self) -> None:
        history = _filled()
        removed = history.truncate(2)
        assert [e.message["content"] for e in removed] == ["m0", "m1", "m2"]
        assert [e.message["content"] for e in history] == ["m3", "m4"]
        assert [e.index for e in history] == [0, 1]

    def test_truncating_to_more_than_there_is_drops_nothing(self) -> None:
        history = _filled()
        assert history.truncate(99) == []
        assert history.length == 5


class TestForking:
    def test_the_copy_carries_the_entries_and_the_registry(self) -> None:
        history = _filled()
        history.system = "Persona."
        forked = history.fork("other")
        assert forked.id == "other"
        assert forked.length == 5
        assert forked.system == "Persona."

    def test_the_two_do_not_share_a_transcript(self) -> None:
        history = _filled()
        forked = history.fork()
        forked.append({"role": "user", "content": "only mine"})
        assert (history.length, forked.length) == (5, 6)

    def test_the_two_do_not_share_a_registry(self) -> None:
        history = _filled()
        history.system = "Original."
        forked = history.fork()
        forked.system = "Changed."
        assert history.system == "Original."

    def test_editing_a_forked_message_does_not_edit_the_original(self) -> None:
        history = _filled()
        forked = history.fork()
        entry = forked.at(0)
        assert entry is not None
        entry.message["content"] = "tampered"
        original = history.at(0)
        assert original is not None and original.message["content"] == "m0"


class TestTheSnapshot:
    def test_it_survives_a_round_trip(self) -> None:
        history = _filled()
        history.system = "Persona."
        history.set_metadata("contextStrategy", "layered")
        back = ConversationHistory.restore(history.dump())
        assert back.length == 5
        assert back.system == "Persona."
        assert back.metadata["contextStrategy"] == "layered"

    def test_the_registry_wins_over_the_flattened_string(self) -> None:
        # `system` in the snapshot is a VIEW of the registry. Restoring both
        # would write a copy of every layer back in as one more layer.
        history = ConversationHistory()
        history.system = "Legacy."
        history.registry.set(
            "agentloop.system", "Persona.", priority=PRIORITY_AGENTLOOP_SYSTEM, tags=["system"]
        )
        back = ConversationHistory.restore(history.dump())
        assert back.system == "Persona.\n\nLegacy."

    def test_a_snapshot_with_no_registry_still_restores_its_system(self) -> None:
        snapshot: dict[str, Any] = {
            "id": "old",
            "entries": [],
            "system": "From before the registry.",
            "metadata": {},
        }
        back = ConversationHistory.restore(snapshot)
        assert back.system == "From before the registry."


def test_the_strategy_is_recorded_where_the_guard_reads_it() -> None:
    assert ConversationHistory("x", strategy="layered").metadata["contextStrategy"] == "layered"


def test_opting_out_is_recorded_as_false_rather_than_dropped() -> None:
    # `False` and "not set" mean different things to the guard: one opts out,
    # the other takes the default strategy.
    assert ConversationHistory("x", strategy=False).metadata["contextStrategy"] is False
    assert "contextStrategy" not in ConversationHistory("y").metadata
