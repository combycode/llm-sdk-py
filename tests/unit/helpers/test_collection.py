"""A named corner of the engine's store.

The failures here are all one failure: a prefix applied in one place and not the
other. A collection that writes `subagents:alpha` but lists without the prefix
reads every other subsystem's rows and reports them as its own; one that lists
with the prefix but forgets to strip it hands back keys its caller cannot use.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")

import pytest

from combycode_llm_sdk import Engine
from combycode_llm_sdk.helpers.collection import Collection, create_collection
from combycode_llm_sdk.helpers.engine import clear_default_engine
from combycode_llm_sdk.persistence import MemoryPersistence


@pytest.fixture
def engine() -> Any:
    return Engine(persistence=MemoryPersistence(), register_as_default=False)


@pytest.fixture
def subagents(engine: Any) -> Collection[Any]:
    return create_collection("subagents", engine=engine)


class TestTheNamespace:
    def test_a_value_survives_a_round_trip(self, subagents: Collection[Any]) -> None:
        subagents.set("alpha", {"role": "researcher"})
        assert subagents.get("alpha") == {"role": "researcher"}

    def test_a_missing_key_is_none_rather_than_an_error(
        self, subagents: Collection[Any]
    ) -> None:
        assert subagents.get("nobody") is None

    def test_the_prefix_is_written_but_never_returned(
        self, engine: Any, subagents: Collection[Any]
    ) -> None:
        subagents.set("alpha", 1)
        # Stored under the prefix...
        assert engine.persistence.list() == ["subagents:alpha"]
        # ...and answered without it.
        assert subagents.keys() == ["alpha"]

    def test_two_collections_do_not_see_each_other(self, engine: Any) -> None:
        # The whole point: one flat store, shared by every subsystem.
        create_collection("subagents", engine=engine).set("alpha", "a")
        prompts = create_collection("prompts", engine=engine)
        prompts.set("alpha", "p")
        assert prompts.keys() == ["alpha"]
        assert prompts.get("alpha") == "p"
        assert create_collection("subagents", engine=engine).get("alpha") == "a"

    def test_a_key_that_merely_starts_the_same_is_not_included(
        self, engine: Any
    ) -> None:
        # `subagents` must not swallow `subagents_archive`: the separator is
        # part of the prefix precisely so that it cannot.
        create_collection("subagents_archive", engine=engine).set("old", 1)
        assert create_collection("subagents", engine=engine).keys() == []

    def test_delete_removes_it(self, subagents: Collection[Any]) -> None:
        subagents.set("alpha", 1)
        subagents.delete("alpha")
        assert subagents.keys() == []
        assert not subagents.has("alpha")

    def test_deleting_what_is_not_there_is_quiet(
        self, subagents: Collection[Any]
    ) -> None:
        subagents.delete("nobody")


class TestReadingItWhole:
    def test_values_and_items_agree_with_keys(self, subagents: Collection[Any]) -> None:
        subagents.set("a", 1)
        subagents.set("b", 2)
        assert sorted(subagents.keys()) == ["a", "b"]
        assert sorted(subagents.values()) == [1, 2]
        assert sorted(subagents.items()) == [("a", 1), ("b", 2)]

    def test_an_empty_collection_reads_empty(self, subagents: Collection[Any]) -> None:
        assert subagents.keys() == []
        assert subagents.values() == []
        assert subagents.items() == []

    def test_len_and_in_work_the_obvious_way(self, subagents: Collection[Any]) -> None:
        subagents.set("a", 1)
        assert len(subagents) == 1
        assert "a" in subagents
        assert "b" not in subagents

    def test_a_non_string_key_is_simply_absent(self, subagents: Collection[Any]) -> None:
        assert 7 not in subagents

    def test_a_row_that_reads_back_as_none_is_skipped_not_returned(
        self, subagents: Collection[Any]
    ) -> None:
        # The store cannot distinguish "stored None" from "gone since I listed
        # it", and neither is a `T`. So the key is still there and the value is
        # not -- returning `None` in a list of values would push the check onto
        # every caller.
        subagents.set("real", 1)
        subagents.set("empty", None)
        assert sorted(subagents.keys()) == ["empty", "real"]
        assert subagents.values() == [1]
        assert subagents.items() == [("real", 1)]


class TestWhatIsRefused:
    def test_an_empty_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            create_collection("")

    def test_a_name_with_a_slash_is_refused(self) -> None:
        # It would nest under whatever the file backend makes of a separator,
        # so two collections could quietly share a directory.
        with pytest.raises(ValueError, match="without '/'"):
            create_collection("a/b")

    def test_with_no_engine_at_all_it_says_so(self) -> None:
        clear_default_engine()
        try:
            with pytest.raises(ValueError, match="no engine"):
                create_collection("orphans").keys()
        finally:
            clear_default_engine()


class TestTheEnginesStore:
    """What `Engine(persistence=...)` accepts, since the collection needs it."""

    def test_the_default_is_memory_rather_than_nothing(self) -> None:
        # Always present: a caller that had to check for `None` before every
        # write would grow a second in-process store to fall back to, and the
        # two would disagree about what was saved.
        engine = Engine(register_as_default=False)
        assert isinstance(engine.persistence, MemoryPersistence)

    def test_a_store_instance_is_used_as_it_is(self) -> None:
        own = MemoryPersistence()
        assert Engine(persistence=own, register_as_default=False).persistence is own

    def test_a_config_asking_for_memory_gets_memory(self) -> None:
        engine = Engine(persistence={"type": "memory"}, register_as_default=False)
        assert isinstance(engine.persistence, MemoryPersistence)

    def test_a_file_store_needs_somewhere_to_put_it(self) -> None:
        with pytest.raises(ValueError, match="requires a `dir` field"):
            Engine(persistence={"type": "file"}, register_as_default=False)

    def test_an_unknown_type_is_refused_rather_than_defaulted(self) -> None:
        # Falling back to memory would look like it worked and lose every row
        # at exit -- the failure would surface as missing data, not as a typo.
        with pytest.raises(ValueError, match="unknown persistence type"):
            Engine(persistence={"type": "redis"}, register_as_default=False)

    def test_a_file_store_is_built_when_a_directory_is_given(self, tmp_path: Any) -> None:
        engine = Engine(
            persistence={"type": "file", "dir": str(tmp_path)}, register_as_default=False
        )
        engine.persistence.set("k", {"v": 1})
        assert engine.persistence.get("k") == {"v": 1}


class TestTheEngineIsResolvedLate:
    def test_a_collection_built_before_the_engine_still_finds_it(self) -> None:
        # A module-level constant is built at import time, before any engine
        # exists. Capturing the store then would bind it to the wrong one.
        clear_default_engine()
        try:
            early = create_collection("early")
            engine = Engine(persistence=MemoryPersistence(), register_as_default=True)
            early.set("a", 1)
            assert engine.persistence.list() == ["early:a"]
        finally:
            clear_default_engine()
