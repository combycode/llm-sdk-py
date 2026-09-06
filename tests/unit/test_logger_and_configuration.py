"""The logger's eleven renderers, and the configuration registry.

Both are small and neither touches the network, so what is worth testing is
what a rewrite gets wrong: a sink that raises taking the log down with it, a
level filter that formats before it filters, and a settings registry that hands
out the object it stored rather than a copy of it.
"""

from __future__ import annotations

import io
from typing import Any, cast

import pytest

from combycode_llm_sdk.bus.hook_bus import HookBus
from combycode_llm_sdk.configuration import (
    DEFAULT_STORAGE_KEY,
    SNAPSHOT_VERSION,
    ConfigurationPlugin,
)
from combycode_llm_sdk.cost_collector import CostEntry


#: `HookContext` is annotated `dict[str, Any]` for all 51 hooks, but
#: `onCostEntry` really carries the `CostEntry` itself -- see the comment at
#: its emit site. The cast keeps the test emitting what the library emits
#: rather than a dict the library never sends.
def emit_cost(hooks: HookBus, entry: CostEntry) -> None:
    hooks.emit_sync("onCostEntry", cast("dict[str, Any]", entry))
from combycode_llm_sdk.logger import (
    LOG_LEVEL_RANK,
    ConsoleSink,
    LogEvent,
    Logger,
    LogLevel,
    default_format,
)
from combycode_llm_sdk.persistence import MemoryPersistence


class Collecting:
    """A sink that keeps what it was given."""

    def __init__(self) -> None:
        self.events: list[LogEvent] = []

    def log(self, event: LogEvent) -> None:
        self.events.append(event)


class Broken:
    """A sink that always fails."""

    def __init__(self, message: str = "bad") -> None:
        self.message = message
        self.calls = 0

    def log(self, event: LogEvent) -> None:
        self.calls += 1
        raise RuntimeError(self.message)


ALL_LEVELS: tuple[LogLevel, ...] = ("trace", "debug", "info", "warn", "error")


def logger(min_level: LogLevel = "trace") -> tuple[Logger, Collecting]:
    sink = Collecting()
    return Logger([sink], min_level=min_level), sink


class TestLevels:
    def test_the_ranking_is_ordered_by_severity(self) -> None:
        ranks = [LOG_LEVEL_RANK[level] for level in ALL_LEVELS]
        assert ranks == sorted(ranks)
        assert len(set(ranks)) == len(ranks)

    def test_events_below_the_minimum_never_reach_a_sink(self) -> None:
        log, sink = logger(min_level="warn")
        for level in ALL_LEVELS:
            log.log(LogEvent(level=level, source="s", kind="k"))
        assert [e.level for e in sink.events] == ["warn", "error"]

    def test_the_default_minimum_is_info(self) -> None:
        sink = Collecting()
        log = Logger([sink])
        log.log(LogEvent(level="debug", source="s", kind="k"))
        log.log(LogEvent(level="info", source="s", kind="k"))
        assert [e.level for e in sink.events] == ["info"]

    def test_a_logger_with_no_sinks_is_refused(self) -> None:
        # Silent logging looks exactly like a quiet system, which is the one
        # thing a logger must never be mistaken for.
        with pytest.raises(ValueError, match="at least one sink"):
            Logger([])


class TestABrokenSink:
    def test_it_does_not_take_down_the_caller(self) -> None:
        log = Logger([Broken()], min_level="trace")
        log.log(LogEvent(level="info", source="s", kind="k"))  # must not raise

    def test_the_other_sinks_still_receive_the_event(self) -> None:
        good = Collecting()
        log = Logger([Broken(), good], min_level="trace")
        log.log(LogEvent(level="info", source="s", kind="k"))
        assert len(good.events) == 1

    def test_the_failure_is_reported_to_stderr_not_through_the_logger(
        self, capsys: Any
    ) -> None:
        # Routing it back through `log()` would hand it to the same broken sink.
        # That is how one bad write becomes a recursion.
        broken = Broken("disk full")
        log = Logger([broken], min_level="trace")
        log.log(LogEvent(level="info", source="s", kind="completion"))

        err = capsys.readouterr().err
        assert "sink-error: disk full" in err
        assert "original_kind=completion" in err
        assert broken.calls == 1, "the failure must not be re-sent to the sink"


class TestTheConsoleSink:
    def test_warnings_and_errors_go_to_stderr_and_the_rest_to_stdout(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        sink = ConsoleSink(stdout=out, stderr=err)
        for level in ALL_LEVELS:
            sink.log(LogEvent(level=level, source="s", kind=level))

        assert [line.split()[-1] for line in out.getvalue().splitlines()] == [
            "trace", "debug", "info",
        ]
        assert [line.split()[-1] for line in err.getvalue().splitlines()] == ["warn", "error"]

    def test_the_default_line_names_the_level_source_and_kind(self) -> None:
        line = default_format(
            LogEvent(level="warn", source="openai", kind="retry", message="again",
                     timestamp=0.0)
        )
        assert line.startswith("[1970-01-01T00:00:00Z] [WARN] [openai] retry: again")

    def test_ctx_and_data_are_rendered_only_when_present(self) -> None:
        bare = default_format(LogEvent(level="info", source="s", kind="k"))
        assert bare.endswith("k"), "an empty event must not trail a colon and braces"

        full = default_format(
            LogEvent(level="info", source="s", kind="k", message="m",
                     ctx={"requestId": "r1"}, data={"attempt": 2})
        )
        assert '"requestId": "r1"' in full
        assert '"attempt": 2' in full

    def test_a_custom_format_replaces_the_whole_line(self) -> None:
        out = io.StringIO()
        ConsoleSink(format=lambda e: f"<{e.kind}>", stdout=out).log(
            LogEvent(level="info", source="s", kind="k")
        )
        assert out.getvalue() == "<k>\n"

    def test_data_that_does_not_json_serialise_is_still_rendered(self) -> None:
        # A hook payload carries whatever a provider or a plugin put in it. A
        # logger that raises on an unexpected value is worse than one that
        # prints its repr.
        line = default_format(
            LogEvent(level="error", source="s", kind="k", data={"error": ValueError("x")})
        )
        assert "x" in line


class TestEveryHookIsRendered:
    def attached(self, min_level: LogLevel = "trace") -> tuple[HookBus, Collecting]:
        hooks = HookBus()
        sink = Collecting()
        Logger([sink], min_level=min_level).attach(hooks)
        return hooks, sink

    def test_a_warning_becomes_a_warn_event_carrying_its_code(self) -> None:
        hooks, sink = self.attached()
        hooks.emit_sync(
            "onWarning",
            {"source": "files", "code": "file_skipped", "message": "too big",
             "details": {"fileId": "f1"}},
        )
        (event,) = sink.events
        assert (event.level, event.source, event.kind) == ("warn", "files", "warning")
        assert event.message == "too big"
        assert event.data == {"code": "file_skipped", "fileId": "f1"}

    def test_a_completion_reports_the_token_counts(self) -> None:
        hooks, sink = self.attached()
        hooks.emit_sync(
            "onCompletion",
            {
                "provider": "openai",
                "model": "gpt-5.4-nano",
                "response": {"usage": {"inputTokens": 12, "outputTokens": 3},
                             "finishReason": "stop", "latencyMs": 240},
                "ctx": {"requestId": "r1"},
            },
        )
        (event,) = sink.events
        assert event.message == "openai/gpt-5.4-nano 12->3 tok"
        assert event.level == "info"
        assert event.ctx == {"requestId": "r1"}

    def test_a_retryable_model_error_is_a_warning_and_a_final_one_an_error(self) -> None:
        # Logging a failure that is about to be retried at error level trains
        # people to ignore the level.
        hooks, sink = self.attached()
        for will_retry in (True, False):
            hooks.emit_sync(
                "onModelError",
                {"provider": "openai", "error": {"message": "boom", "kind": "server_error"},
                 "queueName": "openai", "attempt": 1, "willRetry": will_retry},
            )
        assert [e.level for e in sink.events] == ["warn", "error"]
        assert sink.events[0].message == "boom (openai)"

    def test_a_retry_says_which_attempt_and_how_long(self) -> None:
        hooks, sink = self.attached()
        hooks.emit_sync(
            "onRetry",
            {"provider": "xai", "attempt": 2, "reason": "rate_limit", "backoffMs": 800},
        )
        (event,) = sink.events
        assert event.message == "retry #2 (rate_limit) after 800ms"
        assert event.data == {"attempt": 2, "reason": "rate_limit", "backoffMs": 800}

    def test_a_rate_limit_names_the_status(self) -> None:
        hooks, sink = self.attached()
        hooks.emit_sync(
            "onRateLimitHit", {"provider": "google", "status": 429, "retryAfterMs": 1200}
        )
        (event,) = sink.events
        assert event.message == "rate limited (HTTP 429)"
        assert event.kind == "rate_limit"

    def test_media_events_report_kind_and_count(self) -> None:
        hooks, sink = self.attached()
        hooks.emit_sync("onMediaGenerated", {"provider": "openai", "mediaType": "image",
                                             "count": 3})
        hooks.emit_sync("onMediaError", {"provider": "openai", "error": "quota"})
        assert [e.message for e in sink.events] == ["image x3", "quota"]
        assert [e.level for e in sink.events] == ["info", "error"]

    def test_an_internal_error_names_its_source(self) -> None:
        hooks, sink = self.attached()
        hooks.emit_sync(
            "onInternalError",
            {"source": "queue", "error": {"message": "leak"}, "queueName": "q", "provider": "p"},
        )
        (event,) = sink.events
        assert (event.level, event.source, event.message) == ("error", "queue", "leak")

    def test_budget_events_report_the_percentage_and_the_overrun(self) -> None:
        hooks, sink = self.attached()
        hooks.emit_sync(
            "onBudgetWarning",
            {"budgetId": "daily", "percentage": 82.4, "current": 8.24, "limit": 10},
        )
        hooks.emit_sync("onBudgetExceeded", {"budgetId": "daily", "current": 10.5, "limit": 10})
        assert sink.events[0].message == "budget daily at 82%"
        assert sink.events[1].message == "budget daily exceeded ($10.5000 / $10)"
        assert [e.source for e in sink.events] == ["cost", "cost"]

    def test_a_cost_entry_is_logged_at_debug_with_its_total(self) -> None:
        from combycode_llm_sdk.cost import Cost

        hooks, sink = self.attached()
        emit_cost(
            hooks,
            CostEntry(
                id="c1", timestamp=0.0, provider="anthropic", model="claude-haiku-4.5",
                tokens={}, cost=Cost(input=0.001, output=0.002, total=0.003, source="catalog"),
            ),
        )
        (event,) = sink.events
        assert event.level == "debug"
        assert event.message == "$0.003000 (catalog)"

    def test_an_unpriced_call_is_not_logged_as_free(self) -> None:
        # `cost` is None precisely so "we could not price it" and "it cost
        # nothing" cannot be confused. Printing $0.000000 would confuse them.
        hooks, sink = self.attached()
        emit_cost(
            hooks,
            CostEntry(id="c2", timestamp=0.0, provider="xai", model="new-model",
                      tokens={}, cost=None),
        )
        (event,) = sink.events
        assert "unpriced" in event.message
        assert "0.000000" not in event.message

    def test_detaching_stops_the_flow(self) -> None:
        hooks = HookBus()
        sink = Collecting()
        log = Logger([sink], min_level="trace").attach(hooks)
        hooks.emit_sync("onWarning", {"source": "s", "code": "c", "message": "before"})
        log.detach()
        hooks.emit_sync("onWarning", {"source": "s", "code": "c", "message": "after"})
        assert [e.message for e in sink.events] == ["before"]

    def test_it_can_be_used_as_a_context_manager(self) -> None:
        hooks = HookBus()
        sink = Collecting()
        with Logger([sink], min_level="trace").attach(hooks):
            hooks.emit_sync("onWarning", {"source": "s", "code": "c", "message": "inside"})
        hooks.emit_sync("onWarning", {"source": "s", "code": "c", "message": "outside"})
        assert [e.message for e in sink.events] == ["inside"]

    def test_every_hook_it_claims_is_one_the_bus_knows(self) -> None:
        # A typo in a hook name would subscribe to nothing and the renderer
        # would never fire -- silently, which is the failure mode this checks.
        from combycode_llm_sdk.bus.hook_map import HOOK_NAMES

        hooks = HookBus()
        Logger([Collecting()], min_level="trace").attach(hooks)
        claimed = set(Logger([Collecting()])._handlers())
        assert claimed <= set(HOOK_NAMES)
        assert len(claimed) == 11


class TestTheConfigurationRegistry:
    def test_a_name_round_trips(self) -> None:
        config = ConfigurationPlugin()
        config.set("prod", {"rateLimit": 100, "retry": {"attempts": 3}})
        assert config.get("prod") == {"rateLimit": 100, "retry": {"attempts": 3}}

    def test_an_unknown_name_is_none_rather_than_an_empty_bundle(self) -> None:
        # Empty settings and no settings mean different things to a consumer:
        # one says "use these", the other "use your own defaults".
        assert ConfigurationPlugin().get("nope") is None

    def test_an_empty_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ConfigurationPlugin().set("", {"a": 1})

    def test_settings_that_are_not_a_mapping_are_refused_by_type(self) -> None:
        with pytest.raises(TypeError, match="mapping"):
            ConfigurationPlugin().set("x", cast("dict[str, Any]", [1, 2]))

    def test_the_stored_copy_is_independent_of_the_caller(self) -> None:
        # A caller who keeps editing the dict they passed must not be editing
        # the configuration of a running system.
        config = ConfigurationPlugin()
        mine = {"retry": {"attempts": 3}}
        config.set("prod", mine)
        mine["retry"]["attempts"] = 99
        stored = config.get("prod")
        assert stored is not None and stored["retry"] == {"attempts": 3}

    def test_what_a_consumer_reads_cannot_be_edited(self) -> None:
        config = ConfigurationPlugin()
        config.set("prod", {"a": 1})
        got = config.get("prod")
        assert got is not None
        with pytest.raises(TypeError):
            got["a"] = 2  # type: ignore[index]

    def test_extend_stores_only_the_overrides_and_get_merges(self) -> None:
        config = ConfigurationPlugin()
        config.set("base", {"rateLimit": 100, "timeout": 30})
        config.extend("base", "batch", {"rateLimit": 10})

        assert config.get("batch") == {"rateLimit": 10, "timeout": 30}
        assert config.get("base") == {"rateLimit": 100, "timeout": 30}

    def test_changing_the_base_changes_everything_derived_from_it(self) -> None:
        # The whole point of storing only the overrides.
        config = ConfigurationPlugin()
        config.set("base", {"timeout": 30})
        config.extend("base", "batch", {"rateLimit": 10})
        config.set("base", {"timeout": 60})
        assert config.get("batch") == {"rateLimit": 10, "timeout": 60}

    def test_a_chain_several_deep_resolves_child_over_parent(self) -> None:
        config = ConfigurationPlugin()
        config.set("a", {"x": 1, "y": 1, "z": 1})
        config.extend("a", "b", {"y": 2, "z": 2})
        config.extend("b", "c", {"z": 3})
        assert config.get("c") == {"x": 1, "y": 2, "z": 3}

    def test_extending_an_unknown_base_says_what_is_registered(self) -> None:
        config = ConfigurationPlugin()
        config.set("prod", {})
        with pytest.raises(KeyError, match="prod"):
            config.extend("staging", "staging-batch", {})

    def test_a_nested_value_is_replaced_wholesale_not_merged(self) -> None:
        # Documented, because the other choice is defensible: a half-merged
        # nested policy is harder to reason about than a replaced one.
        config = ConfigurationPlugin()
        config.set("base", {"retry": {"attempts": 3, "backoff": "exponential"}})
        config.extend("base", "child", {"retry": {"attempts": 1}})
        assert config.get("child") == {"retry": {"attempts": 1}}

    def test_has_and_names_see_derived_names_too(self) -> None:
        config = ConfigurationPlugin()
        config.set("base", {})
        config.extend("base", "child", {})
        assert config.has("child") and "child" in config
        assert config.names() == ["base", "child"]
        assert len(config) == 2

    def test_deleting_a_base_does_not_cascade(self) -> None:
        # Deleting a base should not silently delete work derived from it.
        config = ConfigurationPlugin()
        config.set("base", {"timeout": 30})
        config.extend("base", "child", {"rateLimit": 10})
        config.delete("base")

        assert config.has("child")
        assert config.get("child") == {"rateLimit": 10}
        assert config.get("base") is None

    def test_initial_entries_are_seeded_and_copied(self) -> None:
        seed = {"prod": {"a": 1}}
        config = ConfigurationPlugin(initial=seed)
        seed["prod"]["a"] = 2
        assert config.get("prod") == {"a": 1}


class TestPersistingIt:
    def test_a_snapshot_round_trips_through_a_store(self) -> None:
        store = MemoryPersistence()
        config = ConfigurationPlugin(persistence=store)
        config.set("base", {"timeout": 30})
        config.extend("base", "child", {"rateLimit": 10})
        config.save()

        restored = ConfigurationPlugin(persistence=store)
        assert restored.load() is True
        assert restored.get("child") == {"rateLimit": 10, "timeout": 30}

    def test_loading_from_an_empty_store_says_so_rather_than_failing(self) -> None:
        assert ConfigurationPlugin(persistence=MemoryPersistence()).load() is False

    def test_loading_with_no_store_is_false_and_saving_is_an_error(self) -> None:
        # Asymmetric on purpose: nothing to load is a fact, nothing to save to
        # is a mistake in how the plugin was built.
        config = ConfigurationPlugin()
        assert config.load() is False
        with pytest.raises(RuntimeError, match="no persistence"):
            config.save()

    def test_it_is_stored_under_a_named_key(self) -> None:
        store = MemoryPersistence()
        ConfigurationPlugin(persistence=store, initial={"a": {}}).save()
        assert store.has(DEFAULT_STORAGE_KEY)

        other = MemoryPersistence()
        ConfigurationPlugin(persistence=other, storage_key="cfg").save()
        assert other.has("cfg") and not other.has(DEFAULT_STORAGE_KEY)

    def test_an_unknown_snapshot_version_is_refused_by_number(self) -> None:
        # A file written by something newer must fail with its version, not with
        # a KeyError three frames deeper.
        with pytest.raises(ValueError, match="99"):
            ConfigurationPlugin().deserialize({"version": 99, "entries": {}, "parents": {}})

    def test_deserialize_replaces_rather_than_merges(self) -> None:
        config = ConfigurationPlugin()
        config.set("old", {"a": 1})
        config.deserialize({"version": SNAPSHOT_VERSION, "entries": {"new": {"b": 2}},
                            "parents": {}})
        assert config.names() == ["new"]

    def test_a_cycle_in_a_loaded_snapshot_terminates(self) -> None:
        # `extend` cannot make one, but a hand-edited file can, and a resolver
        # that loops forever on it is worse than one that returns what it found.
        #
        # Note the failure mode if the `seen` guard is ever removed: this test
        # HANGS rather than failing, because the loop is CPU-bound and there is
        # no portable way to bound one in-process. Measured with the guard
        # deleted -- pytest never returns. If this file ever stops finishing,
        # look here first.
        config = ConfigurationPlugin()
        config.deserialize(
            {"version": SNAPSHOT_VERSION,
             "entries": {"a": {"x": 1}, "b": {"y": 2}},
             "parents": {"a": "b", "b": "a"}}
        )
        assert config.get("a") == {"y": 2, "x": 1}
