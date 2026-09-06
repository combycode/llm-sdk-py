"""What a hook handler receives: attribute access, and typed events.

The bus carries wire-shaped keys (`toolName`) and a Python subscriber writes
`ctx.tool_name`. Both spellings work and each has a job -- these pin which is
which, and that mutation still reaches the emitter, because several hooks are
in/out parameters.
"""

from __future__ import annotations

from typing import Any

import pytest

from combycode_llm_sdk.bus.context import HookContext, as_context, to_camel
from combycode_llm_sdk.bus.events import (
    CompletionEvent,
    HookEvent,
    RetryContext,
    RetryEvent,
    ToolCallStartEvent,
    WarningEvent,
    build_event,
    to_snake,
)
from combycode_llm_sdk.bus.hook_bus import HookBus


class TestTheContextView:
    def test_a_snake_attribute_reads_a_camel_key(self) -> None:
        ctx = HookContext({"toolName": "search", "step": 2})
        assert ctx.tool_name == "search"
        assert ctx.step == 2

    def test_mapping_access_still_works(self) -> None:
        # Every handler inside the library reads it this way, and the field set
        # is open -- a plugin adding a key must not break a reader.
        ctx = HookContext({"toolName": "search"})
        assert ctx["toolName"] == "search"
        assert dict(ctx) == {"toolName": "search"}

    def test_a_missing_field_names_what_is_there(self) -> None:
        with pytest.raises(AttributeError, match="carries: step, toolName"):
            HookContext({"toolName": "s", "step": 1}).banana  # noqa: B018

    def test_writing_reaches_the_underlying_dict(self) -> None:
        # `onBeforeSubmit` lets a cache short-circuit the call by writing here.
        payload: dict[str, Any] = {"intercepted": False}
        ctx = HookContext(payload)
        ctx["intercepted"] = True
        assert payload["intercepted"] is True

    def test_an_attribute_write_updates_the_existing_spelling(self) -> None:
        # Otherwise a handler setting `ctx.tool_name` would add a second key
        # beside `toolName` and the emitter would keep reading the old one.
        payload = {"toolName": "search"}
        ctx = HookContext(payload)
        ctx.tool_name = "replaced"
        assert payload == {"toolName": "replaced"}

    def test_wrapping_is_idempotent(self) -> None:
        # Hooks are re-emitted, and a view of a view resolves names one layer
        # deeper on every hop.
        once = as_context({"a": 1})
        assert as_context(once) is once

    def test_a_non_mapping_payload_passes_through(self) -> None:
        assert as_context("not a payload") == "not a payload"

    def test_to_camel_leaves_a_single_word_alone(self) -> None:
        assert to_camel("code") == "code"
        assert to_camel("queue_length") == "queueLength"


class TestDeliveryThroughTheBus:
    def test_a_handler_receives_the_view(self) -> None:
        bus = HookBus()
        seen: list[Any] = []
        bus.on("onToolCallStart", seen.append)
        bus.emit_sync("onToolCallStart", {"toolName": "search", "step": 1})
        assert seen[0].tool_name == "search"

    def test_one_handlers_write_is_seen_by_the_next(self) -> None:
        # The ordering ContextGuard depends on.
        bus = HookBus()
        bus.on("onMessageResolve", lambda ctx: ctx.__setitem__("messages", ["rewritten"]))
        seen: list[Any] = []
        bus.on("onMessageResolve", lambda ctx: seen.append(ctx["messages"]))
        payload: dict[str, Any] = {"messages": ["original"]}
        bus.emit_sync("onMessageResolve", payload)
        assert seen == [["rewritten"]]
        assert payload["messages"] == ["rewritten"]

    def test_nothing_is_wrapped_when_nobody_subscribed(self) -> None:
        # The hot paths emit once per stream chunk.
        bus = HookBus()
        bus.emit_sync("onToolCallStart", {"toolName": "search"})


class TestTheEventValues:
    def test_a_named_hook_gets_its_class(self) -> None:
        assert isinstance(build_event("onCompletion", {}), CompletionEvent)
        assert isinstance(build_event("onWarning", {}), WarningEvent)
        assert isinstance(build_event("onToolCallStart", {}), ToolCallStartEvent)
        assert isinstance(build_event("onRetry", {}), RetryEvent)

    def test_an_unnamed_hook_still_arrives_carrying_its_name(self) -> None:
        # A logger never has to enumerate the catalog.
        event = build_event("onCostEntry", {"x": 1})
        assert type(event) is HookEvent
        assert event.type == "on_cost_entry"

    def test_the_type_is_the_name_you_would_subscribe_with(self) -> None:
        assert to_snake("onCostEntry") == "on_cost_entry"
        assert build_event("onCompletion", {}).type == "on_completion"

    def test_an_event_matches_structurally(self) -> None:
        matched = None
        match build_event("onWarning", {"code": "unpriced_model"}):
            case WarningEvent(ctx=ctx):
                matched = ctx["code"]
            case _:
                matched = "fell through"
        assert matched == "unpriced_model"

    def test_one_can_be_built_from_a_context_alone(self) -> None:
        # An event is a value: storable, queueable, replayable, because the name
        # and the payload are one object.
        event = RetryEvent(ctx=RetryContext(provider="openai", model="m", attempt=1, reason="429"))
        assert event.type == "on_retry"
        assert event.ctx.reason == "429"

    def test_the_bus_delivers_events_to_a_catch_all(self) -> None:
        bus = HookBus()
        seen: list[Any] = []
        bus.on_any(seen.append)
        bus.emit_sync("onWarning", {"code": "x"})
        assert isinstance(seen[0], WarningEvent)
        assert seen[0].ctx.code == "x"
