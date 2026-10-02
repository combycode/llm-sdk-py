"""Anthropic's code-execution container, and the skills loaded into it.

Transposed from `unified-library-ts/tests/unit/llm/anthropic-container.test.ts`.

Fixtures transcribed from live captures (2026-10-02), all on a plain key with no
beta header -- the param is GA. Three of them pin things a reasonable guess gets
wrong:

- When STREAMING, `message_start` carries `container: null` and the real container
  arrives on `message_delta.delta.container`. Reading the opening frame reports
  `None` for every streamed turn.
- A requested `version: "latest"` comes back RESOLVED (`"20260914"`), so the
  response's version is worth reporting rather than echoing the request.
- `container: null` is the NORMAL answer for a turn that ran no code. Not an error,
  and not worth a warning.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest

sys.path.insert(0, "src")

from combycode_llm_sdk.llm.providers.anthropic.container import (
    container_from_wire,
    to_wire_container,
)
from combycode_llm_sdk.llm.providers.anthropic.messages import AnthropicAdapter

ADAPTER = AnthropicAdapter({"apiKey": "k"})

WIRE_CONTAINER: dict[str, Any] = {
    "id": "container_011qsGm3etjwdPk4vQ14efrr",
    "expires_at": "2026-10-02T12:55:18.075903Z",
    "skills": [{"type": "anthropic", "skill_id": "xlsx", "version": "20260914"}],
}


def body_for(provider_options: dict[str, Any]) -> dict[str, Any]:
    req = ADAPTER.build_request(
        {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "maxTokens": 256,
            "tools": [{"type": "code_interpreter"}],
            "providerOptions": provider_options,
        }
    )
    body = req.body if hasattr(req, "body") else req.get("body")
    return dict(body or {})


class TestAskingForAContainer:
    def test_it_sends_an_id_so_a_warm_container_is_reused(self) -> None:
        assert body_for({"container": {"id": "container_abc"}})["container"] == {
            "id": "container_abc"
        }

    def test_it_renames_skill_id_to_the_wire_name(self) -> None:
        # The one thing the spec language cannot do, and the reason this is a named
        # transform rather than a passthrough.
        assert to_wire_container(
            {"skills": [{"type": "anthropic", "skillId": "xlsx", "version": "latest"}]}
        ) == {"skills": [{"type": "anthropic", "skill_id": "xlsx", "version": "latest"}]}

    def test_it_omits_a_version_the_caller_did_not_pin(self) -> None:
        assert to_wire_container({"skills": [{"type": "custom", "skillId": "skl_1"}]}) == {
            "skills": [{"type": "custom", "skill_id": "skl_1"}]
        }

    def test_it_sends_an_id_and_skills_together(self) -> None:
        body = body_for(
            {
                "container": {
                    "id": "container_abc",
                    "skills": [{"type": "anthropic", "skillId": "pdf"}],
                }
            }
        )
        assert body["container"] == {
            "id": "container_abc",
            "skills": [{"type": "anthropic", "skill_id": "pdf"}],
        }

    def test_it_sends_no_empty_keys_for_a_blank_container(self) -> None:
        # `skills: []` would be asking for something they did not ask for.
        assert to_wire_container({}) == {}
        assert to_wire_container({"skills": []}) == {}

    def test_it_refuses_a_skill_ref_with_no_skill_id(self) -> None:
        # Sending it would ask the provider about a field the caller never wrote.
        with pytest.raises(ValueError, match="skillId must be a non-empty string"):
            to_wire_container({"skills": [{"type": "anthropic"}]})

    def test_it_refuses_a_skill_ref_with_an_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="must be 'anthropic' or 'custom'"):
            to_wire_container({"skills": [{"type": "builtin", "skillId": "xlsx"}]})

    def test_it_refuses_something_that_is_not_a_skill_ref_at_all(self) -> None:
        # A bare string is the obvious mistake. Skipping it would load nothing while
        # the caller believes a skill is loaded.
        with pytest.raises(TypeError, match="must be"):
            to_wire_container({"skills": ["xlsx"]})

    def test_it_sends_nothing_when_no_container_was_asked_for(self) -> None:
        # The regression that carries every existing caller of the code tool.
        assert "container" not in body_for({})


class TestReadingTheContainerBack:
    def test_it_camel_cases_it_and_keeps_the_resolved_skill_version(self) -> None:
        # `latest` was the request; `20260914` is what ran, and that is the useful
        # fact.
        assert container_from_wire(WIRE_CONTAINER) == {
            "id": WIRE_CONTAINER["id"],
            "expiresAt": "2026-10-02T12:55:18.075903Z",
            "skills": [{"type": "anthropic", "skillId": "xlsx", "version": "20260914"}],
        }

    def test_it_reports_no_container_for_a_turn_that_ran_no_code(self) -> None:
        # `null` is the normal answer there -- no container was created.
        assert container_from_wire(None) is None

    def test_it_ignores_a_container_with_no_id(self) -> None:
        assert container_from_wire({"expires_at": "x"}) is None

    def test_it_leaves_skills_off_when_none_were_loaded(self) -> None:
        info = container_from_wire({"id": "c", "expires_at": "x"})
        assert info == {"id": "c", "expiresAt": "x"}

    def test_it_leaves_skills_off_for_an_empty_list_too(self) -> None:
        # An absent key and an empty list mean the same thing -- no skills were
        # loaded -- and reporting `skills: []` would invite a caller to believe the
        # field distinguishes them.
        info = container_from_wire({"id": "c", "expires_at": "x", "skills": []})
        assert info is not None
        assert "skills" not in info

    def test_it_appears_on_a_buffered_response(self) -> None:
        parsed = ADAPTER.parse_response(
            {
                "id": "msg_1",
                "model": "claude-sonnet-4-5",
                "role": "assistant",
                "content": [{"type": "text", "text": "4"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "container": WIRE_CONTAINER,
            },
            1,
        )
        assert parsed["container"]["id"] == WIRE_CONTAINER["id"]
        assert parsed["container"]["skills"][0]["skillId"] == "xlsx"

    def test_it_is_absent_from_a_buffered_response_that_had_none(self) -> None:
        parsed = ADAPTER.parse_response(
            {
                "id": "msg_1",
                "model": "claude-sonnet-4-5",
                "role": "assistant",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "container": None,
            },
            1,
        )
        assert not parsed.get("container")


OPENING: dict[str, Any] = {
    "type": "message_start",
    "message": {
        "id": "msg_1",
        "model": "claude-sonnet-4-5",
        "role": "assistant",
        "content": [],
        # Measured: null even on a turn that DOES create a container.
        "container": None,
        "usage": {"input_tokens": 1, "output_tokens": 0},
    },
}

CLOSING: dict[str, Any] = {
    "type": "message_delta",
    "delta": {"stop_reason": "end_turn", "container": WIRE_CONTAINER},
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


def stream_of(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parse = AnthropicAdapter({"apiKey": "k"}).create_stream_parser()
    events: list[dict[str, Any]] = []
    for payload in payloads:
        sse = {"event": payload["type"], "data": json.dumps(payload)}
        events.extend(dict(e) for e in parse(sse))
    return events


class TestTheContainerInAStream:
    def test_it_rides_the_terminal_frame(self) -> None:
        done = next(e for e in stream_of([OPENING, CLOSING]) if e["type"] == "done")
        assert done["container"] == {
            "id": WIRE_CONTAINER["id"],
            "expiresAt": "2026-10-02T12:55:18.075903Z",
            "skills": [{"type": "anthropic", "skillId": "xlsx", "version": "20260914"}],
        }

    def test_it_is_not_taken_from_the_opening_frame(self) -> None:
        # Taking it from `message_start` would report nothing for every streamed
        # turn.
        assert not any(e.get("container") for e in stream_of([OPENING]))

    def test_it_is_not_reported_from_the_opening_frame_even_if_one_appeared(self) -> None:
        # Guards the fix rather than the symptom. `message_start` is measured to
        # send `null`, so reading it also "works" -- until the day it does not, when
        # the same turn would report its container twice, from two different frames,
        # with two `done` events.
        events = stream_of(
            [{**OPENING, "message": {**OPENING["message"], "container": WIRE_CONTAINER}}]
        )
        assert not any(e.get("container") for e in events)
        assert not any(e["type"] == "done" for e in events)

    def test_it_leaves_done_alone_when_the_turn_had_no_container(self) -> None:
        events = stream_of(
            [
                OPENING,
                {"type": "message_delta", "delta": {"stop_reason": "end_turn", "container": None}},
            ]
        )
        done = next(e for e in events if e["type"] == "done")
        assert "container" not in done
