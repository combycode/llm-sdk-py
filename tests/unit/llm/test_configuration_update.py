"""Changing how hard a stored conversation thinks, from this turn on.

Transposed from `unified-library-ts/tests/unit/llm/configuration-update.test.ts`.

`thinking.effort` already existed and was not enough: it applies to ITS request
and nothing else. Measured on `gpt-5.6-luna` on 2026-10-01, three runs per arm,
setting the effort in turn 1 and naming nothing in turn 2:

    via `configuration_update`   turn 2 reasoning tokens 0, 0, 0
    via the top-level option     turn 2 reasoning tokens 244, 189, 172
    nothing at all               turn 2 reasoning tokens 155, 129, 198

The option does not persist; the item does. `none` is the arm that settles it --
a flat zero against a default near 170 leaves nothing to argue about -- which is
why the part's vocabulary has to be able to say it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, "src")

from combycode_llm_sdk.llm.providers.openai.reasoning_effort import OPENAI_REASONING_EFFORT
from combycode_llm_sdk.llm.providers.openai.responses import OpenAIResponsesAdapter

ADAPTER = OpenAIResponsesAdapter({"apiKey": "test-key"})

#: What OpenAI stores, copied verbatim from a conversation items listing on
#: 2026-10-01 (`gpt-5.6-luna`).
STORED: dict[str, Any] = {
    "id": "cnfu_0fafbf4bd614c625006abe7d27d0688193a610cfa5c9ecb349",
    "type": "configuration_update",
    "reasoning": {"effort": "high"},
}


def update(effort: str) -> dict[str, Any]:
    return {"type": "configuration_update", "reasoning": {"effort": effort}}


def wire_input(messages: list[dict[str, Any]]) -> list[Any]:
    """The `input` array the adapter builds for one conversation."""
    built = ADAPTER.build_request({"model": "gpt-5.6-sol", "messages": messages})
    return list(built.body["input"])


def configuration_items(items: list[Any]) -> list[Any]:
    return [i for i in items if isinstance(i, dict) and i.get("type") == "configuration_update"]


class TestTheUpdateOnTheWayOut:
    def test_it_is_its_own_top_level_item_not_message_content(self) -> None:
        items = wire_input(
            [{"role": "user", "content": [update("high"), {"type": "text", "text": "hi"}]}]
        )
        assert configuration_items(items) == [
            {"type": "configuration_update", "reasoning": {"effort": "high"}}
        ]
        message = next(i for i in items if isinstance(i, dict) and i.get("role") == "user")
        assert "hi" in json.dumps(message)
        assert "configuration_update" not in json.dumps(message)

    def test_it_comes_before_the_message_it_travels_with(self) -> None:
        # The API applies an update to SUBSEQUENT responses. Placed after the
        # message it was meant to govern it governs the next one instead, and the
        # caller sees their change take effect a turn late with nothing reporting
        # it.
        items = wire_input(
            [{"role": "user", "content": [{"type": "text", "text": "hi"}, update("low")]}]
        )
        assert items[0]["type"] == "configuration_update"

    def test_it_travels_on_an_assistant_turn_too(self) -> None:
        # A conversation restored from history carries the update that was in it.
        # If only user turns emitted it, a replayed transcript would quietly drop
        # the configuration and think harder (or less) than the original.
        items = wire_input(
            [{"role": "assistant", "content": [update("none"), {"type": "text", "text": "ok"}]}]
        )
        assert configuration_items(items)

    def test_it_does_not_echo_the_provider_id(self) -> None:
        # `cnfu_...` names the STORED item; sending it back claims to update an
        # item that already exists, and nothing requires the round trip.
        part = {**update("high"), "id": "cnfu_abc"}
        items = wire_input([{"role": "user", "content": [part, {"type": "text", "text": "hi"}]}])
        assert "id" not in configuration_items(items)[0]

    def test_it_sends_nothing_when_no_part_asks_for_it(self) -> None:
        items = wire_input([{"role": "user", "content": "plain string content"}])
        assert not configuration_items(items)

    def test_max_is_mapped_to_the_rung_openai_actually_has(self) -> None:
        # Measured 2026-09-30: `effort: "max"` is a 400 on `gpt-5.4-nano`. `max`
        # means "the most this model will do", so it is mapped, never sent -- even
        # though the configuration_update validator on gpt-5.6-* does list `max`,
        # because one meaning for the word across the surface is worth more than
        # one fewer line.
        items = wire_input([{"role": "user", "content": [update("max"), {"type": "text", "text": "hi"}]}])
        assert items[0]["reasoning"] == {"effort": "xhigh"}

    def test_none_and_minimal_pass_through(self) -> None:
        for effort in ("none", "minimal"):
            items = wire_input(
                [{"role": "user", "content": [update(effort), {"type": "text", "text": "x"}]}]
            )
            assert items[0]["reasoning"] == {"effort": effort}


    def test_it_does_not_warn_that_it_was_dropped(self) -> None:
        """The note said the opposite of what happened.

        `_input_content` notes every content kind it cannot represent, so that a
        part vanishing from a request is never silent. This part is represented --
        as a top-level item -- so the note fired on a request that carried it
        perfectly well, saying it had been "sent without it". A warning that is
        wrong teaches a reader to stop believing the ones that are right.
        """
        built = ADAPTER.build_request(
            {
                "model": "gpt-5.6-sol",
                "messages": [
                    {"role": "user", "content": [update("high"), {"type": "text", "text": "hi"}]}
                ],
            }
        )
        assert not [n for n in (built.notes or []) if "configuration_update" in n]
        # And the item really is on the request, so this is not a vacuous pass.
        assert configuration_items(list(built.body["input"]))


class TestTheTwoEffortTablesAgree:
    def test_it_matches_the_wire_spec_entry_for_entry(self) -> None:
        # The same mapping exists twice: here in Python for the item the adapter
        # builds, and as a `$table` in the spec for the top-level field the
        # interpreter builds. An effort meaning one thing on a request and another
        # on a stored update, in the same conversation, is a bug nobody would look
        # for -- so this is a mechanical guard rather than a note asking a future
        # edit to touch both.
        spec_path = (
            Path("src/combycode_llm_sdk/wire/specs/openai-responses.json").resolve()
        )
        table = json.loads(spec_path.read_text(encoding="utf-8"))["tables"]["reasoningEffort"]
        assert {k: v for k, v in table.items() if not k.startswith("_")} == OPENAI_REASONING_EFFORT


class TestTheUpdateOnTheWayBackIn:
    def test_it_becomes_a_typed_part_carrying_the_stored_id(self) -> None:
        parsed = ADAPTER.parse_response({"id": "resp_1", "output": [STORED], "usage": {}}, 1)
        part = next(p for p in parsed["content"] if p.get("type") == "configuration_update")
        assert part["reasoning"]["effort"] == "high"
        assert part["id"] == STORED["id"]

    def test_it_is_not_invented_from_an_item_with_no_effort(self) -> None:
        # Nothing useful to carry, and a part claiming an undefined effort would
        # be sent back as one.
        parsed = ADAPTER.parse_response(
            {"id": "resp_1", "output": [{"id": "cnfu_x", "type": "configuration_update"}], "usage": {}},
            1,
        )
        assert not [p for p in parsed["content"] if p.get("type") == "configuration_update"]

    def test_it_survives_a_round_trip(self) -> None:
        # Measured: this item does NOT appear in `response.output` -- four turns
        # that set one came back `output: [message]`, and it was found only
        # through the conversation-items endpoint. It is parsed anyway because
        # OpenAI's types put it in the output union and the cost of being wrong
        # runs one way: dropping it from history silently reverts the effort.
        parsed = ADAPTER.parse_response({"id": "resp_1", "output": [STORED], "usage": {}}, 1)
        items = wire_input([{"role": "assistant", "content": list(parsed["content"])}])
        assert configuration_items(items) == [
            {"type": "configuration_update", "reasoning": {"effort": "high"}}
        ]
