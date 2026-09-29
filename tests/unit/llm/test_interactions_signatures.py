"""Google Interactions: the signature a turn must hand back, and the reason a
failed one gives.

Transposed from
`unified-library-ts/tests/unit/llm/interactions-signatures.test.ts`.

Both measured live on 2026-09-29 against `gemini-3.1-flash-lite`:

- an ordinary turn returns a `thought` step carrying NOTHING but a `signature`,
  which this library used to drop: the buffered parse had no case for the step
  type, and the stream spec called the delta "internal";
- echoing that step on the next turn is accepted (200) and the model answers
  normally;
- echoing it with the signature corrupted is refused 400 "Corrupted thought
  signature", so the server READS it rather than tolerating it. That is what
  makes dropping it a defect and not a tidy-up.

In a stream the signature reaches the client only as its own `step.delta`
(`delta.type == "thought_signature"`): `step.start` announces the thought
without one, and the terminal `interaction.completed` carries the envelope with
no steps at all.
"""

from __future__ import annotations

import json
from typing import Any

from combycode_llm_sdk.llm.client_internal import build_assistant_message
from combycode_llm_sdk.llm.providers.google.interactions import GoogleInteractionsAdapter

KEY = {"apiKey": "test-key"}
SIGNATURE = "EnMKcQFpFH0TSvkjoKwOEghRRZdGvC3ICA0FZOoJaq4F2i"

ADAPTER = GoogleInteractionsAdapter(KEY)


def interaction(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "int_1",
        "object": "interaction",
        "status": "completed",
        "model": "gemini-3.1-flash-lite",
        "steps": [
            {"type": "thought", "signature": SIGNATURE},
            {"type": "model_output", "content": [{"type": "text", "text": "OK"}]},
        ],
        "usage": {"prompt_tokens": 3, "candidates_tokens": 1, "total_tokens": 4},
    }
    body.update(over)
    return body


class TestASignedStepSurvivesTheParse:
    def test_it_keeps_the_step_verbatim(self) -> None:
        # Verbatim, because verbatim is what goes back.
        res = ADAPTER.parse_response(interaction(), 1)
        assert res["signatures"] == [{"type": "thought", "signature": SIGNATURE}]

    def test_it_says_nothing_when_no_step_carried_one(self) -> None:
        res = ADAPTER.parse_response(
            interaction(steps=[{"type": "model_output", "content": [{"type": "text", "text": "OK"}]}]),
            1,
        )
        assert res.get("signatures") is None

    def test_it_keeps_a_signed_step_of_a_type_we_do_not_model(self) -> None:
        # Matched on the PRESENCE of a signature, not on a list of step types:
        # `processing_call`, `processing_result`, `retrieval_call` and
        # `retrieval_result` all declare one and are all accepted as input types
        # -- the live API enumerates them when it rejects an unknown one. A type
        # list would lose each new one silently until someone edited it.
        res = ADAPTER.parse_response(
            interaction(
                steps=[
                    {"type": "processing_call", "id": "p1", "signature": "sig-processing"},
                    {"type": "model_output", "content": [{"type": "text", "text": "OK"}]},
                ]
            ),
            1,
        )
        assert res["signatures"] == [
            {"type": "processing_call", "id": "p1", "signature": "sig-processing"}
        ]


class TestAndItReachesTheNextRequest:
    @staticmethod
    def _assistant(response: Any) -> dict[str, Any]:
        return build_assistant_message(
            response,
            {"provider": "google", "model": "gemini-3.1-flash-lite", "api": "interactions"},
        )

    @staticmethod
    def _input(messages: list[Any]) -> list[dict[str, Any]]:
        body = ADAPTER.build_request(
            {"model": "gemini-3.1-flash-lite", "messages": messages}
        ).body
        assert isinstance(body, dict)
        return list(body["input"])

    def test_it_is_stamped_onto_the_assistant_message(self) -> None:
        msg = self._assistant(ADAPTER.parse_response(interaction(), 1))
        assert msg["origin"]["signatures"] == [{"type": "thought", "signature": SIGNATURE}]

    def test_it_goes_back_out_before_the_model_output_it_preceded(self) -> None:
        msg = self._assistant(ADAPTER.parse_response(interaction(), 1))
        sent = self._input(
            [{"role": "user", "content": "hi"}, msg, {"role": "user", "content": "again"}]
        )
        assert [i["type"] for i in sent] == [
            "user_input",
            "thought",
            "model_output",
            "user_input",
        ]
        assert sent[1] == {"type": "thought", "signature": SIGNATURE}

    def test_it_never_sends_another_providers_signature(self) -> None:
        # `origin.signatures` is provider-bound by contract: a blob minted
        # elsewhere is meaningless here and a 400 at best.
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "OK"}],
            "origin": {
                "provider": "openai",
                "model": "gpt-5.4-nano",
                "signatures": [{"type": "thought", "signature": "not-ours"}],
            },
        }
        sent = self._input([{"role": "user", "content": "hi"}, msg])
        assert [i["type"] for i in sent] == ["user_input", "model_output"]

    def test_it_sends_nothing_extra_when_the_turn_carried_no_signature(self) -> None:
        msg = self._assistant(
            ADAPTER.parse_response(
                interaction(
                    steps=[{"type": "model_output", "content": [{"type": "text", "text": "OK"}]}]
                ),
                1,
            )
        )
        sent = self._input([{"role": "user", "content": "hi"}, msg])
        assert [i["type"] for i in sent] == ["user_input", "model_output"]


class TestAStreamedTurnKeepsItToo:
    @staticmethod
    def _event(payload: dict[str, Any]) -> dict[str, Any]:
        return {"event": payload.get("event_type"), "data": json.dumps(payload)}

    def test_it_rebuilds_the_step_and_carries_it_on_done(self) -> None:
        feed = ADAPTER.create_stream_parser()
        feed(self._event({"event_type": "step.start", "step": {"type": "thought"}}))
        feed(
            self._event(
                {
                    "event_type": "step.delta",
                    "delta": {"type": "thought_signature", "signature": SIGNATURE},
                }
            )
        )
        feed(self._event({"event_type": "step.stop"}))
        events = feed(
            self._event(
                {
                    "event_type": "interaction.completed",
                    "interaction": {"id": "int_1", "status": "completed"},
                }
            )
        )
        done = [e for e in events if e.get("type") == "done"]
        assert done and done[0]["signatures"] == [{"type": "thought", "signature": SIGNATURE}]

    def test_it_leaves_done_alone_when_nothing_was_signed(self) -> None:
        feed = ADAPTER.create_stream_parser()
        feed(self._event({"event_type": "step.start", "step": {"type": "model_output"}}))
        events = feed(
            self._event(
                {
                    "event_type": "interaction.completed",
                    "interaction": {"id": "int_1", "status": "completed"},
                }
            )
        )
        done = [e for e in events if e.get("type") == "done"]
        assert done and "signatures" not in done[0]


class TestAFailedInteractionSaysWhy:
    def test_it_lifts_errors_onto_the_response_error(self) -> None:
        # Before this, a failed interaction arrived as `finishReason: "error"`
        # and nothing else: an empty answer, no exception to catch, and no way to
        # tell a content refusal from a platform fault.
        res = ADAPTER.parse_response(
            interaction(
                status="failed",
                steps=[],
                errors=[
                    {"code": "https://developers.google.com/errors/internal", "message": "boom"}
                ],
            ),
            1,
        )
        assert res["finishReason"] == "error"
        assert res["error"] == {
            "code": "https://developers.google.com/errors/internal",
            "message": "boom",
        }

    def test_it_joins_every_recorded_message(self) -> None:
        res = ADAPTER.parse_response(
            interaction(status="failed", steps=[], errors=[{"message": "first"}, {"message": "second"}]),
            1,
        )
        assert res["error"]["message"] == "first; second"

    def test_it_still_says_something_when_no_detail_was_recorded(self) -> None:
        res = ADAPTER.parse_response(interaction(status="failed", steps=[]), 1)
        assert "failed" in res["error"]["message"]

    def test_a_completed_interaction_carries_no_error(self) -> None:
        # `errors[]` is documented as diagnostics, not as the cause. Putting it
        # on `error` for a completed turn would report a successful call as
        # failed; it stays reachable on `response["raw"]`.
        res = ADAPTER.parse_response(interaction(errors=[{"message": "a diagnostic"}]), 1)
        assert res.get("error") is None
        assert res["finishReason"] == "stop"
