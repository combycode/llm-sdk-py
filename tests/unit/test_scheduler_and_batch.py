"""Deferred work and batched work, on hand-wound clocks.

Nothing here sleeps, and that is the point of both designs: neither class owns a
thread, so both can be driven by a clock a test moves by hand. A scheduler that
fired on a real timer could only be tested by waiting, and a test that waits is a
test that is either slow or flaky.

The batch half is really one assertion wearing several hats: results are matched
by ID. The stub answers in REVERSE on purpose, because providers do -- Anthropic
returns a two-item batch back to front -- and a batcher that paired by position
would hand each caller the other's answer with both looking perfectly correct.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from combycode_llm_sdk import (
    AutoBatcher,
    BatchStrategy,
    LLMError,
    MemoryPersistence,
    Scheduler,
    TransportResponse,
)
from combycode_llm_sdk.batch import (
    COMPLETED,
    EXPIRED,
    PROCESSING,
    AnthropicBatchAdapter,
    BatchRequest,
)
from combycode_llm_sdk.helpers.batch import BatchJob, batch, submit_batch
from combycode_llm_sdk.scheduler import ScheduledTask, parse_duration

MODEL = "anthropic/claude-haiku-4.5"
BATCH_ID = "msgbatch_1"
BATCH_PATH = "/v1/messages/batches"


class Clock:
    """A clock moved by hand."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, amount: float) -> None:
        self.now += amount


# -- the scheduler -----------------------------------------------------------


class TestDurations:
    def test_every_unit_lands_in_milliseconds(self) -> None:
        # Milliseconds because that is what the store holds. A scheduler whose
        # durations and clock disagreed about units would be wrong by a factor
        # of a thousand and still look plausible in any test using only one.
        assert parse_duration("500ms") == 500
        assert parse_duration("30s") == 30_000
        assert parse_duration("5m") == 300_000
        assert parse_duration("1h") == 3_600_000
        assert parse_duration("2d") == 172_800_000

    def test_a_fraction_is_allowed(self) -> None:
        assert parse_duration("1.5h") == 5_400_000

    def test_a_bare_number_is_already_milliseconds(self) -> None:
        assert parse_duration(250) == 250
        assert parse_duration("250") == 250

    def test_nonsense_is_refused_with_the_forms_that_work(self) -> None:
        with pytest.raises(ValueError, match="30s, 5m, 1h, 2d"):
            parse_duration("soon")

    def test_an_unknown_unit_is_refused(self) -> None:
        with pytest.raises(ValueError):
            parse_duration("5 fortnights")

    def test_a_bool_is_not_a_duration(self) -> None:
        # `isinstance(True, int)` is True, so a numeric fast path would have
        # quietly read `True` as one millisecond.
        with pytest.raises(ValueError):
            parse_duration(True)


class TestScheduling:
    def scheduler(self) -> tuple[Scheduler, Clock, MemoryPersistence, list[Any]]:
        clock = Clock(now=1_700_000_000_000.0)
        store = MemoryPersistence()
        scheduler = Scheduler(store, clock=clock)
        fired: list[Any] = []
        scheduler.register("report", fired.append)
        return scheduler, clock, store, fired

    def test_nothing_is_due_before_its_time(self) -> None:
        scheduler, _, _, _ = self.scheduler()
        scheduler.after("5m", "report", {"to": "ops"})
        assert scheduler.due() == []
        assert len(scheduler.pending()) == 1

    def test_asking_is_what_makes_it_due(self) -> None:
        scheduler, clock, _, _ = self.scheduler()
        scheduler.after("5m", "report", {"to": "ops"})
        clock.advance(parse_duration("5m"))
        assert [t.name for t in scheduler.due()] == ["report"]

    def test_reading_due_does_not_run_anything(self) -> None:
        # `due()` is a question and `run_due()` is the answer; a question with a
        # side effect cannot be asked twice.
        scheduler, clock, _, fired = self.scheduler()
        scheduler.after("5m", "report", {"to": "ops"})
        clock.advance(parse_duration("5m"))
        scheduler.due()
        scheduler.due()
        assert fired == []
        assert scheduler.run_due() == 1
        assert fired == [{"to": "ops"}]

    def test_a_one_shot_task_is_gone_once_it_has_run(self) -> None:
        scheduler, clock, _, _ = self.scheduler()
        scheduler.after("5m", "report", {})
        clock.advance(parse_duration("5m"))
        scheduler.run_due()
        assert scheduler.pending() == []
        assert scheduler.run_due() == 0

    def test_a_periodic_task_comes_back(self) -> None:
        scheduler, clock, _, fired = self.scheduler()
        scheduler.every("1h", "report", {"to": "ops"})
        clock.advance(parse_duration("1h"))
        assert scheduler.run_due() == 1
        assert len(scheduler.pending()) == 1, "a periodic task reschedules itself"
        clock.advance(parse_duration("1h"))
        assert scheduler.run_due() == 1
        assert len(fired) == 2

    def test_at_takes_a_timestamp_on_the_clocks_own_scale(self) -> None:
        scheduler, clock, _, _ = self.scheduler()
        scheduler.at(clock.now + 10, "report", {})
        assert scheduler.due() == []
        clock.advance(10)
        assert scheduler.run_due() == 1

    def test_a_cancelled_task_never_fires(self) -> None:
        scheduler, clock, _, fired = self.scheduler()
        task_id = scheduler.after("5m", "report", {})
        scheduler.cancel(task_id)
        clock.advance(parse_duration("1h"))
        assert scheduler.run_due() == 0
        assert fired == []

    def test_pending_is_ordered_by_when_it_fires(self) -> None:
        scheduler, _, _, _ = self.scheduler()
        scheduler.after("1h", "report", {"n": 2})
        scheduler.after("5m", "report", {"n": 1})
        assert [t.args["n"] for t in scheduler.pending()] == [1, 2]

    def test_the_schedule_outlives_the_object_that_made_it(self) -> None:
        # The whole reason it lives in a store: a process that dies with an hour
        # left on a task picks it up on the next start.
        scheduler, clock, store, _ = self.scheduler()
        scheduler.after("1h", "report", {"to": "finance"})

        restarted = Scheduler(store, clock=clock)
        resumed: list[Any] = []
        restarted.register("report", resumed.append)
        clock.advance(parse_duration("1h"))
        assert restarted.run_due() == 1
        assert resumed == [{"to": "finance"}]

    def test_a_task_nobody_registered_is_left_for_a_process_that_did(self) -> None:
        # No handler HERE is not the same as no handler anywhere: this process
        # may simply not be the one that runs this task.
        scheduler, clock, _, _ = self.scheduler()
        scheduler.after("5m", "unknown_job", {})
        clock.advance(parse_duration("5m"))
        assert scheduler.run_due() == 0
        assert len(scheduler.pending()) == 1

    def test_one_failing_handler_does_not_stop_the_rest(self) -> None:
        # The tasks are unrelated. Letting the exception out mid-loop would
        # delay every later job by one tick, for as long as the bad one kept
        # failing.
        clock = Clock(now=1000.0)
        scheduler = Scheduler(MemoryPersistence(), clock=clock)
        ran: list[str] = []

        def bad(args: Any) -> None:
            raise RuntimeError("no")

        scheduler.register("bad", bad)
        scheduler.register("good", lambda args: ran.append("good"))
        scheduler.after(0, "bad", {})
        scheduler.after(0, "good", {})
        with pytest.raises(ExceptionGroup):
            scheduler.run_due()
        assert ran == ["good"], "the good task ran in the SAME pass"

    def test_every_failure_is_reported_not_just_the_first(self) -> None:
        # Swallowing them is worse than raising them, and raising only the first
        # hides however many others there were.
        clock = Clock(now=1000.0)
        scheduler = Scheduler(MemoryPersistence(), clock=clock)

        def bad(args: Any) -> None:
            raise RuntimeError(f"failed {args['n']}")

        scheduler.register("bad", bad)
        scheduler.after(0, "bad", {"n": 1})
        scheduler.after(0, "bad", {"n": 2})
        with pytest.raises(ExceptionGroup) as caught:
            scheduler.run_due()
        assert len(caught.value.exceptions) == 2

    def test_a_failed_task_is_retired_rather_than_re_firing_instantly(self) -> None:
        # A periodic task that re-fires the moment it fails is a hot loop.
        clock = Clock(now=1000.0)
        scheduler = Scheduler(MemoryPersistence(), clock=clock)

        def bad(args: Any) -> None:
            raise RuntimeError("no")

        scheduler.register("bad", bad)
        scheduler.after(0, "bad", {})
        with pytest.raises(ExceptionGroup):
            scheduler.run_due()
        assert scheduler.pending() == []

    def test_a_task_round_trips_through_the_store_unchanged(self) -> None:
        task = ScheduledTask(
            id="task_1", name="report", args={"to": "ops"}, fire_at=5.0, type="periodic",
            interval=10.0,
        )
        assert ScheduledTask.of(task.as_row()) == task

    def test_the_store_may_hold_more_than_the_schedule(self) -> None:
        # Which is why the keys are prefixed: a shared store is the common case.
        store = MemoryPersistence()
        store.set("checkpoint:1", {"unrelated": True})
        scheduler = Scheduler(store, clock=Clock(now=0.0))
        scheduler.after(0, "report", {})
        assert len(scheduler.pending()) == 1


# -- the batcher -------------------------------------------------------------


class Provider:
    """The Anthropic batch API, answering results in REVERSE."""

    def __init__(self, *, status: str = "ended", fail: set[str] | None = None) -> None:
        self.replies: dict[str, str] = {}
        self.submissions: list[Any] = []
        self.polls = 0
        self._status = status
        self._fail = fail or set()

    def __call__(self, request: Any) -> TransportResponse:
        url = request["url"] if isinstance(request, dict) else request.url
        if url.endswith(f"{BATCH_PATH}/{BATCH_ID}/results"):
            return TransportResponse(status=200, body=self._results())
        if url.endswith(f"{BATCH_PATH}/{BATCH_ID}"):
            self.polls += 1
            return TransportResponse(status=200, body={"processing_status": self._status})
        if url.endswith(BATCH_PATH):
            body = request["body"] if isinstance(request, dict) else request.body
            self.submissions.append(body)
            return TransportResponse(status=200, body={"id": BATCH_ID})
        raise AssertionError(f"unexpected batch url: {url}")

    def _results(self) -> str:
        lines = []
        for custom_id, text in reversed(list(self.replies.items())):
            if custom_id in self._fail:
                result: dict[str, Any] = {"type": "errored", "error": {"message": text}}
            else:
                result = {"type": "succeeded", "message": self._message(text)}
            lines.append(json.dumps({"custom_id": custom_id, "result": result}))
        return "\n".join(lines) + "\n"

    @staticmethod
    def _message(text: str) -> dict[str, Any]:
        return {
            "id": "msg_1",
            "model": "claude-haiku-4-5",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }


def batcher_for(provider: Provider, clock: Clock, **kwargs: Any) -> AutoBatcher:
    strategy = kwargs.pop("strategy", BatchStrategy(window_seconds=10.0, min_batch_size=2))
    return AutoBatcher(
        model=MODEL,
        strategy=strategy,
        clock=clock,
        api_key="k",
        transport=provider,
        max_tokens=16,
        **kwargs,
    )


class TestCollecting:
    def test_one_item_is_not_a_batch(self) -> None:
        clock = Clock()
        provider = Provider()
        batcher = batcher_for(provider, clock)
        batcher.add({"prompt": "one"})
        clock.advance(60)
        assert batcher.tick() is None
        assert provider.submissions == []

    def test_the_window_holds_the_batch_open(self) -> None:
        clock = Clock()
        provider = Provider()
        batcher = batcher_for(provider, clock)
        batcher.add({"prompt": "one"})
        batcher.add({"prompt": "two"})
        assert batcher.pending == 2
        assert batcher.tick() is None, "the window is still open"
        assert provider.submissions == []

    def test_past_the_window_the_next_touch_submits(self) -> None:
        clock = Clock()
        provider = Provider()
        batcher = batcher_for(provider, clock)
        batcher.add({"prompt": "one"})
        batcher.add({"prompt": "two"})
        clock.advance(11)
        assert batcher.tick() == BATCH_ID
        assert len(provider.submissions) == 1
        assert len(provider.submissions[0]["requests"]) == 2

    def test_a_full_batch_goes_without_waiting(self) -> None:
        # Waiting out a window that cannot grow the batch only adds latency.
        clock = Clock()
        provider = Provider()
        batcher = batcher_for(
            provider, clock, strategy=BatchStrategy(window_seconds=999.0, max_batch_size=2)
        )
        batcher.add({"prompt": "one"})
        batcher.add({"prompt": "two"})
        assert provider.submissions, "a full batch should submit on the spot"
        assert batcher.pending == 0

    def test_every_item_carries_an_id_of_its_own(self) -> None:
        clock = Clock()
        batcher = batcher_for(Provider(), clock)
        first = batcher.add({"prompt": "one"})
        second = batcher.add({"prompt": "two"})
        assert first.custom_id != second.custom_id

    def test_the_body_is_the_one_a_lone_call_would_have_sent(self) -> None:
        clock = Clock()
        provider = Provider()
        batcher = batcher_for(provider, clock)
        batcher.add({"prompt": "hello"})
        batcher.add({"prompt": "again", "max_tokens": 99})
        clock.advance(11)
        batcher.tick()
        requests = provider.submissions[0]["requests"]
        first, second = requests[0]["params"], requests[1]["params"]
        assert first["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        assert first["max_tokens"] == 16
        assert second["max_tokens"] == 99, "a per-item value overrides the batcher's"

    def test_messages_can_be_given_in_full(self) -> None:
        clock = Clock()
        provider = Provider()
        batcher = batcher_for(provider, clock)
        batcher.add({"messages": [{"role": "user", "content": "direct"}]})
        batcher.add({"prompt": "two"})
        clock.advance(11)
        batcher.tick()
        params = provider.submissions[0]["requests"][0]["params"]
        assert params["messages"][0]["content"][0]["text"] == "direct"

    def test_an_item_with_neither_is_refused(self) -> None:
        clock = Clock()
        batcher = batcher_for(Provider(), clock)
        batcher.add({"nonsense": 1})
        batcher.add({"prompt": "two"})
        clock.advance(11)
        with pytest.raises(ValueError, match="needs a 'prompt' or 'messages'"):
            batcher.tick()


class TestSettling:
    def test_each_caller_gets_their_own_answer(self) -> None:
        # The provider answers in REVERSE. Correlating by position would hand
        # each caller the other's answer, and both would look correct.
        clock = Clock()
        provider = Provider()
        batcher = batcher_for(provider, clock)
        first = batcher.add({"prompt": "ticket 4711"})
        second = batcher.add({"prompt": "ticket 815"})
        provider.replies[first.custom_id] = "4711: the printer is on fire"
        provider.replies[second.custom_id] = "815: the printer is fine"
        clock.advance(11)
        batcher.tick()
        batcher.wait(poll_seconds=0, timeout_seconds=5)
        assert first.result().text.startswith("4711")
        assert second.result().text.startswith("815")

    def test_a_ticket_with_no_answer_refuses_rather_than_inventing_one(self) -> None:
        # An empty completion here reads as a model that declined to answer,
        # which is a different and much quieter kind of wrong.
        clock = Clock()
        batcher = batcher_for(Provider(), clock)
        ticket = batcher.add({"prompt": "one"})
        with pytest.raises(LLMError, match="no answer yet"):
            ticket.result()

    def test_a_failed_item_reports_its_own_failure(self) -> None:
        clock = Clock()
        provider = Provider()
        batcher = batcher_for(provider, clock)
        first = batcher.add({"prompt": "one"})
        second = batcher.add({"prompt": "two"})
        provider.replies[first.custom_id] = "over quota"
        provider.replies[second.custom_id] = "815: fine"
        provider._fail = {first.custom_id}
        clock.advance(11)
        batcher.wait(poll_seconds=0, timeout_seconds=5)
        with pytest.raises(LLMError, match="over quota"):
            first.result()
        # And the other caller is unaffected, which is the point of per-item ids.
        assert second.result().text.startswith("815")

    def test_usage_survives_the_batch(self) -> None:
        clock = Clock()
        provider = Provider()
        batcher = batcher_for(provider, clock)
        ticket = batcher.add({"prompt": "one"})
        batcher.add({"prompt": "two"})
        provider.replies[ticket.custom_id] = "an answer"
        clock.advance(11)
        batcher.wait(poll_seconds=0, timeout_seconds=5)
        assert ticket.result().usage.input_tokens == 10

    def test_an_id_we_never_sent_is_dropped_not_guessed_at(self) -> None:
        clock = Clock()
        provider = Provider()
        batcher = batcher_for(provider, clock)
        ticket = batcher.add({"prompt": "one"})
        batcher.add({"prompt": "two"})
        provider.replies["req_never_sent"] = "somebody else's answer"
        provider.replies[ticket.custom_id] = "mine"
        clock.advance(11)
        settled = batcher.wait(poll_seconds=0, timeout_seconds=5)
        assert settled == 1
        assert ticket.result().text == "mine"

    def test_wait_submits_what_is_still_collecting(self) -> None:
        clock = Clock()
        provider = Provider()
        batcher = batcher_for(provider, clock)
        ticket = batcher.add({"prompt": "one"})
        provider.replies[ticket.custom_id] = "answered anyway"
        batcher.wait(poll_seconds=0, timeout_seconds=5)
        assert provider.submissions, "wait() must not sit on an unsent batch"
        assert ticket.result().text == "answered anyway"

    def test_a_job_that_never_finishes_times_out_naming_its_state(self) -> None:
        clock = Clock()
        provider = Provider(status="in_progress")
        batcher = batcher_for(provider, clock)
        batcher.add({"prompt": "one"})
        batcher.add({"prompt": "two"})
        clock.advance(11)
        with pytest.raises(LLMError, match="did not finish"):
            batcher.wait(poll_seconds=0, timeout_seconds=0.05)
        assert provider.polls > 0

    def test_reading_before_anything_is_submitted_is_refused(self) -> None:
        batcher = batcher_for(Provider(), Clock())
        with pytest.raises(LLMError, match="nothing has been submitted"):
            batcher.status()
        with pytest.raises(LLMError, match="nothing to submit"):
            batcher.submit()


class TestTheAdapter:
    def test_the_submit_body_carries_every_item_inline(self) -> None:
        adapter = AnthropicBatchAdapter("k")
        request = adapter.submit_request(
            [BatchRequest("a", {"model": "m"}), BatchRequest("b", {"model": "m"})]
        )
        assert request["url"].endswith(BATCH_PATH)
        assert [r["custom_id"] for r in request["body"]["requests"]] == ["a", "b"]
        assert request["headers"]["x-api-key"] == "k"

    def test_a_results_read_asks_for_text(self) -> None:
        # JSONL, not JSON. Asking for `json` breaks every batch read and no type
        # catches it -- the TypeScript records the same trap.
        provider = Provider()
        adapter = AnthropicBatchAdapter("k")
        seen: list[Any] = []

        def spy(request: Any) -> Any:
            seen.append(request)
            return {"status": 200, "body": provider._results()}

        adapter.get_results(BATCH_ID, spy)
        assert seen[0]["responseType"] == "text"

    def test_a_status_request_carries_no_body(self) -> None:
        # `bodyKind: none` means absent, and `"body": None` on a GET is not the
        # same document.
        seen: list[Any] = []

        def spy(request: Any) -> Any:
            seen.append(request)
            return {"status": 200, "body": {"processing_status": "ended"}}

        AnthropicBatchAdapter("k").get_status(BATCH_ID, spy)
        assert "body" not in seen[0]
        assert seen[0]["method"] == "GET"

    def test_provider_states_are_normalised(self) -> None:
        def answering(status: str) -> Any:
            return lambda request: {
                "status": 200,
                "body": {
                    "processing_status": status,
                    "request_counts": {"succeeded": 2, "errored": 1, "processing": 3},
                },
            }

        adapter = AnthropicBatchAdapter("k")
        assert adapter.get_status(BATCH_ID, answering("ended")).status == COMPLETED
        assert adapter.get_status(BATCH_ID, answering("in_progress")).status == PROCESSING
        assert adapter.get_status(BATCH_ID, answering("expired")).status == EXPIRED
        counted = adapter.get_status(BATCH_ID, answering("ended"))
        assert (counted.total, counted.completed, counted.failed) == (6, 2, 1)

    def test_a_finished_status_says_so(self) -> None:
        adapter = AnthropicBatchAdapter("k")
        state = adapter.get_status(
            BATCH_ID, lambda r: {"status": 200, "body": {"processing_status": "ended"}}
        )
        assert state.finished is True

    def test_a_failed_submit_names_the_status(self) -> None:
        adapter = AnthropicBatchAdapter("k")
        with pytest.raises(RuntimeError, match="429"):
            adapter.submit(
                [BatchRequest("a", {})],
                lambda r: {"status": 429, "body": {"error": "slow down"}},
            )

    def test_a_submit_with_no_id_is_refused(self) -> None:
        # Returning None here would make every later poll ask about "None".
        adapter = AnthropicBatchAdapter("k")
        with pytest.raises(RuntimeError, match="no id"):
            adapter.submit([BatchRequest("a", {})], lambda r: {"status": 200, "body": {}})


class TestConfiguration:
    def test_a_bare_model_without_a_provider_is_refused(self) -> None:
        with pytest.raises(ValueError, match="name the provider"):
            AutoBatcher(model="claude-haiku-4.5", api_key="k")

    def test_a_provider_with_no_batch_adapter_says_which_have_one(self) -> None:
        # OpenRouter fronts other providers' models but hosts no batch API of
        # its own, so this is a fact about the provider rather than a gap here.
        with pytest.raises(ValueError, match="hosts no batch API"):
            AutoBatcher(model="openrouter/openai/gpt-4.1", api_key="k")

    def test_every_provider_with_one_can_be_batched(self) -> None:
        for provider, model in (
            ("anthropic", "claude-haiku-4.5"),
            ("openai", "gpt-5.4-nano"),
            ("google", "gemini-3.1-flash-lite"),
            ("xai", "grok-4.3"),
        ):
            batcher = AutoBatcher(model=f"{provider}/{model}", api_key="k")
            assert batcher.provider == provider

    def test_google_is_built_with_the_model_its_endpoint_names(self) -> None:
        # Its batch URL is model-scoped, so an adapter built without one
        # would send every batch to `/models/None:batchGenerateContent`.
        batcher = AutoBatcher(model="google/gemini-3.1-flash-lite", api_key="k")
        url = batcher._adapter.submit_request([]).get("url")
        assert batcher.model in url

    def test_no_key_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no API key"):
            AutoBatcher(model=MODEL)

    def test_the_provider_id_is_sent_not_our_alias(self) -> None:
        # Found live: a batch submitted under the unified alias fails EVERY item
        # with `not_found_error`, ninety seconds after submission -- which is the
        # worst possible place to discover a model name. Every other entry point
        # resolves through the catalog; this one did not.
        clock = Clock()
        provider = Provider()
        batcher = batcher_for(provider, clock)
        assert batcher.model == "claude-haiku-4-5-20251001"
        batcher.add({"prompt": "one"})
        batcher.add({"prompt": "two"})
        clock.advance(11)
        batcher.tick()
        params = provider.submissions[0]["requests"][0]["params"]
        assert params["model"] == "claude-haiku-4-5-20251001"

    def test_a_model_id_the_catalog_does_not_know_passes_through(self) -> None:
        # An unknown model is not necessarily a wrong one: a preview id the
        # catalog has not caught up with must still be callable.
        batcher = AutoBatcher(model="anthropic/some-unreleased-model", api_key="k")
        assert batcher.model == "some-unreleased-model"

    def test_a_submitted_job_is_recorded_in_the_store(self) -> None:
        # So a process that dies mid-batch can find the job again.
        clock = Clock()
        store = MemoryPersistence()
        provider = Provider()
        batcher = batcher_for(provider, clock, persistence=store)
        batcher.add({"prompt": "one"})
        batcher.add({"prompt": "two"})
        clock.advance(11)
        batcher.tick()
        assert store.get(f"batch:{BATCH_ID}")["batchId"] == BATCH_ID
        batcher.wait(poll_seconds=0, timeout_seconds=5)
        assert store.get(f"batch:{BATCH_ID}") is None, "a finished job is not still pending"


class TestTheOneShotForms:
    """`batch()` and `submit_batch()` -- for a caller who has the whole list.

    A different question from `AutoBatcher`, which exists for callers who arrive
    one at a time and do not know about each other.
    """

    def job_for(self, provider: Provider, **kwargs: Any) -> BatchJob:
        return submit_batch(
            model=MODEL,
            api_key="k",
            transport=provider,
            max_tokens=16,
            requests=[
                {"custom_id": "a", "prompt": "Reply with exactly: A"},
                {"custom_id": "b", "prompt": "Reply with exactly: B"},
            ],
            **kwargs,
        )

    def test_the_caller_s_own_ids_are_used(self) -> None:
        provider = Provider()
        job = self.job_for(provider)
        sent = [r["custom_id"] for r in provider.submissions[0]["requests"]]
        assert sent == ["a", "b"]
        assert job.id == BATCH_ID

    def test_answers_come_back_in_request_order(self) -> None:
        # The provider answers in REVERSE. The IDs settle the tickets; this list
        # then re-imposes the order the caller asked in -- two different jobs,
        # and a caller zipping results against their input needs both.
        provider = Provider()
        provider.replies["a"] = "A"
        provider.replies["b"] = "B"
        results = self.job_for(provider).wait(poll_seconds=0, timeout_seconds=5)
        assert [r.custom_id for r in results] == ["a", "b"]
        assert [r.text for r in results] == ["A", "B"]

    def test_an_item_with_no_id_is_given_one_that_means_something(self) -> None:
        provider = Provider()
        submit_batch(
            model=MODEL, api_key="k", transport=provider,
            requests=[{"prompt": "one"}, {"prompt": "two"}],
        )
        assert [r["custom_id"] for r in provider.submissions[0]["requests"]] == [
            "req-0", "req-1",
        ]

    def test_a_failed_item_is_a_row_not_an_exception(self) -> None:
        # One bad item must not cost the caller the other nine hundred answers.
        provider = Provider(fail={"a"})
        provider.replies["a"] = "over quota"
        provider.replies["b"] = "B"
        results = self.job_for(provider).wait(poll_seconds=0, timeout_seconds=5)
        assert [r.success for r in results] == [False, True]
        assert results[0].text == "", "a failed row reads as empty, never as None"
        assert "over quota" in (results[0].error or "")

    def test_an_item_the_provider_never_answered_is_reported(self) -> None:
        # Dropped rows would hand back a SHORTER list than was asked for, which
        # a caller zipping against their input reads as the wrong answers.
        provider = Provider()
        provider.replies["a"] = "A"
        results = self.job_for(provider).wait(poll_seconds=0, timeout_seconds=5)
        assert len(results) == 2
        assert results[1].success is False
        assert "no result" in (results[1].error or "")

    def test_results_refuses_while_the_job_is_running(self) -> None:
        # "Not finished" and "finished with nothing" are different answers.
        provider = Provider(status="in_progress")
        with pytest.raises(LLMError, match="is not finished"):
            self.job_for(provider).results()

    def test_results_answers_once_the_job_has_ended(self) -> None:
        provider = Provider()
        provider.replies["a"] = "A"
        provider.replies["b"] = "B"
        job = self.job_for(provider)
        assert [r.text for r in job.results()] == ["A", "B"]

    def test_the_blocking_form_submits_and_waits(self) -> None:
        provider = Provider()
        provider.replies["a"] = "A"
        provider.replies["b"] = "B"
        results = batch(
            model=MODEL,
            api_key="k",
            transport=provider,
            max_tokens=16,
            requests=[
                {"custom_id": "a", "prompt": "one"},
                {"custom_id": "b", "prompt": "two"},
            ],
            poll_seconds=0,
            timeout_seconds=5,
        )
        assert [r.text for r in results] == ["A", "B"]

    def test_progress_is_reported_while_waiting(self) -> None:
        provider = Provider()
        provider.replies["a"] = "A"
        provider.replies["b"] = "B"
        seen: list[Any] = []
        self.job_for(provider).wait(
            poll_seconds=0, timeout_seconds=5, on_progress=seen.append
        )
        assert seen and seen[0].status == COMPLETED

    def test_an_empty_request_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match="nothing to submit"):
            submit_batch(model=MODEL, api_key="k", transport=Provider(), requests=[])

    def test_a_duplicate_id_is_refused_rather_than_overwriting(self) -> None:
        # The second ticket would replace the first, leaving one caller holding
        # a ticket that can never settle.
        with pytest.raises(ValueError, match="already in this batch"):
            submit_batch(
                model=MODEL, api_key="k", transport=Provider(),
                requests=[{"custom_id": "a", "prompt": "one"},
                          {"custom_id": "a", "prompt": "two"}],
            )

    def test_the_job_names_its_provider_and_resolved_model(self) -> None:
        job = self.job_for(Provider())
        assert job.provider == "anthropic"
        assert job.model == "claude-haiku-4-5-20251001"


class TestTheStrategy:
    def test_a_window_that_has_not_closed_holds(self) -> None:
        strategy = BatchStrategy(window_seconds=10.0, min_batch_size=2)
        assert strategy.should_submit(2, waited=9.0) is False
        assert strategy.should_submit(2, waited=10.0) is True

    def test_a_batch_under_the_minimum_never_goes(self) -> None:
        strategy = BatchStrategy(window_seconds=10.0, min_batch_size=3)
        assert strategy.should_submit(2, waited=999.0) is False

    def test_a_full_batch_goes_whatever_the_window_says(self) -> None:
        strategy = BatchStrategy(window_seconds=999.0, max_batch_size=2)
        assert strategy.should_submit(2, waited=0.0) is True

    def test_an_empty_collection_never_goes(self) -> None:
        assert BatchStrategy().should_submit(0, waited=999.0) is False

    def test_an_empty_collection_never_goes_even_with_no_minimum(self) -> None:
        # `min_batch_size=0` means "submit whatever you have", and an empty
        # batch is not that -- it is a request to the provider for nothing.
        strategy = BatchStrategy(window_seconds=1.0, min_batch_size=0)
        assert strategy.should_submit(0, waited=999.0) is False
        assert strategy.should_submit(1, waited=999.0) is True

    def test_a_bigger_batch_is_polled_later(self) -> None:
        strategy = BatchStrategy(first_poll_seconds=5.0)
        assert strategy.first_poll_after(1) < strategy.first_poll_after(50)
