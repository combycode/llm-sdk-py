"""Four batch APIs, and how little they agree about anything.

The outbound half is a differential against `tests/fixtures/service-golden.json`
-- nineteen requests frozen from the TypeScript adapters, driven through the
same public methods with the same fake answers, so the two-call flows (OpenAI's
upload-then-create, xAI's create-then-add, OpenAI's status-then-file) are
compared in order rather than one call at a time.

The inbound half is separate and just as important: every provider reports
progress in its own vocabulary, and normalising those is the reason a caller can
write one batching loop. A status map that reads `cancelling` as cancelled, or a
count that treats a running job's partial list as a finished tally, produces a
batcher that stops waiting too early -- which looks like a provider losing
results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from combycode_llm_sdk.batch.adapters import (
    BATCH_ADAPTERS,
    AnthropicBatchAdapter,
    GoogleBatchAdapter,
    OpenAIBatchAdapter,
    XaiBatchAdapter,
)
from combycode_llm_sdk.batch.types import (
    CANCELLED,
    COMPLETED,
    EXPIRED,
    FAILED,
    PENDING,
    PROCESSING,
    BatchRequest,
)

GOLDEN = json.loads(
    (Path(__file__).parent.parent / "fixtures" / "service-golden.json").read_text("utf-8")
)["index"]

#: Exactly the two requests the freeze script used.
REQUESTS = [
    BatchRequest(custom_id="r1", body={"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
    BatchRequest(custom_id="r2", body={"model": "m", "messages": [{"role": "user", "content": "yo"}]}),
]
BATCH_ID = "batch_abc"

#: Enough of an answer for each step to reach the next one: an id for submit, a
#: file id for OpenAI's two-step results, and empty payloads for the rest.
BATCH_RESPONSE = {
    "id": BATCH_ID,
    "name": BATCH_ID,
    "batch": {"name": BATCH_ID},
    "output_file_id": "file_out",
    "results_url": "https://x/results",
    "request_counts": {},
    "metadata": {},
    "data": [],
}

CASES = [
    ("anthropic", "submit"),
    ("anthropic", "get_status"),
    ("anthropic", "get_results"),
    ("anthropic", "cancel"),
    ("openai", "submit"),
    ("openai", "get_status"),
    ("openai", "get_results"),
    ("openai", "cancel"),
    ("google", "submit"),
    ("google", "get_status"),
    ("google", "get_results"),
    ("google", "cancel"),
    ("xai", "submit"),
    ("xai", "get_status"),
    ("xai", "get_results"),
    ("xai", "cancel"),
]

#: `get_status` here is `getStatus` in the frozen keys.
GOLDEN_OP = {
    "submit": "submit",
    "get_status": "getStatus",
    "get_results": "getResults",
    "cancel": "cancel",
}


#: Google's batch endpoint is model-scoped, so its adapter needs a model before
#: it has a request in hand. The freeze used this one.
GOOGLE_MODEL = "gemini-3-flash"


def adapter_for(provider: str) -> Any:
    if provider == "google":
        return GoogleBatchAdapter(api_key="k", model=GOOGLE_MODEL)
    return BATCH_ADAPTERS[provider](api_key="k")


def capturing(body: Any = None, status: int = 200) -> tuple[Any, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def fetch(request: dict[str, Any], options: Any = None) -> dict[str, Any]:
        seen.append(request)
        return {"status": status, "headers": {}, "body": BATCH_RESPONSE if body is None else body}

    return fetch, seen


# ── canonicalisation ────────────────────────────────────────────────────────


def parse_multipart(body: bytes, content_type: str) -> dict[str, Any]:
    """The encoded body read back into the shape the golden froze.

    Parsed rather than trusted: this is the only place the bytes that would
    actually leave the process are looked at.
    """
    boundary = content_type.split("boundary=", 1)[1]
    marker = f"--{boundary}".encode()
    entries: list[dict[str, Any]] = []
    for chunk in body.split(marker):
        if not chunk.strip() or chunk.startswith(b"--"):
            continue
        head, _, payload = chunk.lstrip(b"\r\n").partition(b"\r\n\r\n")
        headers = {
            k.strip().lower(): v.strip()
            for k, _, v in (line.partition(":") for line in head.decode().split("\r\n"))
            if k
        }
        disposition = headers.get("content-disposition", "")
        name = disposition.split('name="', 1)[1].split('"', 1)[0]
        payload = payload[:-2] if payload.endswith(b"\r\n") else payload
        if 'filename="' in disposition:
            entries.append(
                {
                    "name": name,
                    "filename": disposition.split('filename="', 1)[1].split('"', 1)[0],
                    # Parameters dropped: the golden carries Bun's Blob
                    # normalisation, which is the runtime's and not the library's.
                    "type": headers.get("content-type", "").split(";")[0],
                    "size": len(payload),
                }
            )
        else:
            entries.append({"name": name, "value": payload.decode("utf-8")})
    return {"__formData": entries}


def canon(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return {"__bytes": len(value)}
    if isinstance(value, list):
        return [canon(v) for v in value]
    if isinstance(value, dict):
        return {k: canon(value[k]) for k in sorted(value) if value[k] is not None}
    return value


def canon_request(request: dict[str, Any]) -> Any:
    out = dict(request)
    headers = dict(out.get("headers") or {})
    content_type = headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        out["body"] = parse_multipart(out["body"], content_type)
        # The TypeScript hands a FormData to its runtime, which sets this
        # header itself, so the frozen request has no record of it.
        headers.pop("content-type")
    out["headers"] = headers
    return canon(out)


def golden_of(request: dict[str, Any]) -> Any:
    frozen = canon(json.loads(json.dumps(request)))
    body = frozen.get("body") if isinstance(frozen, dict) else None
    if isinstance(body, dict) and "__formData" in body:
        for entry in body["__formData"]:
            if "type" in entry:
                entry["type"] = entry["type"].split(";")[0]
    return frozen


class TestTheSameRequestsAsTheTypeScript:
    """Nineteen frozen requests, held to what the other implementation sent."""

    @pytest.mark.parametrize(("provider", "op"), CASES, ids=[f"{p}/{o}" for p, o in CASES])
    def test_matches_the_frozen_requests(self, provider: str, op: str) -> None:
        adapter = adapter_for(provider)
        fetch, seen = capturing()

        if op == "submit":
            adapter.submit(REQUESTS, fetch)
        else:
            getattr(adapter, op)(BATCH_ID, fetch)

        assert seen, "the adapter sent nothing at all"
        for index, request in enumerate(seen):
            key = f"batch/{provider}/{GOLDEN_OP[op]}" + (f".{index}" if index else "")
            assert key in GOLDEN, f"no frozen request named {key}"
            assert canon_request(request) == golden_of(GOLDEN[key]), key

    def test_the_golden_covers_every_case_and_no_more(self) -> None:
        frozen = {k for k in GOLDEN if k.startswith("batch/")}
        expected = {f"batch/{p}/{GOLDEN_OP[o]}" for p, o in CASES} | {
            # The two-call flows, each captured in order.
            "batch/openai/submit.1",
            "batch/openai/getResults.1",
            "batch/xai/submit.1",
        }
        assert frozen == expected

    def test_the_comparison_can_fail(self) -> None:
        # The canary. Every assertion above is vacuous if the comparison cannot
        # discriminate, so one known-wrong build must be rejected.
        fetch, seen = capturing()
        OpenAIBatchAdapter(api_key="k", base_url="https://wrong.example").cancel(BATCH_ID, fetch)
        assert canon_request(seen[0]) != golden_of(GOLDEN["batch/openai/cancel"])


class TestTheTwoCallFlows:
    """Where one operation is two requests, and the second needs the first."""

    def test_openai_uploads_the_requests_then_creates_a_batch_over_the_file(self) -> None:
        fetch, seen = capturing()
        assert OpenAIBatchAdapter(api_key="k").submit(REQUESTS, fetch) == BATCH_ID

        assert [r["url"] for r in seen] == [
            "https://api.openai.com/v1/files",
            "https://api.openai.com/v1/batches",
        ]
        # The id from the FIRST answer is what the second request references.
        assert seen[1]["body"]["input_file_id"] == BATCH_ID

    def test_the_uploaded_jsonl_is_one_line_per_request(self) -> None:
        lines = OpenAIBatchAdapter(api_key="k").jsonl(REQUESTS).split("\n")
        assert [json.loads(line)["custom_id"] for line in lines] == ["r1", "r2"]
        assert json.loads(lines[0])["url"] == "/v1/responses"
        assert json.loads(lines[0])["body"] == dict(REQUESTS[0].body)

    def test_a_failed_upload_stops_before_creating_a_batch(self) -> None:
        # Creating a batch over a file that was never stored produces a job that
        # can only fail, minutes later, for a reason nothing records.
        fetch, seen = capturing(body={"error": "too large"}, status=413)
        with pytest.raises(RuntimeError, match="413"):
            OpenAIBatchAdapter(api_key="k").submit(REQUESTS, fetch)
        assert len(seen) == 1

    def test_openai_reads_results_by_first_asking_which_file(self) -> None:
        fetch, seen = capturing()
        OpenAIBatchAdapter(api_key="k").get_results(BATCH_ID, fetch)
        assert [r["url"] for r in seen] == [
            f"https://api.openai.com/v1/batches/{BATCH_ID}",
            "https://api.openai.com/v1/files/file_out/content",
        ]
        # JSONL decodes as text. Asking for json here breaks every batch read.
        assert seen[1]["responseType"] == "text"

    def test_no_output_file_yet_is_no_results_rather_than_an_error(self) -> None:
        # A batch that has not finished has no output file. That is not a
        # failure, it is "not yet".
        fetch, seen = capturing(body={"id": BATCH_ID, "status": "in_progress"})
        assert OpenAIBatchAdapter(api_key="k").get_results(BATCH_ID, fetch) == []
        assert len(seen) == 1, "it must not go looking for a file that was not named"

    def test_xai_creates_the_batch_then_adds_the_requests(self) -> None:
        fetch, seen = capturing(body={"batch_id": "b7"})
        assert XaiBatchAdapter(api_key="k").submit(REQUESTS, fetch) == "b7"
        assert [r["url"] for r in seen] == [
            "https://api.x.ai/v1/batches",
            "https://api.x.ai/v1/batches/b7/requests",
        ]

    def test_xai_names_the_half_built_batch_when_adding_fails(self) -> None:
        # The batch exists but is empty. Naming it lets a caller cancel the job
        # instead of polling one that will never fill.
        calls: list[dict[str, Any]] = []

        def fetch(request: dict[str, Any], options: Any = None) -> dict[str, Any]:
            calls.append(request)
            if len(calls) == 1:
                return {"status": 200, "headers": {}, "body": {"batch_id": "b7"}}
            return {"status": 500, "headers": {}, "body": {"error": "nope"}}

        with pytest.raises(RuntimeError, match="b7"):
            XaiBatchAdapter(api_key="k").submit(REQUESTS, fetch)

    def test_the_xai_batch_name_is_a_function_of_its_contents(self) -> None:
        # It used to embed the clock, which made it the one request in the
        # library no fixture could pin.
        first = XaiBatchAdapter(api_key="k").create_request(REQUESTS)["body"]["name"]
        second = XaiBatchAdapter(api_key="k").create_request(REQUESTS)["body"]["name"]
        other = XaiBatchAdapter(api_key="k").create_request(REQUESTS[:1])["body"]["name"]
        assert first == second
        assert first != other


class TestReadingEachProvidersProgress:
    """Four vocabularies, one status."""

    def status(self, provider: str, body: Any) -> Any:
        fetch, _ = capturing(body=body)
        return adapter_for(provider).get_status(BATCH_ID, fetch)

    @pytest.mark.parametrize(
        ("wire", "wanted"),
        [
            ("validating", PENDING),
            ("in_progress", PROCESSING),
            ("finalizing", PROCESSING),
            # Still work in progress: the job is running until it is not.
            ("cancelling", PROCESSING),
            ("completed", COMPLETED),
            ("failed", FAILED),
            ("expired", EXPIRED),
            ("cancelled", CANCELLED),
        ],
    )
    def test_openai_states(self, wire: str, wanted: str) -> None:
        assert self.status("openai", {"status": wire}).status == wanted

    def test_openai_counts_what_is_still_pending(self) -> None:
        got = self.status(
            "openai", {"status": "in_progress", "request_counts": {"total": 10, "completed": 4, "failed": 1}}
        )
        assert (got.total, got.completed, got.failed, got.pending) == (10, 4, 1, 5)

    @pytest.mark.parametrize(
        ("wire", "wanted"),
        [
            ("BATCH_STATE_PENDING", PENDING),
            ("BATCH_STATE_RUNNING", PROCESSING),
            ("BATCH_STATE_SUCCEEDED", COMPLETED),
            ("BATCH_STATE_FAILED", FAILED),
            ("BATCH_STATE_CANCELLED", CANCELLED),
            ("BATCH_STATE_EXPIRED", EXPIRED),
        ],
    )
    def test_google_states_live_under_metadata(self, wire: str, wanted: str) -> None:
        assert self.status("google", {"metadata": {"state": wire}}).status == wanted

    def test_google_done_overrides_the_state(self) -> None:
        assert self.status("google", {"done": True, "metadata": {"state": "BATCH_STATE_RUNNING"}}).status == COMPLETED

    def test_google_counts_nothing_done_until_the_job_is(self) -> None:
        # The inline list is present while the job runs; reading it as a tally
        # would report a half-finished batch as finished.
        running = {
            "metadata": {
                "state": "BATCH_STATE_RUNNING",
                "output": {"inlinedResponses": {"inlinedResponses": [
                    {"response": {"x": 1}}, {"error": {"code": 5}},
                ]}},
            }
        }
        got = self.status("google", running)
        assert (got.status, got.total, got.completed, got.failed, got.pending) == (
            PROCESSING, 2, 0, 0, 2,
        )

        done = {**running, "done": True}
        got = self.status("google", done)
        assert (got.status, got.completed, got.failed, got.pending) == (COMPLETED, 1, 1, 0)

    def test_xai_derives_a_state_from_its_counts(self) -> None:
        running = self.status("xai", {"state": {"num_requests": 5, "num_pending": 2, "num_success": 3}})
        assert (running.status, running.pending) == (PROCESSING, 2)

        finished = self.status(
            "xai", {"state": {"num_requests": 5, "num_pending": 0, "num_success": 4, "num_error": 1}}
        )
        assert (finished.status, finished.completed, finished.failed) == (COMPLETED, 4, 1)

        # Everything errored is a failed job, not a completed one with no answers.
        broken = self.status("xai", {"state": {"num_requests": 3, "num_pending": 0, "num_error": 3}})
        assert broken.status == FAILED

        # And everything cancelled is cancelled, not completed.
        stopped = self.status("xai", {"state": {"num_requests": 2, "num_pending": 0, "num_cancelled": 2}})
        assert stopped.status == CANCELLED

    def test_xai_counts_live_under_state_not_at_the_top_level(self) -> None:
        # Measured live 2026-09-04. Read from the top they are all zero, so
        # `total` is 0, the job never looks finished, and a polling caller waits
        # forever on a batch that completed in under a minute. Both libraries
        # had this; it is why the corpus cell was marked unsupported.
        top_level_only = self.status(
            "xai", {"num_requests": 5, "num_pending": 0, "num_success": 5}
        )
        assert top_level_only.total == 0, "top-level counts are not a shape xAI sends"

        nested = self.status("xai", {"state": {"num_requests": 5, "num_pending": 0, "num_success": 5}})
        assert (nested.total, nested.status) == (5, COMPLETED)

    def test_xai_totals_the_counts_when_the_provider_does_not(self) -> None:
        got = self.status("xai", {"state": {"num_pending": 1, "num_success": 2, "num_error": 1}})
        assert got.total == 4

    def test_a_failed_status_call_is_a_failed_batch_not_an_exception(self) -> None:
        # A poll is a loop. Raising from it would end a run over one bad minute.
        for provider in ("google", "xai"):
            fetch, _ = capturing(body={"error": "gone"}, status=500)
            assert adapter_for(provider).get_status(BATCH_ID, fetch).status == FAILED


class TestReadingEachProvidersResults:
    def test_openai_pairs_each_line_with_its_custom_id(self) -> None:
        lines = "\n".join(
            json.dumps(row)
            for row in [
                {"custom_id": "r2", "response": {"status_code": 200, "body": {"text": "second"}}},
                {"custom_id": "r1", "response": {"status_code": 200, "body": {"text": "first"}}},
            ]
        )
        calls: list[dict[str, Any]] = []

        def fetch(request: dict[str, Any], options: Any = None) -> dict[str, Any]:
            calls.append(request)
            payload: Any = BATCH_RESPONSE if len(calls) == 1 else lines
            return {"status": 200, "headers": {}, "body": payload}

        results = OpenAIBatchAdapter(api_key="k").get_results(BATCH_ID, fetch)
        # Out of order on the wire, and that is exactly the point: the id is the
        # only thing allowed to settle a caller's ticket.
        assert [r.custom_id for r in results] == ["r2", "r1"]
        assert results[0].response == {"text": "second"}
        assert all(r.success for r in results)

    def test_openai_marks_a_non_200_line_as_failed(self) -> None:
        line = json.dumps({"custom_id": "r1", "response": {"status_code": 400, "body": {}},
                           "error": {"message": "bad"}})
        calls: list[dict[str, Any]] = []

        def fetch(request: dict[str, Any], options: Any = None) -> dict[str, Any]:
            calls.append(request)
            return {"status": 200, "headers": {},
                    "body": BATCH_RESPONSE if len(calls) == 1 else line}

        (result,) = OpenAIBatchAdapter(api_key="k").get_results(BATCH_ID, fetch)
        assert result.success is False
        assert result.error is not None and "bad" in result.error

    def test_google_correlates_by_the_metadata_key(self) -> None:
        body = {
            "metadata": {"output": {"inlinedResponses": {"inlinedResponses": [
                {"response": {"text": "one"}, "metadata": {"key": "r1"}},
                {"error": {"code": 7}, "metadata": {"key": "r2"}},
            ]}}}
        }
        fetch, _ = capturing(body=body)
        results = GoogleBatchAdapter(api_key="k", model=GOOGLE_MODEL).get_results(BATCH_ID, fetch)

        assert [(r.custom_id, r.success) for r in results] == [("r1", True), ("r2", False)]
        assert results[1].error is not None and "7" in results[1].error

    def test_google_falls_back_to_a_positional_name_when_the_key_is_missing(self) -> None:
        # Not a good outcome, but better than an empty id: a caller can at least
        # see which position lost its correlation.
        body = {"metadata": {"output": {"inlinedResponses": {"inlinedResponses": [
            {"response": {"text": "one"}},
        ]}}}}
        fetch, _ = capturing(body=body)
        (result,) = GoogleBatchAdapter(api_key="k", model=GOOGLE_MODEL).get_results(BATCH_ID, fetch)
        assert result.custom_id == "req_0"

    def test_google_reads_an_absent_result_list_as_none(self) -> None:
        empty: list[dict[str, Any]] = [{}, {"metadata": {}}, {"metadata": {"output": {}}}]
        for body in empty:
            fetch, _ = capturing(body=body)
            assert GoogleBatchAdapter(api_key="k", model=GOOGLE_MODEL).get_results(BATCH_ID, fetch) == []

    def test_xai_unwraps_the_tagged_result(self) -> None:
        # The answer is nested and TAGGED, exactly as the request is:
        # `batch_result.response.<variant>`, where the variant names the API
        # that ran it. Measured live 2026-09-04 -- reading `row.response`
        # finds nothing, so every answer came back as a failure with no error.
        fetch, _ = capturing(body={"results": [
            {
                "batch_request_id": "r1",
                "batch_result": {
                    "response": {"chat_get_completion": {"object": "chat.completion", "id": "x"}}
                },
            }
        ]})
        (result,) = XaiBatchAdapter(api_key="k").get_results(BATCH_ID, fetch)

        assert result.custom_id == "r1"
        assert result.success is True
        assert result.response == {"object": "chat.completion", "id": "x"}

    def test_xai_takes_the_variant_whatever_it_is_called(self) -> None:
        # xAI has already added variants (`chat_get_completion`, `responses`,
        # `image_generation`, ...). An unknown one is still an answer, so the
        # single nested value is taken rather than matched against a list.
        fetch, _ = capturing(body={"results": [
            {"batch_request_id": "r1",
             "batch_result": {"response": {"something_new": {"id": "y"}}}}
        ]})
        (result,) = XaiBatchAdapter(api_key="k").get_results(BATCH_ID, fetch)
        assert result.success is True
        assert result.response == {"id": "y"}

    def test_xai_reads_either_container_and_either_id_field(self) -> None:
        under_results = {"results": [
            {"batch_request_id": "r1", "batch_result": {"response": {"v": {"text": "a"}}}}
        ]}
        under_data = {"data": [
            {"custom_id": "r2", "batch_result": {"response": {"v": {"text": "b"}}}}
        ]}
        for body, wanted in ((under_results, "r1"), (under_data, "r2")):
            fetch, _ = capturing(body=body)
            (result,) = XaiBatchAdapter(api_key="k").get_results(BATCH_ID, fetch)
            assert result.custom_id == wanted
            assert result.success is True

    def test_xai_prefers_the_providers_own_error_message(self) -> None:
        fetch, _ = capturing(body={"results": [
            {"batch_request_id": "r1", "status": "failed", "error_message": "context too long"},
        ]})
        (result,) = XaiBatchAdapter(api_key="k").get_results(BATCH_ID, fetch)
        assert result.success is False
        assert result.error == "context too long"

    def test_a_failed_results_call_is_empty_rather_than_an_exception(self) -> None:
        for provider in ("google", "xai"):
            fetch, _ = capturing(body={"error": "gone"}, status=500)
            assert adapter_for(provider).get_results(BATCH_ID, fetch) == []


class TestTheAdapterTable:
    def test_every_provider_with_a_batch_api_is_registered(self) -> None:
        assert set(BATCH_ADAPTERS) == {"anthropic", "google", "openai", "xai"}

    def test_each_one_answers_to_its_own_name(self) -> None:
        for name, factory in BATCH_ADAPTERS.items():
            assert factory(api_key="k").name == name

    def test_they_all_share_one_shape(self) -> None:
        # The reason a caller can write one batching loop.
        for factory in BATCH_ADAPTERS.values():
            adapter = factory(api_key="k")
            for method in ("submit", "get_status", "get_results", "cancel"):
                assert callable(getattr(adapter, method)), f"{adapter.name}.{method}"

    def test_openrouter_is_absent_rather_than_mapped_to_something_else(self) -> None:
        # It fronts other providers' models but hosts no batch API of its own.
        assert "openrouter" not in BATCH_ADAPTERS

    def test_every_batch_call_is_routed_under_a_named_queue(self) -> None:
        # A queue keyed on a model that does not exist would be nobody's, and
        # batch traffic must rate-limit separately from the completions it
        # stands in for.
        fetch, seen = capturing()
        AnthropicBatchAdapter(api_key="k").cancel(BATCH_ID, fetch)
        assert seen[0]["model"] == "batch"

        # Google is the exception, and deliberately: its endpoint is
        # model-scoped, so the real model is the routing key.
        fetch, seen = capturing()
        GoogleBatchAdapter(api_key="k", model="gemini-x").cancel(BATCH_ID, fetch)
        assert seen[0]["model"] == "gemini-x"
