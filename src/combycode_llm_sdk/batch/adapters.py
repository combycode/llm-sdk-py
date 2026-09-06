"""Each provider's batch API, built from the shared wire specs.

The specs are the etalon and they already exist here byte-identical, so an
adapter is thin: name the spec, hand it the input, and read the answer back.
What is NOT thin is the reading -- every provider reports status under its own
vocabulary and returns results in its own container, and normalising those is
the whole reason a caller can write one batching loop.

One detail worth naming: a results file is JSONL, so it is fetched as TEXT.
Asking for `json` there breaks every batch read, and nothing in a type system
catches it -- the TypeScript records the same trap.

Transposed from `unified-library-ts/src/llm/providers/*/batch.ts`.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from ..llm.providers.anthropic.constants import ANTHROPIC_API_VERSION
from ..llm.wire_multipart import MultipartFile, encode_multipart, to_form_data
from ..llm.wire_transforms import make_registry
from ..wire.interpreter import build_from_spec
from ..wire.service_specs import service_spec
from .types import (
    CANCELLED,
    COMPLETED,
    EXPIRED,
    FAILED,
    PENDING,
    PROCESSING,
    BatchRequest,
    BatchResult,
    BatchStatus,
)

_REGISTRY = make_registry({})


def _request(
    spec_id: str,
    provider: str,
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    response_type: str = "json",
) -> dict[str, Any]:
    """One spec-built request, in the wire shape an engine fetch takes."""
    built = build_from_spec(
        service_spec(spec_id), dict(payload), _REGISTRY, provider, None, dict(config)
    )
    request: dict[str, Any] = {
        "url": built.url,
        "method": built.method or "POST",
        "headers": dict(built.headers or {}),
        "provider": provider,
        # Every batch call is routed under one model name, so a queue rate-limits
        # the batch API separately from the completions it is standing in for.
        "model": "batch",
        "responseType": response_type,
    }
    # `bodyKind: none` means the request carries no body at all. The interpreter
    # says so with `no_body`; the wire wants the field simply absent, and a
    # `"body": None` on a GET is not the same document.
    if not getattr(built, "no_body", False):
        request["body"] = built.body
    return request


def _body_of(response: Any) -> Any:
    """The body, whether the fetch answered with a wire dict or an object."""
    if isinstance(response, Mapping):
        return response.get("body")
    return getattr(response, "body", None)


def _read(response: Any) -> tuple[int, dict[str, Any]]:
    """Status and JSON body, from whichever shape the fetch answered in."""
    if isinstance(response, Mapping):
        status = response.get("status")
        body = response.get("body")
    else:
        status = getattr(response, "status", None)
        body = getattr(response, "body", None)
    return (
        int(status) if isinstance(status, int) else 0,
        dict(body) if isinstance(body, Mapping) else {},
    )


def _lines(text: Any) -> list[str]:
    if not isinstance(text, str):
        return []
    return [line for line in text.strip().split("\n") if line.strip()]


class AnthropicBatchAdapter:
    """`POST /v1/messages/batches`, with every request inline."""

    name = "anthropic"
    #: Whether the adapter cannot be built without knowing the model.
    #: Only Google, whose batch endpoint is model-scoped.
    needs_model: ClassVar[bool] = False

    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com") -> None:
        self._config = {
            "apiKey": api_key,
            "baseURL": base_url,
            "apiVersion": ANTHROPIC_API_VERSION,
        }

    #: Anthropic's own words for a job's state, in ours.
    _STATUS: ClassVar[dict[str, str]] = {
        "in_progress": PROCESSING,
        "canceling": PROCESSING,
        "ended": COMPLETED,
        "expired": EXPIRED,
        "canceled": CANCELLED,
    }

    def submit_request(self, requests: Sequence[BatchRequest]) -> dict[str, Any]:
        return _request(
            "anthropic/batch.submit",
            "anthropic",
            {"requests": [{"customId": r.custom_id, "body": dict(r.body)} for r in requests]},
            self._config,
        )

    def submit(self, requests: Sequence[BatchRequest], fetch: Any) -> str:
        response = fetch(self.submit_request(requests))
        status, body = _read(response)
        if status >= 400:
            raise RuntimeError(f"Anthropic batch submit failed ({status}): {body}")
        batch_id = body.get("id")
        if not batch_id:
            raise RuntimeError(f"Anthropic batch submit returned no id: {body}")
        return str(batch_id)

    def get_status(self, batch_id: str, fetch: Any) -> BatchStatus:
        response = fetch(
            _request(
                "anthropic/batch.getStatus", "anthropic", {"batchId": batch_id}, self._config
            )
        )
        _, body = _read(response)
        counts = body.get("request_counts") or {}
        succeeded = int(counts.get("succeeded") or 0)
        errored = int(counts.get("errored") or 0)
        expired = int(counts.get("expired") or 0)
        processing = int(counts.get("processing") or 0)
        return BatchStatus(
            id=batch_id,
            status=self._STATUS.get(str(body.get("processing_status") or ""), PENDING),
            total=succeeded + errored + expired + processing + int(counts.get("canceled") or 0),
            completed=succeeded,
            failed=errored + expired,
            pending=processing,
        )

    def get_results(self, batch_id: str, fetch: Any) -> list[BatchResult]:
        response = fetch(
            _request(
                "anthropic/batch.getResults",
                "anthropic",
                {"batchId": batch_id},
                self._config,
                response_type="text",
            )
        )
        out: list[BatchResult] = []
        for line in _lines(_body_of(response)):
            entry = json.loads(line)
            result = entry.get("result") or {}
            succeeded = result.get("type") == "succeeded"
            out.append(
                BatchResult(
                    custom_id=str(entry.get("custom_id") or ""),
                    success=succeeded,
                    response=result.get("message") if succeeded else None,
                    error=None if succeeded else json.dumps(result),
                )
            )
        return out

    def cancel(self, batch_id: str, fetch: Any) -> None:
        fetch(
            _request("anthropic/batch.cancel", "anthropic", {"batchId": batch_id}, self._config)
        )


class OpenAIBatchAdapter:
    """Upload the requests as a JSONL file, then create a batch over it.

    Two calls to submit one batch, and the second cannot be built until the
    first has answered with a file id -- which is why `submit` is a method and
    not just a request builder.
    """

    name = "openai"
    needs_model: ClassVar[bool] = False

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com") -> None:
        self._config = {"apiKey": api_key, "baseURL": base_url}

    #: OpenAI's words for a job's state, in ours. `cancelling` is still work in
    #: progress: the job is running until it is not.
    _STATUS: ClassVar[dict[str, str]] = {
        "validating": PENDING,
        "in_progress": PROCESSING,
        "finalizing": PROCESSING,
        "cancelling": PROCESSING,
        "completed": COMPLETED,
        "failed": FAILED,
        "expired": EXPIRED,
        "cancelled": CANCELLED,
    }

    @staticmethod
    def jsonl(requests: Sequence[BatchRequest]) -> str:
        """The upload body: one request per line, each naming its own endpoint.

        Separators are tight because this is a wire document, and its byte count
        is what the multipart part declares.
        """
        return "\n".join(
            json.dumps(
                {
                    "custom_id": r.custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": dict(r.body),
                },
                separators=(",", ":"),
            )
            for r in requests
        )

    def upload_request(self, jsonl: str) -> dict[str, Any]:
        """Step one of submit: the requests go up as a file.

        The spec names the multipart FIELDS; the bytes are the serialised batch,
        which no spec can hold.
        """
        request = _request("openai/batch.uploadJsonl", "openai", {}, self._config)
        built = build_from_spec(
            service_spec("openai/batch.uploadJsonl"),
            {},
            _REGISTRY,
            "openai",
            None,
            dict(self._config),
        )
        encoded, content_type = encode_multipart(
            to_form_data(
                built.multipart or [],
                MultipartFile(
                    data=jsonl.encode("utf-8"),
                    filename="batch_input.jsonl",
                    mime_type="application/jsonl",
                ),
            )
        )
        request["body"] = encoded
        request["rawBody"] = True
        request["headers"]["content-type"] = content_type
        return request

    def create_request(self, file_id: str) -> dict[str, Any]:
        """Step two: the batch, over the file step one uploaded."""
        return _request("openai/batch.create", "openai", {"fileId": file_id}, self._config)

    def status_request(self, batch_id: str) -> dict[str, Any]:
        return _request("openai/batch.getStatus", "openai", {"batchId": batch_id}, self._config)

    def results_file_request(self, file_id: str) -> dict[str, Any]:
        """The output file is JSONL, so it decodes as TEXT.

        Asking for `json` here breaks every batch read and nothing in a type
        system catches it -- the TypeScript records the same trap.
        """
        return _request(
            "openai/batch.getResults",
            "openai",
            {"fileId": file_id},
            self._config,
            response_type="text",
        )

    def cancel_request(self, batch_id: str) -> dict[str, Any]:
        return _request("openai/batch.cancel", "openai", {"batchId": batch_id}, self._config)

    def submit(self, requests: Sequence[BatchRequest], fetch: Any) -> str:
        uploaded = fetch(self.upload_request(self.jsonl(requests)))
        status, body = _read(uploaded)
        if status >= 400:
            raise RuntimeError(f"OpenAI batch file upload failed ({status}): {body}")
        file_id = body.get("id")
        if not file_id:
            raise RuntimeError(f"OpenAI batch file upload returned no id: {body}")

        created = fetch(self.create_request(str(file_id)))
        status, body = _read(created)
        if status >= 400:
            raise RuntimeError(f"OpenAI batch create failed ({status}): {body}")
        batch_id = body.get("id")
        if not batch_id:
            raise RuntimeError(f"OpenAI batch create returned no id: {body}")
        return str(batch_id)

    def get_status(self, batch_id: str, fetch: Any) -> BatchStatus:
        _, body = _read(fetch(self.status_request(batch_id)))
        counts = body.get("request_counts") or {}
        total = int(counts.get("total") or 0)
        completed = int(counts.get("completed") or 0)
        failed = int(counts.get("failed") or 0)
        return BatchStatus(
            id=batch_id,
            status=self._STATUS.get(str(body.get("status") or ""), PENDING),
            total=total,
            completed=completed,
            failed=failed,
            pending=total - completed - failed,
        )

    def get_results(self, batch_id: str, fetch: Any) -> list[BatchResult]:
        # Two calls: the status names the output file, the file holds the lines.
        _, body = _read(fetch(self.status_request(batch_id)))
        output_file_id = body.get("output_file_id")
        if not output_file_id:
            return []

        response = fetch(self.results_file_request(str(output_file_id)))
        out: list[BatchResult] = []
        for line in _lines(_body_of(response)):
            entry = json.loads(line)
            answer = entry.get("response")
            answer = answer if isinstance(answer, Mapping) else None
            out.append(
                BatchResult(
                    custom_id=str(entry.get("custom_id") or ""),
                    success=answer is not None and answer.get("status_code") == 200,
                    response=answer.get("body") if answer else None,
                    error=json.dumps(entry["error"]) if entry.get("error") else None,
                )
            )
        return out

    def cancel(self, batch_id: str, fetch: Any) -> None:
        fetch(self.cancel_request(batch_id))


def _inlined(body: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Google's per-request answers, four levels down.

    `metadata.output.inlinedResponses.inlinedResponses[]` -- the doubled name is
    the provider's, not a typo here.
    """
    metadata = body.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    output = metadata.get("output")
    output = output if isinstance(output, Mapping) else {}
    wrapper = output.get("inlinedResponses")
    wrapper = wrapper if isinstance(wrapper, Mapping) else {}
    items = wrapper.get("inlinedResponses")
    return [i for i in items if isinstance(i, Mapping)] if isinstance(items, list) else []


class GoogleBatchAdapter:
    """One call to submit, and the answers come back on the job resource itself."""

    name = "google"
    needs_model: ClassVar[bool] = True

    #: The endpoint is model-scoped, so a batch has to know which model it is for
    #: before it has a request in hand.
    DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com",
    ) -> None:
        self._model = model or self.DEFAULT_MODEL
        self._config = {"apiKey": api_key, "baseURL": base_url, "model": self._model}

    #: The live batch resource keeps its state under `metadata.state`.
    _STATUS: ClassVar[dict[str, str]] = {
        "BATCH_STATE_PENDING": PENDING,
        "BATCH_STATE_RUNNING": PROCESSING,
        "BATCH_STATE_SUCCEEDED": COMPLETED,
        "BATCH_STATE_FAILED": FAILED,
        "BATCH_STATE_CANCELLED": CANCELLED,
        "BATCH_STATE_EXPIRED": EXPIRED,
    }

    def _wire(self, spec_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = _request(spec_id, "google", payload, self._config)
        # Routed under the REAL model rather than a literal `batch`: the endpoint
        # is model-scoped, and queueing and cost attribution both key off this.
        request["model"] = self._model
        return request

    def submit_request(self, requests: Sequence[BatchRequest]) -> dict[str, Any]:
        return self._wire("google/batch.submit", {"requests": _as_wire(requests)})

    def status_request(self, batch_id: str) -> dict[str, Any]:
        return self._wire("google/batch.getStatus", {"batchId": batch_id})

    def results_request(self, batch_id: str) -> dict[str, Any]:
        """The same wire as the status call: Google returns results inline, so
        these are two operations that happen to share one request."""
        return self._wire("google/batch.getResults", {"batchId": batch_id})

    def cancel_request(self, batch_id: str) -> dict[str, Any]:
        return self._wire("google/batch.cancel", {"batchId": batch_id})

    def submit(self, requests: Sequence[BatchRequest], fetch: Any) -> str:
        status, body = _read(fetch(self.submit_request(requests)))
        if status >= 400:
            raise RuntimeError(f"Google batch submit failed ({status}): {body}")
        # `name` is the resource path Google polls by. The local id is a last
        # resort, so a submit that answered 200 without one still hands back
        # something a caller can hold rather than an empty string.
        return str(body.get("name") or body.get("id") or f"local-{uuid.uuid4()}")

    def get_status(self, batch_id: str, fetch: Any) -> BatchStatus:
        status, body = _read(fetch(self.status_request(batch_id)))
        if status >= 400:
            return BatchStatus(id=batch_id, status=FAILED)

        metadata = body.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        state = self._STATUS.get(str(metadata.get("state") or ""), PENDING)
        state = COMPLETED if body.get("done") else state

        items = _inlined(body)
        total = len(items)
        failed = sum(1 for i in items if i.get("error") or not i.get("response"))
        done = state == COMPLETED
        return BatchStatus(
            id=batch_id,
            status=state,
            total=total,
            # Nothing counts as done until the JOB is: the inline list is present
            # while the job runs, and reading it as a tally would report a
            # half-finished batch as finished.
            completed=total - failed if done else 0,
            failed=failed if done else 0,
            pending=0 if done else total,
        )

    def get_results(self, batch_id: str, fetch: Any) -> list[BatchResult]:
        status, body = _read(fetch(self.results_request(batch_id)))
        if status >= 400:
            return []
        out: list[BatchResult] = []
        for index, item in enumerate(_inlined(body)):
            metadata = item.get("metadata")
            key = metadata.get("key") if isinstance(metadata, Mapping) else None
            answer = item.get("response")
            out.append(
                BatchResult(
                    # The key is the custom_id the request went out with, so a
                    # caller's ticket is settled by identity and never by order.
                    custom_id=str(key) if key else f"req_{index}",
                    success=bool(answer) and not item.get("error"),
                    response=answer if isinstance(answer, Mapping) else None,
                    error=json.dumps(item["error"]) if item.get("error") else None,
                )
            )
        return out

    def cancel(self, batch_id: str, fetch: Any) -> None:
        fetch(self.cancel_request(batch_id))


class XaiBatchAdapter:
    """Create the batch, then add the requests to it."""

    name = "xai"
    needs_model: ClassVar[bool] = False

    def __init__(self, api_key: str, base_url: str = "https://api.x.ai") -> None:
        self._config = {"apiKey": api_key, "baseURL": base_url}

    def create_request(self, requests: Sequence[BatchRequest]) -> dict[str, Any]:
        """Step one. The batch's NAME is a function of its contents, so the
        request is reproducible -- it used to embed the clock, which made it the
        one request in the library no fixture could pin."""
        return _request("xai/batch.create", "xai", {"requests": _as_wire(requests)}, self._config)

    def add_requests_request(
        self, batch_id: str, requests: Sequence[BatchRequest]
    ) -> dict[str, Any]:
        return _request(
            "xai/batch.addRequests",
            "xai",
            {"batchId": batch_id, "requests": _as_wire(requests)},
            self._config,
        )

    def status_request(self, batch_id: str) -> dict[str, Any]:
        return _request("xai/batch.getStatus", "xai", {"batchId": batch_id}, self._config)

    def results_request(self, batch_id: str) -> dict[str, Any]:
        return _request("xai/batch.getResults", "xai", {"batchId": batch_id}, self._config)

    def cancel_request(self, batch_id: str) -> dict[str, Any]:
        return _request("xai/batch.cancel", "xai", {"batchId": batch_id}, self._config)

    def submit(self, requests: Sequence[BatchRequest], fetch: Any) -> str:
        status, body = _read(fetch(self.create_request(requests)))
        if status >= 400:
            raise RuntimeError(f"xAI batch create failed ({status}): {body}")
        batch_id = body.get("batch_id") or body.get("id")
        if not batch_id:
            raise RuntimeError(f"xAI batch create returned no id: {body}")

        status, body = _read(fetch(self.add_requests_request(str(batch_id), requests)))
        if status >= 400:
            # The batch exists but is empty. Naming it lets a caller cancel the
            # half-built job instead of polling one that will never fill.
            raise RuntimeError(f"xAI batch {batch_id}: adding requests failed ({status}): {body}")
        return str(batch_id)

    def get_status(self, batch_id: str, fetch: Any) -> BatchStatus:
        status, body = _read(fetch(self.status_request(batch_id)))
        if status >= 400:
            return BatchStatus(id=batch_id, status=FAILED)

        # The counts live under `state`, NOT at the top level. Measured live
        # 2026-09-04: reading them from the top gets zeroes for every field, so
        # `total` is 0, the batch never looks finished, and `wait()` polls until
        # something outside it gives up. The job had completed in under a
        # minute; nothing could see it.
        counts = body.get("state")
        counts = counts if isinstance(counts, Mapping) else {}
        pending = int(counts.get("num_pending") or 0)
        succeeded = int(counts.get("num_success") or 0)
        errored = int(counts.get("num_error") or 0)
        cancelled = int(counts.get("num_cancelled") or 0)
        total = int(
            counts.get("num_requests") or (pending + succeeded + errored + cancelled)
        )

        # xAI reports counts rather than a state, so the state is derived: work
        # still pending means running, everything errored means failed, and
        # everything cancelled means cancelled.
        if pending == 0 and total > 0:
            if errored == total:
                state = FAILED
            elif cancelled == total:
                state = CANCELLED
            else:
                state = COMPLETED
        else:
            state = PROCESSING
        return BatchStatus(
            id=batch_id,
            status=state,
            total=total,
            completed=succeeded,
            failed=errored,
            pending=pending,
        )

    def get_results(self, batch_id: str, fetch: Any) -> list[BatchResult]:
        status, body = _read(fetch(self.results_request(batch_id)))
        if status >= 400:
            return []
        rows = body.get("results")
        if not isinstance(rows, list):
            rows = body.get("data")
        out: list[BatchResult] = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, Mapping):
                continue
            answer, error = _xai_result(row)
            out.append(
                BatchResult(
                    custom_id=str(row.get("batch_request_id") or row.get("custom_id") or ""),
                    success=answer is not None and error is None,
                    response=answer,
                    error=error,
                )
            )
        return out

    def cancel(self, batch_id: str, fetch: Any) -> None:
        fetch(self.cancel_request(batch_id))


def _xai_result(row: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, str | None]:
    """One xAI result row, unwrapped.

    The answer is nested and TAGGED, exactly as the request is: the row carries
    `batch_result.response.<variant>`, where the variant names the API that ran
    it (`chat_get_completion`, `responses`, `image_generation`, ...). Reading
    `row.response` -- which is what a flat shape would suggest, and what both
    libraries did -- finds nothing, so every answer came back as a failure with
    no error to explain it. Measured live 2026-09-04.

    The variant is taken WHATEVER it is called rather than matched against a
    list: xAI has already added variants, and an unknown one is still an answer.
    """
    error = row.get("error_message") or row.get("error")
    outer = row.get("batch_result")
    outer = outer if isinstance(outer, Mapping) else row
    if outer.get("error") and not error:
        error = outer["error"]
    answer = outer.get("response")
    if isinstance(answer, Mapping) and len(answer) == 1:
        only = next(iter(answer.values()))
        if isinstance(only, Mapping):
            answer = only
    if not isinstance(answer, Mapping):
        answer = None
    if error is not None and not isinstance(error, str):
        error = json.dumps(error)
    return answer, error or None


def _as_wire(requests: Sequence[BatchRequest]) -> list[dict[str, Any]]:
    """The request list in the shape every batch spec reads."""
    return [{"customId": r.custom_id, "body": dict(r.body)} for r in requests]


#: Provider name -> its batch adapter.
BATCH_ADAPTERS: Mapping[str, type[Any]] = {
    "anthropic": AnthropicBatchAdapter,
    "google": GoogleBatchAdapter,
    "openai": OpenAIBatchAdapter,
    "xai": XaiBatchAdapter,
}


__all__ = [
    "BATCH_ADAPTERS",
    "AnthropicBatchAdapter",
    "GoogleBatchAdapter",
    "OpenAIBatchAdapter",
    "XaiBatchAdapter",
]
