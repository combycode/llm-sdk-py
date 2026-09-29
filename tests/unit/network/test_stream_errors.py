"""A failed stream reports through the taxonomy, like every other failure.

A stream that fails before the first byte used to raise a bare `RuntimeError`
carrying nothing but a status number -- outside the error taxonomy entirely, so
a caller branching on `LLMError.kind` saw nothing and the error BODY, the part
that says why, was thrown away. A connection dropping mid-stream escaped as
whatever the transport raised.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from combycode_llm_sdk.network.errors import AuthError, LLMError, RateLimitError
from combycode_llm_sdk.transport import TransportResponse, as_fetch_stream

REQ: dict[str, Any] = {
    "url": "https://example.com/v1/x",
    "headers": {},
    "body": {},
    "provider": "anthropic",
    "model": "claude-3-5",
}


def _transport(status: int, body: bytes) -> Any:
    def transport(_req: Any) -> TransportResponse:
        return TransportResponse(status=status, headers={}, body=iter([body]))

    return transport


class TestAFailedConnect:
    def test_a_401_is_an_auth_error_not_a_runtime_error(self) -> None:
        fetch = as_fetch_stream(_transport(401, b'{"error":{"message":"bad key"}}'))
        with pytest.raises(AuthError) as excinfo:
            list(fetch(REQ))
        assert "bad key" in str(excinfo.value)

    def test_a_429_is_a_rate_limit_error(self) -> None:
        fetch = as_fetch_stream(_transport(429, b'{"error":{"message":"slow down"}}'))
        with pytest.raises(RateLimitError):
            list(fetch(REQ))

    def test_the_status_survives_even_when_the_body_is_unreadable(self) -> None:
        fetch = as_fetch_stream(_transport(503, b"<html>gateway</html>"))
        with pytest.raises(LLMError) as excinfo:
            list(fetch(REQ))
        assert excinfo.value.status == 503

    def test_a_good_stream_is_untouched(self) -> None:
        fetch = as_fetch_stream(_transport(200, b'data: {"a":1}\n\n'))
        assert [e["data"] for e in fetch(REQ)] == ['{"a":1}']


class TestAFailureMidStream:
    def test_is_wrapped_as_a_non_retryable_network_error(self) -> None:
        def transport(_req: Any) -> TransportResponse:
            def chunks() -> Iterator[bytes]:
                yield b'data: {"a":1}\n\n'
                raise ConnectionResetError("connection reset")

            return TransportResponse(status=200, headers={}, body=chunks())

        fetch = as_fetch_stream(transport)
        seen: list[Any] = []
        with pytest.raises(LLMError) as excinfo:
            # `extend`, not `list(...)`: extend appends as it iterates, so what
            # arrived BEFORE the failure is kept, while `list()` would raise and
            # discard the lot -- and what arrived is the whole point here.
            seen.extend(fetch(REQ))
        assert len(seen) == 1
        assert excinfo.value.kind == "network"
        # NOT retryable: the caller already holds part of the answer, so
        # re-opening would deliver a second, overlapping stream.
        assert excinfo.value.retryable is False
        assert "1 event" in str(excinfo.value)
