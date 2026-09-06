"""Error taxonomy -- each kind gets different retry behaviour.

Transposed from `unified-library-ts/src/network/errors.ts`, with one deliberate
Python shape change. TypeScript has ONE `LLMError` carrying a `kind` string,
because that is how a JavaScript caller branches. The API contract asks for
exceptions instead::

    except RateLimitError as e:
        e.retry_after
    except LLMError as e:            # base class for everything we raise
        e.provider, e.model, e.status

So there is a subclass per kind and `kind` is kept alongside, which means both
readings work and neither is a second source of truth: `classify_error` builds
the subclass, and the subclass sets its own `kind`.

This lives in the network layer so a queue can classify a failure before any
hook fires.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping
from email.utils import parsedate_to_datetime
from typing import Any

#: `type ErrorKind` (errors.ts:4).
ERROR_KINDS = (
    "rate_limit",
    "auth",
    "context_overflow",
    "invalid_request",
    "server_error",
    "timeout",
    "network",
    "content_filter",
    "model_not_found",
    "quota_exceeded",
    "unsupported",
)


class LLMError(Exception):
    """The base of everything this library raises for a failed call."""

    #: The taxonomy entry, kept so a caller can branch on a string as the
    #: TypeScript does, without needing to know the class hierarchy.
    kind = "server_error"

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        model: str = "",
        status: int | None = None,
        retryable: bool = False,
        retry_after_ms: float | None = None,
        raw: Any = None,
        kind: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model
        self.status = status
        self.retryable = retryable
        self.retry_after_ms = retry_after_ms
        self.raw = raw
        if kind is not None:
            self.kind = kind

    @property
    def retry_after(self) -> float | None:
        """How long the server asked us to wait, in SECONDS.

        Seconds because that is what a bare `retry_after` means everywhere else
        in Python -- `time.sleep(err.retry_after)` should be right. The
        millisecond value the retry machinery works in stays available as
        `retry_after_ms`, and the config names say `_ms` for the same reason.
        """
        return None if self.retry_after_ms is None else self.retry_after_ms / 1000

    def __str__(self) -> str:
        where = f"{self.provider}/{self.model}".strip("/")
        status = f" [{self.status}]" if self.status is not None else ""
        return f"{self.message}{status}" + (f" ({where})" if where else "")


class RequestAborted(LLMError):
    """A handler refused the request before it was sent.

    An `LLMError` rather than a bare RuntimeError: a caller wrapping a call in
    `except LLMError` is asking to hear about every reason it did not happen,
    and a context guard declining is one of them.
    """

    kind = "aborted"


class RateLimitError(LLMError):
    """429. Retryable, and usually carries a `Retry-After`."""

    kind = "rate_limit"


class AuthError(LLMError):
    """401 / 403. Never retryable -- the key will not become valid on its own."""

    kind = "auth"


class ContextOverflowError(LLMError):
    """The prompt did not fit. Retrying it unchanged sends the same prompt."""

    kind = "context_overflow"


class InvalidRequestError(LLMError):
    kind = "invalid_request"


class ServerError(LLMError):
    """5xx. Retryable: the request may be fine and the server temporarily not."""

    kind = "server_error"


class TimeoutError(LLMError):  # the taxonomy name; shadowing the builtin is intended
    """Our own deadline, not the server's. Retryable."""

    kind = "timeout"


class NetworkError(LLMError):
    """The request never reached a server. Retryable."""

    kind = "network"


class ContentFilterError(LLMError):
    kind = "content_filter"


class ModelNotFoundError(LLMError):
    kind = "model_not_found"


class QuotaExceededError(LLMError):
    """402 / 413. Not retryable: more attempts spend nothing but time."""

    kind = "quota_exceeded"


class UnsupportedError(LLMError):
    kind = "unsupported"


#: kind -> the class `classify_error` raises for it.
ERROR_CLASSES: dict[str, type[LLMError]] = {
    "rate_limit": RateLimitError,
    "auth": AuthError,
    "context_overflow": ContextOverflowError,
    "invalid_request": InvalidRequestError,
    "server_error": ServerError,
    "timeout": TimeoutError,
    "network": NetworkError,
    "content_filter": ContentFilterError,
    "model_not_found": ModelNotFoundError,
    "quota_exceeded": QuotaExceededError,
    "unsupported": UnsupportedError,
}

_CONTEXT = re.compile(r"context|token|too long|max_tokens|too many tokens", re.IGNORECASE)
_NOT_FOUND = re.compile(r"model.*not found|does not exist|unknown model", re.IGNORECASE)
_UNSUPPORTED = re.compile(r"not support|unsupported", re.IGNORECASE)


def classify_error(
    provider: str,
    status: int,
    body: Any,
    headers: Mapping[str, str] | None = None,
    *,
    model: str = "",
) -> LLMError:
    """Map an HTTP status and a provider error body onto the taxonomy."""
    headers = headers or {}
    message = extract_error_message(body)

    if status in (401, 403):
        return AuthError(message, provider=provider, model=model, status=status, raw=body)

    if status == 429:
        return RateLimitError(
            message,
            provider=provider,
            model=model,
            status=status,
            retryable=True,
            retry_after_ms=parse_retry_after(headers),
            raw=body,
        )

    if status == 400:
        # The status alone cannot tell these apart -- every one of them is a 400
        # -- so the message is the only signal, and the order matters: a context
        # overflow often also says "not supported".
        for pattern, cls in (
            (_CONTEXT, ContextOverflowError),
            (_NOT_FOUND, ModelNotFoundError),
            (_UNSUPPORTED, UnsupportedError),
        ):
            if pattern.search(message):
                return cls(message, provider=provider, model=model, status=status, raw=body)
        return InvalidRequestError(
            message, provider=provider, model=model, status=status, raw=body
        )

    if status in (402, 413):
        return QuotaExceededError(
            message, provider=provider, model=model, status=status, raw=body
        )

    # 5xx, and anything unrecognised: retryable only when it is actually a
    # server error, so a stray 3xx is not retried forever.
    return ServerError(
        message,
        provider=provider,
        model=model,
        status=status,
        retryable=status >= 500,
        raw=body,
    )


def extract_error_message(body: Any) -> str:
    """The human-readable message, wherever this provider put it."""
    if not isinstance(body, Mapping):
        return str(body if body is not None else "Unknown error")
    error = body.get("error")
    if isinstance(error, Mapping):
        return str(error.get("message") or _json(error))
    if isinstance(error, str):
        return error
    if isinstance(body.get("message"), str):
        return str(body["message"])
    return _json(body)[:500]


def parse_retry_after(headers: Mapping[str, str]) -> float | None:
    """`Retry-After` per RFC 9110: delay-seconds OR an HTTP-date.

    Both forms are parsed. A skewed client clock can only produce a value the
    caller's cap rejects, never a negative wait -- `_usable` sees to that.
    """
    ms = headers.get("retry-after-ms")
    if ms:
        try:
            usable = _usable(float(int(ms)))
        except ValueError:
            usable = None
        if usable is not None:
            return usable

    value = headers.get("retry-after")
    if not value:
        return None

    try:
        return _usable(int(value) * 1000)
    except ValueError:
        pass

    try:
        at = parsedate_to_datetime(value).timestamp() * 1000
    except (TypeError, ValueError):
        return None
    return _usable(at - time.time() * 1000)


def _usable(ms: float) -> float | None:
    """A delay is usable only if it is finite and non-negative.

    Anything else -- a NaN from a malformed header, an infinity, a negative from
    a skewed clock -- is discarded rather than propagated: a sleep of NaN returns
    immediately, turning one bad header into a retry storm.
    """
    return ms if math.isfinite(ms) and ms >= 0 else None


def _json(value: Any) -> str:
    from ..wire.interpreter import js_json

    try:
        return js_json(value)
    except (TypeError, ValueError):
        return str(value)


__all__ = [
    "ERROR_CLASSES",
    "ERROR_KINDS",
    "AuthError",
    "ContentFilterError",
    "ContextOverflowError",
    "InvalidRequestError",
    "LLMError",
    "ModelNotFoundError",
    "NetworkError",
    "QuotaExceededError",
    "RateLimitError",
    "ServerError",
    "TimeoutError",
    "UnsupportedError",
    "classify_error",
    "extract_error_message",
    "parse_retry_after",
]
