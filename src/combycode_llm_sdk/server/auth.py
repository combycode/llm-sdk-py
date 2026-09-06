"""Who is asking, and whether they may.

A slot rather than a policy: this library cannot know whether a deployment
authenticates with a shared key, a signed token or nothing at all. What it can
fix is the SHAPE -- a verifier returns an identity or refuses, the identity
scopes stored conversations, and a refusal is a 401 with a reason.

Transposed from `unified-library-ts/src/server/auth.ts`.
"""

from __future__ import annotations

import hmac
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class AuthError(Exception):
    """The credential was missing, malformed, or unknown."""


@dataclass(frozen=True)
class AuthVerifyResult:
    """Who the request turned out to be."""

    #: The stable owner id. Scopes anything stored for this caller.
    user_id: str
    #: Free-form, for whatever the deployment layers on top (roles, scopes).
    metadata: Mapping[str, Any] = field(default_factory=dict)


class AuthPlugin(Protocol):
    """Inspect the headers; return an identity or raise `AuthError`."""

    def verify(self, headers: Mapping[str, str]) -> AuthVerifyResult: ...


class BearerKeyAuth:
    """The simplest authenticator: known keys, each mapped to a user.

    Accepts either a mapping of key -> user id, or a bare list of keys when the
    caller has no user model to attach them to.
    """

    def __init__(self, keys: Mapping[str, str] | Sequence[str]) -> None:
        if isinstance(keys, Mapping):
            self._keys = {str(k): str(v) for k, v in keys.items()}
        else:
            # Anonymous keys still need an id, because that id is what scopes a
            # stored conversation -- sharing one across every key would let one
            # caller continue another's chat.
            self._keys = {str(k): f"key:{str(k)[:8]}" for k in keys}

    def verify(self, headers: Mapping[str, str]) -> AuthVerifyResult:
        header = headers.get("authorization") or headers.get("Authorization")
        if not header or not header.startswith("Bearer "):
            raise AuthError('missing or malformed Authorization header (expected "Bearer <key>")')
        offered = header[7:].strip()
        user_id = self._match(offered)
        if user_id is None:
            raise AuthError("unknown bearer key")
        return AuthVerifyResult(user_id=user_id)

    def _match(self, offered: str) -> str | None:
        """Compare against every key in constant time.

        A dict lookup would answer faster for a wrong key than a right one, and
        that difference is measurable across enough requests. The cost is a walk
        over the key list, which is small and is the point.
        """
        found: str | None = None
        for key, user_id in self._keys.items():
            if hmac.compare_digest(key, offered):
                found = user_id
        return found

    @property
    def size(self) -> int:
        return len(self._keys)

    def __repr__(self) -> str:
        return f"<BearerKeyAuth {len(self._keys)} key(s)>"


__all__ = ["AuthError", "AuthPlugin", "AuthVerifyResult", "BearerKeyAuth"]
