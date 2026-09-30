"""A 403 that means "your token is too narrow" is not a dead end.

Transposed from
`unified-library-ts/tests/unit/plugins/mcp/step-up-authorization.test.ts`.

Two refusals send a client back through authorization, and conflating them costs
one of the two:

* **401** -- the token is missing, expired or rejected. A refresh usually fixes
  it and the scope does not change.
* **403 with ``error="insufficient_scope"``** (SEP-2350) -- the token is valid
  and simply not broad enough. A refresh is useless: it mints another token
  carrying the scope that was just refused.

Only the 401 was handled, so a step-up surfaced as a plain failure and the
operation could never succeed however many times it was tried.

The scope asked for on a step-up is the UNION of what was already requested,
what the stored token was granted, and what the server now demands. Asking for
the challenged scope alone is the failure SEP-2350 exists to describe: the new
grant REPLACES the old one, so escalating one operation silently revokes the
permissions another was relying on.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Mapping
from typing import Any

sys.path.insert(0, "src")


from combycode_llm_sdk.mcp.errors import McpError
from combycode_llm_sdk.mcp.http import HttpTransport
from combycode_llm_sdk.mcp.oauth import parse_bearer_challenge, union_scopes


class TestParseBearerChallenge:
    def test_it_reads_the_parameters_a_step_up_needs(self) -> None:
        assert parse_bearer_challenge(
            'Bearer realm="mcp", error="insufficient_scope", scope="repo:write admin"'
        ) == {"realm": "mcp", "error": "insufficient_scope", "scope": "repo:write admin"}

    def test_it_accepts_bare_values_as_well_as_quoted_ones(self) -> None:
        assert parse_bearer_challenge("Bearer error=insufficient_scope, scope=read") == {
            "error": "insufficient_scope",
            "scope": "read",
        }

    def test_it_reads_resource_metadata_which_a_403_may_carry_too(self) -> None:
        # SEP-985. Carried on both refusals, which is why it is parsed here
        # rather than only on the unauthorized path.
        challenge = parse_bearer_challenge(
            'Bearer resource_metadata="https://a.test/.well-known/oauth-protected-resource"'
        )
        assert challenge is not None
        assert challenge["resource_metadata"] == (
            "https://a.test/.well-known/oauth-protected-resource"
        )

    def test_it_is_none_for_another_scheme_or_no_header(self) -> None:
        assert parse_bearer_challenge('Basic realm="x"') is None
        assert parse_bearer_challenge(None) is None

    def test_an_empty_challenge_is_not_none(self) -> None:
        # "there was a Bearer challenge" and "there was none" are different.
        assert parse_bearer_challenge("Bearer") == {}


class TestUnionScopes:
    def test_it_keeps_order_and_drops_repeats(self) -> None:
        assert union_scopes("a b", "b c") == "a b c"

    def test_it_survives_either_side_being_absent(self) -> None:
        assert union_scopes(None, "c") == "c"
        assert union_scopes("a", None) == "a"
        assert union_scopes(None, None) is None


class TestTheTransportReauthorizesOnTheRefusalsThatMeanIt:
    def drive(self, steps: list[tuple[int, str | None]]) -> list[Any]:
        reauth: list[Any] = []
        calls = {"n": 0}

        def fetch(req: Mapping[str, Any]) -> dict[str, Any]:
            status, www = steps[min(calls["n"], len(steps) - 1)]
            calls["n"] += 1
            headers = {"content-type": "application/json"}
            if www:
                headers["www-authenticate"] = www
            body = req.get("body") or {}
            return {
                "status": status,
                "headers": headers,
                "body": {"jsonrpc": "2.0", "id": body.get("id", 0), "result": {"ok": True}},
            }

        def on_unauthorized(scope: str | None = None) -> bool:
            reauth.append(scope)
            return True

        transport = HttpTransport(
            url="https://a.test/mcp",
            name="s",
            fetch=fetch,
            on_unauthorized=on_unauthorized,
        )
        with contextlib.suppress(McpError):
            # The scripted answers are not a real conversation; the re-auth
            # calls are the assertion. Narrow on purpose -- anything that is not
            # an McpError is a bug in this harness and should surface.
            transport.request("tools/list")
        return reauth

    def test_a_403_insufficient_scope_reauthorizes_with_the_challenged_scope(self) -> None:
        assert self.drive(
            [(403, 'Bearer error="insufficient_scope", scope="repo:write"'), (200, None)]
        ) == ["repo:write"]

    def test_a_401_still_reauthorizes_with_no_scope(self) -> None:
        # Nothing about the existing path changes: a 401 is not a scope problem.
        assert self.drive([(401, None), (200, None)]) == [None]

    def test_a_plain_403_does_not_reauthorize(self) -> None:
        # A real authorization failure: the caller may not do this whatever
        # token they hold, and re-authorizing would only loop.
        assert self.drive([(403, None), (200, None)]) == []

    def test_nor_a_403_naming_a_different_error(self) -> None:
        assert self.drive([(403, 'Bearer error="invalid_token"'), (200, None)]) == []

    def test_it_retries_once(self) -> None:
        challenge = 'Bearer error="insufficient_scope", scope="a"'
        assert len(self.drive([(403, challenge), (403, challenge)])) == 1



class TestAHandlerWrittenBeforeStepUpKeepsWorking:
    """`on_unauthorized` is a public constructor parameter.

    A handler written before step-up existed is ``def on_unauthorized() -> bool``
    and Python -- unlike JavaScript, where a surplus argument is ignored --
    raises TypeError when handed one. So the arity is inspected, not assumed.
    """

    def drive(self, callback: Any, status: int, www: str | None) -> int:
        calls = {"n": 0}

        def fetch(req: Mapping[str, Any]) -> dict[str, Any]:
            calls["n"] += 1
            headers = {"content-type": "application/json"}
            if www and calls["n"] == 1:
                headers["www-authenticate"] = www
            body = req.get("body") or {}
            return {
                "status": status if calls["n"] == 1 else 200,
                "headers": headers,
                "body": {"jsonrpc": "2.0", "id": body.get("id", 0), "result": {"ok": True}},
            }

        transport = HttpTransport(
            url="https://a.test/mcp", name="s", fetch=fetch, on_unauthorized=callback
        )
        transport.request("tools/list")
        return calls["n"]

    def test_a_zero_argument_handler_is_called_without_one(self) -> None:
        seen = {"n": 0}

        def on_unauthorized() -> bool:
            seen["n"] += 1
            return True

        assert self.drive(on_unauthorized, 401, None) == 2
        assert seen["n"] == 1

    def test_and_still_gets_the_retry_on_a_step_up_it_cannot_read(self) -> None:
        # It does not learn the scope -- it cannot -- but the request is still
        # retried, which is strictly better than raising TypeError at it.
        def on_unauthorized() -> bool:
            return True

        assert self.drive(
            on_unauthorized, 403, 'Bearer error="insufficient_scope", scope="admin"'
        ) == 2

    def test_a_one_argument_handler_receives_the_scope(self) -> None:
        got: list[Any] = []

        def on_unauthorized(scope: str | None = None) -> bool:
            got.append(scope)
            return True

        self.drive(on_unauthorized, 403, 'Bearer error="insufficient_scope", scope="admin"')
        assert got == ["admin"]
