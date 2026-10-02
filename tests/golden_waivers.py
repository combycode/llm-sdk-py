"""Fields the specs report that the FROZEN `response-golden.json` predates.

Waived, never re-recorded. Re-recording would silence that corpus's whole purpose:
it would absorb any OTHER drift sitting in the same cells along with the change
being made. So each entry names one key, the output is compared with that key
removed, and the key must actually be PRESENT -- a waiver that stops applying
fails rather than quietly protecting nothing.

Shared by every test that reads the corpus (`test_response_spec_differential`,
`test_stream_spec_differential`, `test_adapter_wiring`) so one deliberate change
is recorded once, in one place, rather than drifting between three copies of the
same list. The TypeScript twin keeps the same list in
`tests/unit/wire/response-spec-differential.test.ts`.

`container` (2026-10-02): Anthropic reports the code-execution container a turn ran
in, and we dropped it. These two cells were recorded on 2026-08-31 already carrying
`container: {id, expires_at}`, so the corpus is the evidence that real containers
had been going in the bin for over a month. Buffered: a new `response["container"]`.
Streamed: it rides the terminal frame, so it lands on the `done` event.
"""

from __future__ import annotations

from typing import Any

#: cell id -> (key, where). `where` is `"done"` when the key lands on a streamed
#: `done` event rather than on a buffered response object.
INTENTIONAL: dict[str, tuple[str, str | None]] = {
    "anthropic/messages::builtin.codeexec": ("container", None),
    "anthropic/messages::stream.builtin.codeexec": ("container", "done"),
}


def without_waived(cell_id: str, built: Any) -> Any:
    """`built` with one waived key removed, raising if it was not there.

    The assertion is the point: a waiver for a field that stopped being emitted
    would otherwise keep passing while protecting nothing.
    """
    waiver = INTENTIONAL.get(cell_id)
    if waiver is None:
        return built
    key, where = waiver
    if where == "done":
        events = [dict(e) for e in built]
        done = [e for e in events if e.get("type") == "done"]
        if not done:
            raise AssertionError(f"{cell_id}: waiver expects a `done` event; none was emitted")
        if not any(key in e for e in done):
            raise AssertionError(
                f"{cell_id}: waived key {key!r} is no longer on any `done` event -- "
                "remove the waiver or fix the regression"
            )
        return [
            {k: v for k, v in e.items() if not (e.get("type") == "done" and k == key)}
            for e in events
        ]
    obj = dict(built)
    if key not in obj:
        raise AssertionError(
            f"{cell_id}: waived key {key!r} is no longer reported -- "
            "remove the waiver or fix the regression"
        )
    del obj[key]
    return obj


__all__ = ["INTENTIONAL", "without_waived"]
