"""`result.warnings`, and stubs that say what they are however you touch them.

Both found by a live corpus run rather than by reading: `10b_code_exec_files.py`
died on `'Completion' object has no attribute 'warnings'`, and
`31_hosted_retrieval.py` on `type object 'Corpus' has no attribute 'create'`.
"""

from __future__ import annotations

from combycode_llm_sdk import Message, complete
from combycode_llm_sdk.results import Completion, WarningNote
from combycode_llm_sdk.transport import TransportRequest, TransportResponse

OK_BODY = {
    "id": "msg_1",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "OK"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


class Stub:
    def __init__(self) -> None:
        self.seen: list[TransportRequest] = []

    def __call__(self, request: TransportRequest) -> TransportResponse:
        self.seen.append(request)
        return TransportResponse(status=200, body=OK_BODY)


class TestWarnings:
    def test_warnings_is_always_a_sequence(self) -> None:
        # `10b` iterates it unconditionally. Empty, never None, so no caller
        # branches on None for the ordinary case.
        got = complete(
            model="anthropic/claude-haiku-4.5",
            api_key="k",
            transport=Stub(),
            prompt="hi",
            max_tokens=8,
        )
        assert got.warnings == ()
        assert list(got.warnings) == []

    def test_a_build_note_reaches_the_caller_as_a_warning(self) -> None:
        # Google refuses code execution beside a PDF, so the request goes without
        # it -- and the caller is TOLD rather than silently losing the capability.
        wire = {
            "text": "ok",
            "warnings": [
                {
                    "source": "llm",
                    "code": "request_adjusted",
                    "message": "code_interpreter dropped: not allowed beside a document",
                    "details": {"provider": "google"},
                }
            ],
        }
        got = Completion.of(wire, provider="google", model="m", api="generate", cost=None)
        assert len(got.warnings) == 1
        assert got.warnings[0].message.startswith("code_interpreter dropped")
        assert got.warnings[0].code == "request_adjusted"

    def test_the_hook_and_the_result_describe_one_adjustment_the_same_way(self) -> None:
        # Two reports of one fact must not disagree; the shapes are shared so a
        # subscriber and a caller holding only the result read the same sentence.
        note = WarningNote.of(
            {"source": "llm", "code": "request_adjusted", "message": "m", "details": {"a": 1}}
        )
        assert (note.source, note.code, note.message, note.details) == (
            "llm",
            "request_adjusted",
            "m",
            {"a": 1},
        )

    def test_a_note_with_no_details_still_builds(self) -> None:
        assert WarningNote.of({"message": "m"}).details == {}


class TestNothingIsAStubAnyMore:
    """The port has no unported names left, and must not grow one back."""

    def test_the_last_stub_was_shadowing_a_type_that_was_already_ported(self) -> None:
        # `Message` was the final stub, and it sat on top of a real alias:
        # llm/types/messages.py has carried `Message = dict[str, Any]` all
        # along, deliberately an alias rather than a TypedDict because these
        # dicts go straight to the wire interpreter, which reads them by string
        # key. So `from combycode_llm_sdk import Message` raised "not ported
        # yet" for something that was right there.
        from combycode_llm_sdk.llm.types.messages import Message as Ported

        assert Message is Ported

    def test_the_stub_machinery_itself_is_gone(self) -> None:
        import combycode_llm_sdk as package

        assert not hasattr(package, "_Unported")

    def test_no_public_name_reports_itself_unported(self) -> None:
        # The gate that replaces the old per-stub tests: it does not need to
        # know which names exist, so a stub reintroduced under any name fails
        # here rather than reaching a caller.
        import combycode_llm_sdk as package

        for name in dir(package):
            if name.startswith("_"):
                continue
            value = getattr(package, name)
            assert type(value).__name__ != "_Unported", name

    def test_the_names_a_caller_reaches_for_are_real(self) -> None:
        from combycode_llm_sdk import LLM, AsyncLLM, Realtime, acomplete, tool

        for real in (LLM, AsyncLLM, Realtime, complete, acomplete, tool):
            assert callable(real)


def test_completion_of_ignores_a_warnings_key_that_is_not_a_list() -> None:
    # Defensive only where a provider could actually send it: `warnings` is ours,
    # but `Completion.of` reads whatever the wire dict holds.
    got = Completion.of(
        {"text": "x", "warnings": None}, provider="p", model="m", api="a", cost=None
    )
    assert got.warnings == ()
