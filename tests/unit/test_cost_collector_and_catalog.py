"""The ledger, the catalog view, and the file descriptor.

Three small surfaces the reviewed examples reach for, each of which had a gap
that only running those examples exposed.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from combycode_llm_sdk import LLM, Engine, TransportResponse
from combycode_llm_sdk.catalog.catalog import ModelCatalog, resolve_catalog
from combycode_llm_sdk.cost_collector import CostCollector, summarize
from combycode_llm_sdk.results import FileOutput

PRICED = "claude-haiku-4.5"


def engine_with(transport: Any) -> Engine:
    return Engine(
        catalog="defaults",
        api_keys={"anthropic": "k"},
        transport=transport,
        register_as_default=False,
    )


def anthropic_body(output_tokens: int = 1000) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1_000_000, "output_tokens": output_tokens},
    }


class TestTheCatalogResolver:
    def test_a_name_becomes_a_catalog(self) -> None:
        assert resolve_catalog("defaults").size > 0

    def test_empty_is_a_catalog_with_nothing_in_it(self) -> None:
        assert resolve_catalog("empty").size == 0

    def test_an_instance_passes_through(self) -> None:
        mine = ModelCatalog()
        assert resolve_catalog(mine) is mine

    def test_nothing_means_the_defaults(self) -> None:
        assert resolve_catalog(None).size > 0

    def test_an_unknown_name_is_an_error_not_a_fallback(self) -> None:
        # A typo would otherwise produce a working engine whose catalog is not
        # the one asked for, and the first symptom is an unpriced model reading
        # as free.
        with pytest.raises(ValueError, match="not a known catalog"):
            resolve_catalog("defualts")

    def test_an_engine_accepts_the_name(self) -> None:
        assert engine_with(lambda r: TransportResponse(body={})).catalog.size > 0


class TestTheCatalogEntryView:
    def test_an_entry_reads_as_attributes(self) -> None:
        info = resolve_catalog("defaults").get("anthropic", PRICED)
        assert info is not None
        assert info.model == PRICED

    def test_it_still_reads_as_a_mapping(self) -> None:
        # Every consumer inside the library does, and the field set is open.
        info = resolve_catalog("defaults").get("anthropic", PRICED)
        assert info is not None
        assert info["model"] == PRICED
        assert "pricing" in dict(info)

    def test_a_missing_field_names_what_the_entry_holds(self) -> None:
        info = resolve_catalog("defaults").get("anthropic", PRICED)
        assert info is not None
        with pytest.raises(AttributeError, match="has no 'banana'"):
            info.banana  # noqa: B018

    def test_list_hands_out_the_same_shape_as_get(self) -> None:
        # One accessor returning a different shape from the other is the kind of
        # difference nobody notices until it is in a loop.
        entries = resolve_catalog("defaults").list("anthropic")
        assert entries
        assert entries[0].model == entries[0]["model"]

    def test_an_unknown_model_is_still_None(self) -> None:
        assert resolve_catalog("defaults").get("anthropic", "no-such-model") is None


class TestTheLedger:
    def test_a_priced_call_contributes_to_the_total(self) -> None:
        engine = engine_with(lambda r: TransportResponse(body=anthropic_body()))
        LLM(model=f"anthropic/{PRICED}", engine=engine).complete("hi")
        summary = engine.cost.total()
        assert summary.entries == 1
        assert summary.unpriced == 0
        assert summary.total > 0

    def test_an_unpriced_model_is_counted_and_named(self) -> None:
        # $0.00 because the model is unknown reads identically to $0.00 because
        # the call was free, unless it is counted separately.
        engine = engine_with(lambda r: TransportResponse(body=anthropic_body()))
        LLM(model="anthropic/claude-private-tune", engine=engine).complete("hi")
        summary = engine.cost.total()
        assert summary.unpriced == 1
        assert summary.unpriced_models == ("anthropic/claude-private-tune",)
        assert summary.total == 0.0

    def test_the_warning_fires_once_per_model_not_per_request(self) -> None:
        # An unpriced model is a configuration fact, not a per-request event;
        # repeating it trains the reader to ignore it.
        engine = engine_with(lambda r: TransportResponse(body=anthropic_body()))
        seen: list[Any] = []
        engine.hooks.on("onWarning", seen.append)
        llm = LLM(model="anthropic/claude-private-tune", engine=engine)
        llm.complete("one")
        llm.complete("two")
        unpriced = [w for w in seen if w["code"] == "unpriced_model"]
        assert len(unpriced) == 1
        assert engine.cost.total().unpriced == 2

    def test_a_summary_of_nothing_is_zero_not_an_error(self) -> None:
        empty = summarize([])
        assert empty.entries == 0
        assert empty.total == 0.0
        assert empty.unpriced_models == ()

    def test_clearing_forgets_the_warning_too(self) -> None:
        # After a clear the next call is the first evidence again; staying
        # silent would leave a fresh ledger with an unexplained unpriced count.
        engine = engine_with(lambda r: TransportResponse(body=anthropic_body()))
        seen: list[Any] = []
        engine.hooks.on("onWarning", seen.append)
        llm = LLM(model="anthropic/claude-private-tune", engine=engine)
        llm.complete("one")
        engine.cost.clear()
        llm.complete("two")
        assert len([w for w in seen if w["code"] == "unpriced_model"]) == 2

    def test_destroy_stops_recording(self) -> None:
        engine = engine_with(lambda r: TransportResponse(body=anthropic_body()))
        engine.cost.destroy()
        LLM(model=f"anthropic/{PRICED}", engine=engine).complete("hi")
        assert engine.cost.total().entries == 0

    def test_a_collector_groups_by_the_id_actually_called(self) -> None:
        # The pinned snapshot (`claude-haiku-4-5-20251001`), not our slug: the
        # ledger records what was billed, and two slugs resolving to one
        # snapshot are one line on the invoice.
        engine = engine_with(lambda r: TransportResponse(body=anthropic_body()))
        LLM(model=f"anthropic/{PRICED}", engine=engine).complete("hi")
        keys = list(engine.cost.by_model())
        assert len(keys) == 1
        assert keys[0].startswith("anthropic/claude-haiku-4-5")

    def test_it_is_wired_before_the_first_call(self) -> None:
        # A collector created when someone first reads `engine.cost` would
        # report an empty run.
        engine = engine_with(lambda r: TransportResponse(body=anthropic_body()))
        assert isinstance(engine.cost, CostCollector)


class TestTheFileDescriptor:
    RESPONSES_BODY: ClassVar[dict[str, Any]] = {
        "id": "resp_1",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "saved"}]},
            {
                "type": "code_interpreter_call",
                "outputs": [
                    {
                        "type": "file",
                        "file_id": "file_abc",
                        "filename": "fib.txt",
                        "mime_type": "text/plain",
                    }
                ],
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }

    def transport(self) -> Any:
        def stub(request: Any) -> TransportResponse:
            if request.url.endswith("/files/file_abc/content"):
                return TransportResponse(body=b"1 1 2 3 5")
            return TransportResponse(body=self.RESPONSES_BODY)

        return stub

    def result(self) -> Any:
        llm = LLM(model="openai/gpt-5.4-nano", api_key="k", transport=self.transport())
        return llm.complete("go", builtin_tools=["code_interpreter"])

    def test_a_saved_file_is_reported_not_only_an_image(self) -> None:
        # A run that wrote a CSV reported no files at all, and `result.files`
        # was empty for the one scenario the feature exists for.
        files = self.result().files
        assert len(files) == 1
        assert files[0].filename == "fib.txt"
        assert files[0].mime_type == "text/plain"

    def test_read_fetches_only_when_asked(self) -> None:
        # A chart can be large; eagerly downloading every file would make an
        # innocuous call slow and occasionally enormous.
        descriptor = self.result().files[0]
        assert descriptor.read() == b"1 1 2 3 5"

    def test_save_writes_the_bytes(self, tmp_path: Any) -> None:
        target = tmp_path / "out.txt"
        assert self.result().files[0].save(target) == target
        assert target.read_bytes() == b"1 1 2 3 5"

    def test_saving_into_a_directory_uses_the_files_own_name(self, tmp_path: Any) -> None:
        written = self.result().files[0].save(tmp_path)
        assert written.name == "fib.txt"

    def test_an_unattached_descriptor_says_why_it_cannot_fetch(self) -> None:
        with pytest.raises(RuntimeError, match="not attached to a client"):
            FileOutput(id="f1", name="x.txt").read()

    def test_the_fetcher_is_not_part_of_identity(self) -> None:
        # Two descriptors of the same file are the same file, whichever client
        # happens to be able to fetch them.
        assert FileOutput(id="f1") == FileOutput(id="f1", fetch=lambda d: b"")


class TestServerStateHandle:
    def test_the_interactions_handle_is_the_providers_name(self) -> None:
        # Reading `id` and minting a uuid gives back a handle the provider has
        # never seen: the request is well-formed, the reply is fine, and the
        # conversation silently starts over on every turn.
        sent: list[Any] = []

        def stub(request: Any) -> TransportResponse:
            sent.append(request.body)
            return TransportResponse(
                body={
                    "name": "interactions/int_123",
                    "steps": [
                        {"type": "model_output", "content": [{"type": "text", "text": "ok"}]}
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                }
            )

        llm = LLM(
            model="google/gemini-2.5-flash", api_key="k", api="interactions", transport=stub
        )
        first = llm.complete("My name is Ada.")
        assert first.state == "interactions/int_123"

        llm.complete("What is my name?", state=first.state)
        assert sent[1]["previous_interaction_id"] == "interactions/int_123"
        assert "Ada" not in str(sent[1])
