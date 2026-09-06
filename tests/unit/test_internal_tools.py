"""A tool whose body is a prompt, and the steps written once so it need not.

What is worth pinning here is the ORDER: the input is refused before anything
is sent, the schema is shown to the model before the answer is checked against
it, and a mismatched answer is reported without being discarded. Each of those
is a decision that would still "work" if reversed, and would cost money or lose
an answer every time it ran.
"""

from __future__ import annotations

from typing import Any

import pytest

from combycode_llm_sdk import TransportResponse
from combycode_llm_sdk.hooks import HookBus
from combycode_llm_sdk.internal_tools import (
    JSON_FORMAT,
    TEXT_FORMAT,
    InternalTool,
    InternalToolContext,
    InternalToolError,
    InternalToolRunner,
    InternalToolRunnerConfig,
    LLMToolDefinition,
    LocalBackend,
    ModelPreference,
    PromptVariant,
    ToolFilter,
    ToolRegistry,
    apply_schema_defaults,
    define_llm_tool,
    format_tool_id,
    id_without_version,
    matches_version,
    parse_json_with_fences,
    parse_tool_id,
    render_template,
    select_variant,
    try_parse_tool_id,
)

MODEL = "openai/gpt-4.1-mini"


def definition(**overrides: Any) -> LLMToolDefinition:
    base: dict[str, Any] = {
        "id": "orxa:sentiment@1.0.0",
        "namespace": "orxa",
        "name": "sentiment",
        "version": "1.0.0",
        "description": "Judge how a customer message reads.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "audience": {"type": "string", "default": "support"},
            },
            "required": ["content"],
        },
        "model_preference": ModelPreference(preferred_model=MODEL, max_tokens=200),
        "output_schema": {"type": "object", "properties": {"label": {"type": "string"}}},
        "output_format": JSON_FORMAT,
        "system_prompt": "Judge the sentiment of a message written to {{audience}}.",
        "user_template": "{{content}}",
    }
    base.update(overrides)
    return LLMToolDefinition(**base)


class Model:
    """One scripted answer, and every request it was handed."""

    def __init__(self, text: str = '{"label": "angry"}') -> None:
        self.text = text
        self.requests: list[Any] = []

    def __call__(self, request: Any) -> TransportResponse:
        self.requests.append(request)
        return TransportResponse(
            status=200,
            body={
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": self.text}],
                    }
                ],
                "usage": {"input_tokens": 11, "output_tokens": 3},
            },
        )


def runner_for(
    model: Model, *, tool: InternalTool | None = None, hooks: HookBus | None = None, **extra: Any
) -> InternalToolRunner:
    resolved = tool if tool is not None else define_llm_tool(definition())
    registry = ToolRegistry().add_backend(LocalBackend().register(resolved))
    return InternalToolRunner(
        InternalToolRunnerConfig(
            registry=registry,
            api_keys={"openai": "k"},
            transport=model,
            hooks=hooks,
            **extra,
        )
    )


class TestTheDeclarationReachesTheWire:
    def test_the_user_template_renders_into_the_request(self) -> None:
        model = Model()
        runner_for(model).run("orxa:sentiment@1.0.0", {"content": "third time"})
        assert model.requests[-1].body["input"][0]["content"] == "third time"

    def test_a_schema_default_fills_the_system_template(self) -> None:
        # Without it the prompt renders a hole for every caller who left the
        # field out, which reads fine to the model and answers the wrong question.
        model = Model()
        runner_for(model).run("orxa:sentiment@1.0.0", {"content": "x"})
        assert "written to support." in model.requests[-1].body["instructions"]

    def test_a_supplied_value_beats_the_default(self) -> None:
        model = Model()
        runner_for(model).run("orxa:sentiment@1.0.0", {"content": "x", "audience": "billing"})
        assert "written to billing." in model.requests[-1].body["instructions"]

    def test_a_json_tool_gets_the_format_contract_first(self) -> None:
        model = Model()
        runner_for(model).run("orxa:sentiment@1.0.0", {"content": "x"})
        assert model.requests[-1].body["instructions"].startswith("You are a JSON API endpoint.")

    def test_the_output_schema_is_shown_to_the_model(self) -> None:
        # Checking the answer against a schema the model never saw is a test it
        # was not told it was sitting.
        model = Model()
        runner_for(model).run("orxa:sentiment@1.0.0", {"content": "x"})
        assert "## Output schema (your JSON must match)" in model.requests[-1].body["instructions"]

    def test_an_example_is_shown_when_one_is_declared(self) -> None:
        model = Model()
        tool = define_llm_tool(definition(output_example={"label": "angry"}))
        runner_for(model, tool=tool).run("orxa:sentiment@1.0.0", {"content": "x"})
        assert "## Output example (copy this shape exactly)" in (
            model.requests[-1].body["instructions"]
        )

    def test_a_text_tool_gets_neither(self) -> None:
        model = Model(text="plain words")
        tool = define_llm_tool(definition(output_format=TEXT_FORMAT, output_schema=None))
        result = runner_for(model, tool=tool).run("orxa:sentiment@1.0.0", {"content": "x"})
        assert result == "plain words"
        assert "JSON API endpoint" not in model.requests[-1].body["instructions"]

    def test_the_model_preference_reaches_the_request(self) -> None:
        model = Model()
        runner_for(model).run("orxa:sentiment@1.0.0", {"content": "x"})
        assert model.requests[-1].body["max_output_tokens"] == 200

    def test_the_declaration_survives_on_the_tool(self) -> None:
        # What lets a bench harness re-render a prompt without paying to run it.
        tool = define_llm_tool(definition())
        assert tool.definition is not None
        assert tool.definition.user_template == "{{content}}"


class TestTheAnswer:
    def test_json_is_read_back_as_a_value(self) -> None:
        result = runner_for(Model('{"label": "angry", "confidence": 0.91}')).run(
            "orxa:sentiment@1.0.0", {"content": "x"}
        )
        assert result == {"label": "angry", "confidence": 0.91}

    def test_a_fenced_answer_is_still_read(self) -> None:
        result = runner_for(Model('```json\n{"label": "calm"}\n```')).run(
            "orxa:sentiment@1.0.0", {"content": "x"}
        )
        assert result == {"label": "calm"}

    def test_a_wrong_shape_is_reported_and_still_returned(self) -> None:
        # The answer is already paid for; discarding it turns a shape problem
        # into a shape problem plus a lost response.
        hooks = HookBus()
        seen: list[Any] = []
        hooks.on("on_warning", seen.append)
        result = runner_for(Model('[{"label": "angry"}]'), hooks=hooks).run(
            "orxa:sentiment@1.0.0", {"content": "x"}
        )
        assert result == [{"label": "angry"}]
        assert [w.code for w in seen] == ["output_schema_mismatch"]

    def test_a_matching_shape_says_nothing(self) -> None:
        hooks = HookBus()
        seen: list[Any] = []
        hooks.on("on_warning", seen.append)
        runner_for(Model(), hooks=hooks).run("orxa:sentiment@1.0.0", {"content": "x"})
        assert seen == []

    def test_a_boolean_does_not_pass_for_a_number(self) -> None:
        # `bool` is an `int` in Python, so the obvious check accepts `True` for
        # a declared confidence.
        hooks = HookBus()
        seen: list[Any] = []
        hooks.on("on_warning", seen.append)
        tool = define_llm_tool(definition(output_schema={"type": "number"}))
        runner_for(Model("true"), tool=tool, hooks=hooks).run(
            "orxa:sentiment@1.0.0", {"content": "x"}
        )
        assert [w.code for w in seen] == ["output_schema_mismatch"]

    def test_non_json_from_a_json_tool_is_an_error_naming_the_tool(self) -> None:
        with pytest.raises(InternalToolError, match="orxa:sentiment@1.0.0"):
            runner_for(Model("I'd rather not.")).run("orxa:sentiment@1.0.0", {"content": "x"})

    def test_a_truncated_answer_says_so_rather_than_blaming_the_prompt(self) -> None:
        # Live, a reasoning model spent the tool's whole 200-token budget on
        # thinking and returned `{"label": "Negative", `. "Not JSON" and "cut
        # off" are opposite problems -- one wants a bigger budget, the other a
        # better prompt -- and the chain would re-run the first identically on
        # every fallback model.
        class Truncating:
            def __call__(self, request: Any) -> TransportResponse:
                return TransportResponse(
                    status=200,
                    body={
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": '{"label": "N'}],
                            }
                        ],
                        "incomplete_details": {"reason": "max_output_tokens"},
                        "status": "incomplete",
                        "usage": {"input_tokens": 11, "output_tokens": 200},
                    },
                )

        with pytest.raises(InternalToolError, match="cut off at max_tokens=200"):
            runner_for(Truncating()).run(  # type: ignore[arg-type]
                "orxa:sentiment@1.0.0", {"content": "x"}
            )


class TestWhatIsRefusedBeforeItCosts:
    def test_a_missing_required_input_is_refused(self) -> None:
        model = Model()
        with pytest.raises(InternalToolError, match="missing required input: content"):
            runner_for(model).run("orxa:sentiment@1.0.0", {"audience": "billing"})
        assert model.requests == []

    def test_a_non_object_input_is_refused(self) -> None:
        model = Model()
        with pytest.raises(InternalToolError, match="expects an object"):
            runner_for(model).run("orxa:sentiment@1.0.0", ["content"])
        assert model.requests == []

    def test_an_unregistered_id_is_refused(self) -> None:
        with pytest.raises(InternalToolError, match="not found in registry"):
            runner_for(Model()).run("orxa:nothing@1.0.0", {"content": "x"})

    def test_no_key_for_the_whole_chain_fails_once_naming_the_providers(self) -> None:
        # Rather than once per model with "no API key", which buries the fact
        # that no model in the chain was ever reachable.
        tool = define_llm_tool(
            definition(
                model_preference=ModelPreference(
                    preferred_model="anthropic/claude-haiku-4.5",
                    fallback_models=("google/gemini-2.5-flash",),
                )
            )
        )
        registry = ToolRegistry().add_backend(LocalBackend().register(tool))
        runner = InternalToolRunner(
            InternalToolRunnerConfig(
                registry=registry, api_keys={"openai": "k"}, transport=Model()
            )
        )
        with pytest.raises(InternalToolError, match=r"anthropic, google.*openai"):
            runner.run("orxa:sentiment@1.0.0", {"content": "x"})


class TestTheModelChain:
    def test_the_preferred_model_is_tried_first(self) -> None:
        model = Model()
        runner_for(model).run("orxa:sentiment@1.0.0", {"content": "x"})
        assert model.requests[-1].model == "gpt-4.1-mini"

    def test_a_failure_falls_through_to_the_next_model_and_warns(self) -> None:
        # The failure has to be one the tool raises, not one the transport
        # raises: the executor retries a transport error itself, so a flaky
        # socket never reaches the model chain at all.
        class Fussy:
            """Answers JSON only for the fallback model."""

            def __init__(self) -> None:
                self.models: list[str] = []

            def __call__(self, request: Any) -> TransportResponse:
                self.models.append(request.model)
                text = "no thanks" if request.model == "gpt-4.1-mini" else '{"label": "angry"}'
                return Model(text)(request)

        transport = Fussy()
        hooks = HookBus()
        seen: list[Any] = []
        hooks.on("on_warning", seen.append)
        tool = define_llm_tool(
            definition(
                model_preference=ModelPreference(
                    preferred_model=MODEL, fallback_models=("openai/gpt-4.1",), max_tokens=200
                )
            )
        )
        result = runner_for(transport, tool=tool, hooks=hooks).run(  # type: ignore[arg-type]
            "orxa:sentiment@1.0.0", {"content": "x"}
        )
        assert result == {"label": "angry"}
        assert "internal_tool_fallback" in [w.code for w in seen]
        # The fallback must actually reach the fallback MODEL. Pooling clients
        # by provider handed it the client pinned to the one that just failed.
        assert transport.models == ["gpt-4.1-mini", "gpt-4.1"]

    def test_a_chain_that_fails_everywhere_names_every_failure(self) -> None:
        class Broken:
            def __call__(self, request: Any) -> TransportResponse:
                raise RuntimeError("down")

        tool = define_llm_tool(
            definition(
                model_preference=ModelPreference(
                    preferred_model=MODEL, fallback_models=("openai/gpt-4.1",)
                )
            )
        )
        with pytest.raises(InternalToolError, match="failed on all 2 model"):
            runner_for(Broken(), tool=tool).run(  # type: ignore[arg-type]
                "orxa:sentiment@1.0.0", {"content": "x"}
            )

    def test_a_benchmark_recommendation_outranks_the_tools_own_preference(self) -> None:
        from combycode_llm_sdk.internal_tools import ToolCompat

        model = Model()
        runner = runner_for(
            model, compat={"orxa:sentiment@1.0.0": ToolCompat(recommended=("openai/gpt-4.1",))}
        )
        runner.run("orxa:sentiment@1.0.0", {"content": "x"})
        assert model.requests[-1].model == "gpt-4.1"

    def test_one_client_is_pooled_per_provider(self) -> None:
        # A client per model would mean one rate limiter per model against a
        # quota that is the provider's.
        model = Model()
        runner = runner_for(model)
        runner.run("orxa:sentiment@1.0.0", {"content": "x"})
        runner.run("orxa:sentiment@1.0.0", {"content": "y"})
        assert runner.pool_size == 1
        runner.destroy()
        assert runner.pool_size == 0


class TestAToolWhoseBodyIsCode:
    def test_it_runs_without_any_model(self) -> None:
        # The runner cannot tell a prompt-backed tool from a real one, which is
        # what lets one replace the other without touching a caller.
        def execute(value: Any, ctx: InternalToolContext) -> Any:
            return {"label": str(value["content"]).upper()}

        tool = InternalTool(
            id="orxa:shout@1.0.0",
            namespace="orxa",
            name="shout",
            version="1.0.0",
            description="Shout it back.",
            input_schema={"type": "object", "required": ["content"]},
            execute=execute,
        )
        registry = ToolRegistry().add_backend(LocalBackend().register(tool))
        runner = InternalToolRunner(InternalToolRunnerConfig(registry=registry))
        assert runner.run("orxa:shout@1.0.0", {"content": "hi"}) == {"label": "HI"}

    def test_its_failure_is_not_swallowed(self) -> None:
        def execute(value: Any, ctx: InternalToolContext) -> Any:
            raise ValueError("no")

        tool = InternalTool(
            id="orxa:bad@1.0.0",
            namespace="orxa",
            name="bad",
            version="1.0.0",
            description="Fails.",
            input_schema={"type": "object"},
            execute=execute,
        )
        registry = ToolRegistry().add_backend(LocalBackend().register(tool))
        runner = InternalToolRunner(InternalToolRunnerConfig(registry=registry))
        with pytest.raises(ValueError, match="no"):
            runner.run("orxa:bad@1.0.0", {})


class TestTheRegistry:
    def test_first_backend_wins_an_id_conflict(self) -> None:
        # So a local override is a matter of ordering, not of deleting anything.
        first = LocalBackend().register(define_llm_tool(definition(description="first")))

        class Second(LocalBackend):
            name = "second"

        second = Second().register(define_llm_tool(definition(description="second")))
        registry = ToolRegistry().add_backend(first).add_backend(second)
        found = registry.get("orxa:sentiment@1.0.0")
        assert found is not None and found.description == "first"

    def test_a_duplicate_backend_name_is_refused(self) -> None:
        registry = ToolRegistry().add_backend(LocalBackend())
        with pytest.raises(ValueError, match="already registered"):
            registry.add_backend(LocalBackend())

    def test_registering_the_same_id_twice_is_refused(self) -> None:
        backend = LocalBackend().register(define_llm_tool(definition()))
        with pytest.raises(ValueError, match="already registered"):
            backend.register(define_llm_tool(definition()))

    def test_replace_is_the_way_to_say_it_out_loud(self) -> None:
        backend = LocalBackend().register(define_llm_tool(definition()))
        backend.replace(define_llm_tool(definition(description="new")))
        assert backend.size == 1
        replaced = backend.get("orxa:sentiment@1.0.0")
        assert replaced is not None and replaced.description == "new"

    def test_a_removed_backend_is_gone_from_the_merged_view(self) -> None:
        registry = ToolRegistry().add_backend(LocalBackend().register(define_llm_tool(definition())))
        assert len(registry) == 1
        assert registry.remove_backend("local") is True
        assert registry.list() == []

    def test_search_ranks_an_exact_name_above_a_description_match(self) -> None:
        exact = define_llm_tool(definition())
        other = define_llm_tool(
            definition(
                id="orxa:triage@1.0.0", name="triage", description="route by sentiment"
            )
        )
        registry = ToolRegistry().add_backend(LocalBackend().register(exact).register(other))
        assert [t.name for t in registry.search("sentiment")] == ["sentiment", "triage"]

    def test_a_filter_selects_by_namespace_and_tag(self) -> None:
        tagged = define_llm_tool(definition(tags=("nlp",)))
        registry = ToolRegistry().add_backend(LocalBackend().register(tagged))
        assert registry.find(ToolFilter(tag="nlp")) == [tagged]
        assert registry.find(ToolFilter(tag="vision")) == []
        assert registry.find(ToolFilter(namespace="other")) == []

    def test_models_for_without_a_catalog_answers_nothing_rather_than_guessing(self) -> None:
        registry = ToolRegistry().add_backend(LocalBackend().register(define_llm_tool(definition())))
        assert registry.models_for("orxa:sentiment@1.0.0") == []


class TestIds:
    def test_an_id_carries_the_version_because_the_prompt_is_the_implementation(self) -> None:
        parsed = parse_tool_id("orxa:sentiment@1.0.0")
        assert (parsed.namespace, parsed.name, parsed.version) == ("orxa", "sentiment", "1.0.0")

    def test_a_missing_version_is_not_an_id(self) -> None:
        with pytest.raises(ValueError, match="invalid tool id"):
            parse_tool_id("orxa:sentiment")

    def test_format_refuses_what_parse_would_reject(self) -> None:
        assert format_tool_id("orxa", "sentiment", "1.0.0") == "orxa:sentiment@1.0.0"
        with pytest.raises(ValueError, match="invalid version"):
            format_tool_id("orxa", "sentiment", "1.0")

    def test_the_unversioned_form_is_for_grouping(self) -> None:
        assert id_without_version("orxa:sentiment@1.0.0") == "orxa:sentiment"
        assert id_without_version("not-an-id") == "not-an-id"

    def test_the_namespace_is_checked_before_the_name(self) -> None:
        # Both are wrong here; the message has to name the first one, or a
        # caller fixes the name and gets the identical error again.
        with pytest.raises(ValueError, match="invalid namespace"):
            format_tool_id("BAD", "ALSO_BAD", "bad")

    def test_try_parse_answers_none_instead_of_raising(self) -> None:
        assert try_parse_tool_id("orxa:sentiment@1.0.0") is not None
        assert try_parse_tool_id("not-an-id") is None

    def test_a_version_match_is_exact_equality(self) -> None:
        # NOT semver ranges: a version identifies a WORDING, and 1.0.1 can
        # answer differently from 1.0.0 in a way no range syntax describes.
        assert matches_version("1.0.0", "1.0.0")
        assert not matches_version("1.0.0", "1.0.1")
        assert not matches_version("1.0.0", "1.0.0 ")


class TestTemplates:
    def test_a_missing_variable_is_refused_rather_than_left_as_a_hole(self) -> None:
        with pytest.raises(KeyError, match="who"):
            render_template("hello {{who}}", {})

    def test_a_dotted_path_reaches_into_a_nested_value(self) -> None:
        assert render_template("{{a.b}}", {"a": {"b": "deep"}}) == "deep"

    def test_a_non_string_is_rendered_as_json(self) -> None:
        assert render_template("{{n}}", {"n": [1, 2]}) == "[1, 2]"

    def test_none_renders_as_nothing_not_as_the_word(self) -> None:
        assert render_template("[{{n}}]", {"n": None}) == "[]"

    def test_a_fenced_scalar_needs_the_fence_stripped(self) -> None:
        # The block extractor looks for a brace or a bracket, so it cannot find
        # this one: a fenced scalar is the case the fence strip actually earns.
        assert parse_json_with_fences('```json\n"calm"\n```') == "calm"
        assert parse_json_with_fences("```\n0.91\n```") == 0.91

    def test_json_is_found_after_a_preamble(self) -> None:
        assert parse_json_with_fences('Sure! {"a": 1} hope that helps') == {"a": 1}

    def test_a_brace_inside_a_string_does_not_end_the_block(self) -> None:
        assert parse_json_with_fences('x {"a": "}"} y') == {"a": "}"}

    def test_text_with_no_json_at_all_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="no valid JSON"):
            parse_json_with_fences("I'd rather not.")


class TestVariants:
    def test_a_model_match_beats_a_provider_match(self) -> None:
        by_provider = PromptVariant(id="p", supported_providers=("openai",))
        by_model = PromptVariant(id="m", supported_models=("openai/gpt-4.1-mini",))
        chosen = select_variant(
            [by_provider, by_model], provider="openai", model="gpt-4.1-mini"
        )
        assert chosen.id == "m"

    def test_a_mode_narrows_the_field_before_anything_else(self) -> None:
        default = PromptVariant(id="d", is_default=True)
        strict = PromptVariant(id="s", modes=("strict",))
        assert select_variant([default, strict], provider="openai", model="m", mode="strict").id == (
            "s"
        )

    def test_the_default_is_the_fallback(self) -> None:
        default = PromptVariant(id="d", is_default=True)
        other = PromptVariant(id="o", supported_providers=("google",))
        assert select_variant([other, default], provider="openai", model="m").id == "d"

    def test_no_match_and_no_default_says_what_was_tried(self) -> None:
        with pytest.raises(ValueError, match="no match and no default"):
            select_variant(
                [PromptVariant(id="o", supported_providers=("google",))],
                provider="openai",
                model="m",
            )


class TestSchemaDefaults:
    def test_a_declared_default_is_applied(self) -> None:
        schema = {"type": "object", "properties": {"a": {"default": 1}}}
        assert apply_schema_defaults({}, schema) == {"a": 1}

    def test_a_supplied_value_is_left_alone(self) -> None:
        schema = {"type": "object", "properties": {"a": {"default": 1}}}
        assert apply_schema_defaults({"a": 2}, schema) == {"a": 2}

    def test_an_explicit_none_is_not_treated_as_absent(self) -> None:
        schema = {"type": "object", "properties": {"a": {"default": 1}}}
        assert apply_schema_defaults({"a": None}, schema) == {"a": None}

    def test_a_non_object_schema_contributes_nothing(self) -> None:
        assert apply_schema_defaults({"a": 1}, {"type": "string"}) == {"a": 1}


class TestHookNames:
    def test_a_hook_can_be_subscribed_under_either_spelling(self) -> None:
        # A subscriber who reads `ctx.tool_name` and then has to write
        # `'onInternalToolCallStart'` has been handed the boundary, not spared it.
        bus = HookBus()
        seen: list[Any] = []
        bus.on("on_warning", seen.append)
        bus.emit_sync("onWarning", {"code": "x"})
        assert [w.code for w in seen] == ["x"]

    def test_an_emit_under_the_snake_name_reaches_a_camel_subscriber(self) -> None:
        bus = HookBus()
        seen: list[Any] = []
        bus.on("onWarning", seen.append)
        bus.emit_sync("on_warning", {"code": "x"})
        assert len(seen) == 1

    def test_has_and_off_answer_to_either_spelling(self) -> None:
        bus = HookBus()
        bus.on("onWarning", lambda ctx: None)
        assert bus.has("on_warning") is True
        bus.off("on_warning")
        assert bus.has("onWarning") is False

    def test_a_name_no_hook_is_emitted_under_is_still_refused(self) -> None:
        # The whole reason the catalog is checked: a subscription that reaches
        # nobody looks exactly like nothing happening.
        bus = HookBus()
        with pytest.raises(ValueError, match="unknown hook"):
            bus.on("on_complete", lambda ctx: None)
