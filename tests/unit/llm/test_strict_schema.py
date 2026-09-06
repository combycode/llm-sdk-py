"""Translated from `unified-library-ts/tests/unit/llm/strict-schema.test.ts`.

Strict mode is defaulted ON, but only where the provider can honour it.

The library forced `strict: true` on every OpenAI function tool while sending the
schema as written. OpenAI's strict mode requires EVERY property to be listed in
`required`, at every nesting level, and rejects the request outright when one is
not -- `400: 'required' is required to be supplied`. So any tool with an optional
parameter was unusable. Most real MCP tools have one; the DeepWiki server used
throughout the example corpus happens to declare everything required, which is
why nothing caught it.

The two providers constrain DIFFERENT things, and the constraints are disjoint --
each of the shapes below was accepted by one provider and rejected by the other,
measured against the live APIs on 2026-08-16 (claude-haiku-4.5 / gpt-5.4-nano):

    optional property           openai REJECT   anthropic ok
    nested obj, no inner req.   openai REJECT   anthropic ok
    `maximum` / `multipleOf`    openai ok       anthropic REJECT

Hence a per-dialect check rather than one shared notion of "strict-safe".

ONE TRANSPOSITION, stated plainly. The TypeScript drives the "what actually goes
on the wire" describes through `new OpenAIResponsesAdapter(...).buildRequest()`.
Those adapters are not in this port batch. The etalon wire specs ARE vendored
here, and TypeScript's own `spec-differential.test.ts` is what proves a spec
reproduces its adapter -- so the same assertions are driven through
`build_from_spec(resolve_spec(<the adapter's spec id>), ...)`, which runs the
same `wire-transforms` rules the adapter runs. The only stand-in is a no-op
message builder for the spec's `messages`/`input` block: no assertion below ever
looks at the messages.
"""

from __future__ import annotations

from typing import Any

from combycode_llm_sdk.llm.types.schema_utils import StrictSupport, strict_support
from combycode_llm_sdk.llm.wire_transforms import make_registry
from combycode_llm_sdk.wire.inherit import resolve_spec
from combycode_llm_sdk.wire.interpreter import build_from_spec
from combycode_llm_sdk.wire.registry import WIRE_SPECS

# -- the shapes, exactly as measured ----------------------------------------
ALL_REQUIRED: dict[str, Any] = {
    "type": "object",
    "properties": {"q": {"type": "string"}},
    "required": ["q"],
}
OPTIONAL_PROP: dict[str, Any] = {
    "type": "object",
    "properties": {"q": {"type": "string"}, "page": {"type": "number"}},
    "required": ["q"],
}
NESTED_NO_INNER_REQUIRED: dict[str, Any] = {
    "type": "object",
    "properties": {"filter": {"type": "object", "properties": {"since": {"type": "string"}}}},
    "required": ["filter"],
}
WITH_MAXIMUM: dict[str, Any] = {
    "type": "object",
    "properties": {"n": {"type": "number", "maximum": 10}},
    "required": ["n"],
}
#: The shape a generic tool router needs: `input` must accept any tool's arguments.
FREE_FORM_NESTED: dict[str, Any] = {
    "type": "object",
    "properties": {
        "calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "input": {"type": "object", "additionalProperties": True},
                },
                "required": ["name", "input"],
            },
        }
    },
    "required": ["calls"],
}


def req(**over: Any) -> dict[str, Any]:
    """strict-schema.test.ts:69-70."""
    return {"model": "m", "messages": [{"role": "user", "content": "hi"}], **over}


def fn_tool(parameters: dict[str, Any], strict: bool | None = None) -> dict[str, Any]:
    """strict-schema.test.ts:72-78."""
    tool: dict[str, Any] = {
        "type": "function",
        "name": "t",
        "description": "d",
        "parameters": parameters,
    }
    if strict is not None:
        tool["strict"] = strict
    return tool


class _MessageBuilderStub:
    """Stands in for the provider adapters' message builders, which land later.

    NOT a port of anything, and deliberately trivial: the spec's `messages` /
    `input` block has no `when` guard, so the build cannot finish without one --
    but no assertion in this file reads the messages it produces.
    """

    def build_input_items(self, msg: Any, tool_names: Any, notes: Any = None) -> list[Any]:
        return [msg]

    def build_messages(self, msg: Any, notes: Any = None) -> list[Any]:
        return [msg]

    def build_message(
        self, msg: Any, request: Any, cache_last: bool, notes: Any = None
    ) -> Any:
        return msg


_HANDLES = {
    "anthropic": _MessageBuilderStub(),
    "openai_responses": _MessageBuilderStub(),
    "openai_completions": _MessageBuilderStub(),
}


def body(spec_id: str, request: dict[str, Any]) -> dict[str, Any]:
    return build_from_spec(
        resolve_spec(spec_id, WIRE_SPECS), request, make_registry(_HANDLES)
    ).body


# -- the predicate ----------------------------------------------------------


class TestStrictSupportOpenaiDialect:
    """strict-schema.test.ts:82."""

    def test_accepts_a_schema_where_every_property_is_required(self) -> None:
        # strict-schema.test.ts:84
        assert strict_support(ALL_REQUIRED, "openai").ok is True

    def test_rejects_an_optional_property_naming_it(self) -> None:
        # strict-schema.test.ts:88-90
        r = strict_support(OPTIONAL_PROP, "openai")
        assert r.ok is False
        assert r.reason is not None and "page" in r.reason

    def test_rejects_a_nested_object_whose_own_properties_are_not_required(self) -> None:
        # strict-schema.test.ts:94-99. The reason has to point at the nesting level
        # that is actually wrong, or it sends the reader to the wrong part of their
        # schema.
        r = strict_support(NESTED_NO_INNER_REQUIRED, "openai")
        assert r.ok is False
        assert r.reason is not None
        assert "filter" in r.reason
        assert "since" in r.reason

    def test_descends_into_array_items_and_any_of_branches(self) -> None:
        # strict-schema.test.ts:103-111
        assert (
            strict_support(
                {
                    "type": "object",
                    "properties": {"xs": {"type": "array", "items": OPTIONAL_PROP}},
                    "required": ["xs"],
                },
                "openai",
            ).ok
            is False
        )
        assert (
            strict_support(
                {
                    "type": "object",
                    "properties": {"x": {"anyOf": [OPTIONAL_PROP]}},
                    "required": ["x"],
                },
                "openai",
            ).ok
            is False
        )

    def test_does_not_care_about_keywords_that_only_anthropic_refuses(self) -> None:
        # strict-schema.test.ts:115
        assert strict_support(WITH_MAXIMUM, "openai").ok is True

    def test_rejects_a_free_form_object(self) -> None:
        # strict-schema.test.ts:122-124. A generic router tool --
        # `call_tools([{name, input}])` -- needs `input` to accept any shape, and
        # under strict it simply cannot. Found by an experiment building exactly
        # that: the check said yes and the API said 400.
        r = strict_support(FREE_FORM_NESTED, "openai")
        assert r.ok is False
        assert r.reason is not None and "input" in r.reason

    def test_rejects_an_explicit_additional_properties_true(self) -> None:
        # strict-schema.test.ts:128-133
        r = strict_support(
            {
                "type": "object",
                "properties": {"k": {"type": "string"}},
                "required": ["k"],
                "additionalProperties": True,
            },
            "openai",
        )
        assert r.ok is False
        assert r.reason is not None and "additionalProperties" in r.reason

    def test_accepts_a_no_argument_tool_whose_properties_are_empty_but_present(self) -> None:
        # strict-schema.test.ts:139. The case that must NOT be caught by the rule
        # above: `properties: {}` is accepted live; only a MISSING `properties` is
        # not.
        assert strict_support({"type": "object", "properties": {}}, "openai").ok is True


class TestStrictSupportAnthropicDialect:
    """strict-schema.test.ts:143."""

    def test_accepts_optional_properties_which_openai_refuses(self) -> None:
        # strict-schema.test.ts:145-146
        assert strict_support(OPTIONAL_PROP, "anthropic").ok is True
        assert strict_support(NESTED_NO_INNER_REQUIRED, "anthropic").ok is True

    def test_rejects_the_measured_unsupported_keywords(self) -> None:
        # strict-schema.test.ts:150-155
        for key in ["minimum", "maximum", "exclusiveMinimum", "multipleOf", "maxItems"]:
            schema = {
                "type": "object",
                "properties": {"n": {"type": "number", key: 1}},
                "required": ["n"],
            }
            r = strict_support(schema, "anthropic")
            assert r.ok is False
            assert r.reason is not None and key in r.reason

    def test_accepts_the_neighbouring_keywords_that_are_supported(self) -> None:
        # strict-schema.test.ts:161-164. The denylist is asymmetric -- asserting the
        # accepted side keeps a future over-broad "just deny all numeric keywords"
        # from passing quietly.
        for key in ["minItems", "maxLength", "minLength"]:
            schema = {
                "type": "object",
                "properties": {"n": {"type": "string", key: 1}},
                "required": ["n"],
            }
            assert strict_support(schema, "anthropic").ok is True


# -- what actually goes on the wire -----------------------------------------


class TestOpenaiResponsesStrictOnTools:
    """strict-schema.test.ts:170."""

    @staticmethod
    def tools_of(schema: dict[str, Any], strict: bool | None = None) -> dict[str, Any]:
        # strict-schema.test.ts:172-174
        built = body("openai/responses", req(tools=[fn_tool(schema, strict)]))
        tool: dict[str, Any] = built["tools"][0]
        return tool

    def test_keeps_strict_on_when_the_schema_qualifies(self) -> None:
        # strict-schema.test.ts:177
        assert self.tools_of(ALL_REQUIRED)["strict"] is True

    def test_drops_strict_rather_than_sending_a_request_the_api_will_reject(self) -> None:
        # strict-schema.test.ts:182-183. The regression: this was hardcoded `true`
        # and the API rejected the request.
        assert self.tools_of(OPTIONAL_PROP)["strict"] is False
        assert self.tools_of(NESTED_NO_INNER_REQUIRED)["strict"] is False

    def test_still_honours_an_explicit_strict_in_both_directions(self) -> None:
        # strict-schema.test.ts:187-188
        assert self.tools_of(OPTIONAL_PROP, True)["strict"] is True
        assert self.tools_of(ALL_REQUIRED, False)["strict"] is False

    def test_applies_the_same_rule_to_structured_output(self) -> None:
        # strict-schema.test.ts:192-196
        def text_of(schema: dict[str, Any]) -> dict[str, Any]:
            built = body("openai/responses", req(structured={"schema": schema}))
            text: dict[str, Any] = built["text"]
            return text

        assert text_of(ALL_REQUIRED)["format"]["strict"] is True
        assert text_of(OPTIONAL_PROP)["format"]["strict"] is False


class TestOpenaiChatCompletionsStrictOnTools:
    """strict-schema.test.ts:200."""

    @staticmethod
    def fn_of(schema: dict[str, Any], strict: bool | None = None) -> dict[str, Any]:
        # strict-schema.test.ts:202-207
        built = body("openai/chat-completions", req(tools=[fn_tool(schema, strict)]))
        fn: dict[str, Any] = built["tools"][0]["function"]
        return fn

    def test_does_not_ask_for_strict_unless_the_caller_does(self) -> None:
        # strict-schema.test.ts:213-215. Opt-in on this API, as it always was.
        # Briefly defaulted on for consistency with Responses, then reverted:
        # strict changes nothing measurable about argument quality (40/40
        # conformant either way, both providers) and only adds exposure.
        fn = self.fn_of(ALL_REQUIRED)
        assert ("strict" in fn) is False
        assert fn["parameters"] == ALL_REQUIRED

    def test_conforms_the_schema_when_strict_is_asked_for(self) -> None:
        # strict-schema.test.ts:219-223. Strict without `additionalProperties:
        # false` is rejected, so opting in has to conform the schema or it fails
        # the very request it opted into.
        fn = self.fn_of(ALL_REQUIRED, True)
        assert fn["strict"] is True
        assert fn["parameters"]["additionalProperties"] is False


class TestAnthropicStrictOnTools:
    """strict-schema.test.ts:227."""

    @staticmethod
    def tool_of(schema: dict[str, Any], strict: bool | None = None) -> dict[str, Any]:
        # strict-schema.test.ts:229-232
        built = body(
            "anthropic/messages@4.7", req(tools=[fn_tool(schema, strict)], maxTokens=16)
        )
        tool: dict[str, Any] = built["tools"][0]
        return tool

    def test_does_not_ask_for_strict_unless_the_caller_does(self) -> None:
        # strict-schema.test.ts:240-243. Strict is OPT-IN here. It was briefly
        # defaulted on -- it is the only thing that stops this model calling a tool
        # that was never declared (10/10 -> 0/10) -- but that benefit only applies
        # when something puts an undeclared tool in front of the model, and the
        # cost turned out to be limits no per-schema check can predict.
        assert ("strict" in self.tool_of(ALL_REQUIRED)) is False
        assert ("strict" in self.tool_of(OPTIONAL_PROP)) is False
        # The schema goes out exactly as written when strict is not requested.
        assert self.tool_of(WITH_MAXIMUM)["input_schema"] == WITH_MAXIMUM

    def test_honours_an_explicit_strict_and_conforms_the_schema_when_it_does(self) -> None:
        # strict-schema.test.ts:247-249
        assert self.tool_of(ALL_REQUIRED, True)["strict"] is True
        assert self.tool_of(ALL_REQUIRED, True)["input_schema"]["additionalProperties"] is False
        assert ("strict" in self.tool_of(ALL_REQUIRED, False)) is False

    def test_refuses_an_open_object_under_strict_as_openai_does(self) -> None:
        # strict-schema.test.ts:253-255
        assert strict_support(FREE_FORM_NESTED, "anthropic").ok is False
        # ...but unlike OpenAI it accepts an object that simply declares no
        # properties.
        assert strict_support({"type": "object"}, "anthropic") == StrictSupport(ok=True)


class TestAnthropicStrictStaysOptInBecauseItsLimitsAreUnpredictable:
    """strict-schema.test.ts:277.

    Three limits, measured live 2026-08-16, none of which a per-SCHEMA predicate
    can see, because two are aggregates over the whole request and the third has
    no published formula at all:

      20 strict tools    21 answers "Too many strict tools (21)"
      24 optional params summed across all strict schemas, nested ones included;
                         25 answers "too many optional parameters (25) ... limit: 24"
      complexity         those same 24 optional params in ONE tool instead of four
                         answers "Schema is too complex for compilation"

    Non-strict tools count toward none of them. Twelve ordinary tools with five
    optional parameters each already exceed the second, so defaulting strict on
    broke realistic tool sets -- found when a benchmark of 12 such tools stopped
    working. The first two could be counted; the third cannot be predicted, which
    is what makes opt-in the only honest default here rather than merely the safer
    one.
    """

    @staticmethod
    def many_tools(n: int, strict: bool | None = None) -> list[dict[str, Any]]:
        # strict-schema.test.ts:279-280
        return [{**fn_tool(ALL_REQUIRED, strict), "name": f"op_{i}"} for i in range(n)]

    @staticmethod
    def strict_count(tools: list[dict[str, Any]]) -> int:
        # strict-schema.test.ts:281-284
        built = body("anthropic/messages@4.7", req(tools=tools, maxTokens=16))
        return len([t for t in built["tools"] if t.get("strict") is True])

    def test_asks_for_strict_on_no_tool_by_default_at_any_tool_count(self) -> None:
        # strict-schema.test.ts:287-288
        assert self.strict_count(self.many_tools(1)) == 0
        assert self.strict_count(self.many_tools(60)) == 0

    def test_sends_exactly_the_strict_tools_the_caller_asked_for_and_no_more(self) -> None:
        # strict-schema.test.ts:294-295. Past the provider's limits this is a 400 --
        # about a schema the caller chose, which is the difference between their
        # decision and our default.
        assert self.strict_count(self.many_tools(21, True)) == 21
        off = [{**t, "name": f"off_{i}"} for i, t in enumerate(self.many_tools(40, False))]
        assert self.strict_count([*self.many_tools(5, True), *off]) == 5
