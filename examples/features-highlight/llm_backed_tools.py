"""A tool whose implementation is a prompt and a schema, not a function body.

`define_llm_tool` takes a declaration -- instructions, a user template, the
shape of the answer -- and returns an executable tool the `InternalToolRunner`
resolves by id out of a `ToolRegistry`. The prompt stays DATA: readable on
`tool.definition`, versioned in the id, swappable per model without code.

Written as a function instead, every such tool would restate the same six steps
-- render, pick a model, call, strip the fence, parse, check the shape --
around its one paragraph of prompt, and the paragraph that mattered would be
the smallest thing in the file.

The declared `output_schema` is not decoration. It is shown to the model, and
the answer is checked against it: a tool that promised an object and returned a
list warns rather than handing that quietly to whoever reads it. Warns and not
raises, because the answer is already paid for.

Deterministic: stub transport, no keys, no network.
"""

from typing import Any

from _check import check, report

from combycode_llm_sdk import (
    InternalToolRunner,
    ToolRegistry,
    TransportResponse,
    define_llm_tool,
)
from combycode_llm_sdk.hooks import HookBus
from combycode_llm_sdk.internal_tools import (
    JSON_FORMAT,
    InternalToolError,
    InternalToolRunnerConfig,
    LLMToolDefinition,
    LocalBackend,
    ModelPreference,
)

TOOL_ID = "orxa:sentiment@1.0.0"
MODEL = "openai/gpt-4.1-mini"
MAX_TOKENS = 200

MESSAGE = "This is the third time I have written about this order."

sentiment = define_llm_tool(
    LLMToolDefinition(
        id=TOOL_ID,
        namespace="orxa",
        name="sentiment",
        version="1.0.0",
        description="Judge how a customer message reads.",
        input_schema={
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                # A default, so the template never renders a hole for the many
                # callers who leave it out.
                "audience": {"type": "string", "default": "support"},
            },
            "required": ["content"],
        },
        model_preference=ModelPreference(preferred_model=MODEL, max_tokens=MAX_TOKENS),
        output_schema={
            "type": "object",
            "properties": {"label": {"type": "string"}, "confidence": {"type": "number"}},
            "required": ["label", "confidence"],
        },
        output_format=JSON_FORMAT,
        system_prompt="Judge the sentiment of a message written to {{audience}}.",
        user_template="{{content}}",
    )
)


class Model:
    """One scripted answer, and every request it was given."""

    def __init__(self, text: str) -> None:
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


def runner_for(model: Model, hooks: HookBus | None = None) -> InternalToolRunner:
    registry = ToolRegistry().add_backend(LocalBackend().register(sentiment))
    return InternalToolRunner(
        InternalToolRunnerConfig(
            registry=registry,
            api_keys={"openai": "k"},
            # The seam that makes a tool whose body is a model call testable at
            # all: everything above it is the real path.
            transport=model,
            hooks=hooks,
        )
    )


answering = Model('{"label": "angry", "confidence": 0.91}')
output = runner_for(answering).run(TOOL_ID, {"content": MESSAGE})
check(output == {"label": "angry", "confidence": 0.91}, "a JSON tool is read back as a value")

# The declaration is what reached the wire, not an approximation of it.
sent = answering.requests[-1].body
check(sent["input"][0]["content"] == MESSAGE, "the user template renders into the request")
check("Judge the sentiment of a message written to support." in sent["instructions"],
      "the schema default fills the system template before it renders")
check(sent["instructions"].startswith("You are a JSON API endpoint."),
      "a json tool gets the format contract, and the tool never restates it")
check("## Output schema (your JSON must match)" in sent["instructions"],
      "the declared output schema is shown to the model, not merely checked after")
check(sent["max_output_tokens"] == MAX_TOKENS, "the tool's model preference reaches the request")

# The prompt stays inspectable, which is what lets a bench harness re-render one
# without paying to run it.
check(sentiment.definition is not None and sentiment.definition.user_template == "{{content}}",
      "the declaration survives on the tool")

# A model that answered with another shape entirely: the value still comes back,
# and the mismatch is reported rather than absorbed.
hooks = HookBus()
warnings: list[Any] = []
hooks.on("on_warning", warnings.append)
wrong_shape = runner_for(Model('[{"label": "angry"}]'), hooks).run(TOOL_ID, {"content": MESSAGE})
check(wrong_shape == [{"label": "angry"}], "a paid answer is never thrown away over a schema")
check([w.code for w in warnings] == ["output_schema_mismatch"],
      "an answer that does not match the declared schema must be reported")

# The input schema is enforced before any of that costs anything.
blocked = Model("never asked")
try:
    runner_for(blocked).run(TOOL_ID, {"audience": "billing"})
    refused = False
except InternalToolError:
    refused = True
check(refused, "a missing required input is refused")
check(blocked.requests == [], "and refused before the request is sent, not after")

report(tool=TOOL_ID, output=output, warnings=[w.code for w in warnings])
