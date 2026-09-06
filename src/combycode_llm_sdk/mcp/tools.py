"""An MCP tool, as a tool this library already knows how to run.

By the time one of these reaches an agent it is an ordinary `Tool` -- which is
the whole point. Lazy loading, the permission checks, the collision policy and
the reports all apply to a server's tools without any of them knowing that a
subprocess is involved.

The name is namespaced (`<server>__<tool>`) because two servers may both publish
`search`, and the model must be able to say which one it means. Execution uses
the RAW name: the namespace is ours, not the server's.

Transposed from `unified-library-ts/src/plugins/mcp/tools.ts`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..helpers.tool import Tool
from ..util.json_schema import DEFAULT_ERROR_LIMIT, validate_json_schema
from ..wire.interpreter import js_json
from .client import McpClient

_UNSAFE = re.compile(r"[^a-zA-Z0-9_]")


def sanitize_namespace(name: str) -> str:
    """A namespace a provider will accept as part of a tool name.

    Unsafe characters become underscores rather than being deleted as the
    TypeScript does: `my server/v2` reads as `my_server_v2` instead of
    `myserverv2`, and the namespace is a thing a person has to recognise in a
    tool name and in a log line.

    Which means the fallback cannot test the result for emptiness -- `!!!`
    substitutes to `___`, which is truthy and says nothing. It tests for an
    actual character instead.
    """
    cleaned = _UNSAFE.sub("_", name)
    return cleaned if any(c.isalnum() for c in cleaned) else "mcp"


def _block_to_part(block: Mapping[str, Any]) -> dict[str, Any] | None:
    """One MCP content block as one of our content parts, or None if unknown."""
    kind = block.get("type")
    if kind == "text":
        return {"type": "text", "text": str(block.get("text") or "")}
    if kind in ("image", "audio"):
        return {
            "type": kind,
            "source": {
                "type": "base64",
                "mimeType": str(block.get("mimeType") or ""),
                "data": str(block.get("data") or ""),
            },
        }
    if kind == "resource":
        resource = block.get("resource")
        if isinstance(resource, Mapping):
            if resource.get("text"):
                return {"type": "text", "text": str(resource["text"])}
            if resource.get("uri"):
                return {"type": "text", "text": f"[resource {resource['uri']}]"}
    if kind == "resource_link" and block.get("uri"):
        return {"type": "text", "text": f"[resource {block['uri']}]"}
    return None


def mcp_content_to_result(result: Mapping[str, Any]) -> str | list[dict[str, Any]]:
    """A `tools/call` result as a tool return value.

    Plain text when the content is text-only, which is what almost every tool
    returns and what a model reads best; a part list only when there is media to
    preserve. A tool-level `isError` becomes text the model can act on rather
    than an exception it never sees.
    """
    parts: list[dict[str, Any]] = []
    has_media = False
    content = result.get("content")
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        for block in content:
            if not isinstance(block, Mapping):
                continue
            part = _block_to_part(block)
            if part is None:
                continue
            if part["type"] != "text":
                has_media = True
            parts.append(part)
    if has_media:
        return parts
    text = "".join(p["text"] for p in parts if p["type"] == "text")
    return f"Tool error: {text}" if result.get("isError") else text


def mcp_prompt_to_messages(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A `prompts/get` result as messages, ready to drop into a request."""
    messages: list[dict[str, Any]] = []
    raw = result.get("messages")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return messages
    for message in raw:
        if not isinstance(message, Mapping):
            continue
        block = message.get("content")
        part = _block_to_part(block) if isinstance(block, Mapping) else None
        if part is None:
            content: Any = ""
        elif part["type"] == "text":
            content = part["text"]
        else:
            content = [part]
        messages.append({"role": message.get("role", "user"), "content": content})
    return messages


def mcp_tool_to_tool(
    client: McpClient,
    definition: Mapping[str, Any],
    namespace: str,
    *,
    lazy: bool = False,
    validate_output: bool = False,
) -> Tool:
    """Wrap one MCP tool definition as a `Tool` the agent loop can run.

    `validate_output` opts into the server's `outputSchema` as a CONTRACT rather
    than a hint, and it is off by default because saying yes changes what the
    tool returns. Declaring the schema to a provider makes the provider require
    a JSON result matching it -- OpenAI Responses rejects the turn outright with
    "expected a JSON string because the function declares output_schema" if the
    tool answers in prose. So the schema, the validation, and returning
    `structuredContent` instead of rendered text travel together or not at all;
    forwarding the schema alone would silently reshape every existing tool
    result.
    """
    raw_name = str(definition.get("name") or "")
    description = str(
        definition.get("description") or definition.get("title") or raw_name or "an MCP tool"
    )
    parameters = definition.get("inputSchema")
    if not isinstance(parameters, Mapping):
        parameters = {"type": "object", "properties": {}}

    output_schema = definition.get("outputSchema")
    declares_output = validate_output and isinstance(output_schema, Mapping)
    if not isinstance(output_schema, Mapping):
        output_schema = {}

    # `taskSupport: "required"` means the server will refuse a plain
    # `tools/call` -- the work outlives one request, so it has to be started as
    # a task and collected afterwards. Honoured here rather than left to the
    # caller: a tool that says how it must be invoked and is then invoked the
    # other way is a server error we already know how to avoid.
    execution = definition.get("execution")
    as_task = (
        isinstance(execution, Mapping) and execution.get("taskSupport") == "required"
    )

    def run_as_task(arguments: Mapping[str, Any]) -> str | list[dict[str, Any]]:
        task = client.call_tool_task(raw_name, arguments)
        task_id = str(task.get("taskId") or "")
        finished = client.await_task(task_id)
        status = str(finished.get("status") or "")
        if status != "completed":
            # The MODEL is the audience for this, exactly as for `isError`: a
            # task that was cancelled or failed is a disappointing answer, not
            # a broken connection, and the model can often work around it.
            message = str(finished.get("statusMessage") or status or "unknown")
            return mcp_content_to_result(
                {
                    "content": [{"type": "text", "text": f"task {status or 'ended'}: {message}"}],
                    "isError": True,
                }
            )
        # Through the same finish as a plain call: the output contract is a
        # property of the TOOL, and a task-only tool that answered in prose
        # would break the promise its own declaration made to the provider.
        return _finish(client.get_task_result(task_id))

    def _finish(result: Mapping[str, Any]) -> str | list[dict[str, Any]]:
        return _validated(result) if declares_output else mcp_content_to_result(result)

    def call(**arguments: Any) -> str | list[dict[str, Any]]:
        if as_task:
            return run_as_task(arguments)
        return _finish(client.call_tool(raw_name, arguments))

    def _validated(result: Mapping[str, Any]) -> str | list[dict[str, Any]]:
        """The structured half of a result, once it has been checked.

        MCP sends `structuredContent` exactly when a tool publishes an
        `outputSchema`, so a missing one means this call had nothing structured
        to give and the rendered content is the answer.
        """
        structured = result.get("structuredContent")
        if structured is None:
            return mcp_content_to_result(result)

        errors = validate_json_schema(output_schema, structured)
        if errors:
            # Returned to the MODEL rather than raised: a server that broke its
            # own contract is a disappointing answer, not a broken connection,
            # and a model told what was wrong will often call again correctly.
            shown = "; ".join(errors[:DEFAULT_ERROR_LIMIT])
            return f"Tool output failed schema validation: {shown}"
        if result.get("isError"):
            return mcp_content_to_result(result)
        return js_json(structured)

    # The loop reads `func.__name__` when a tool is registered, and a closure is
    # called `call` unless told otherwise -- which would make every MCP tool in a
    # fleet share one name in the logs.
    call.__name__ = f"{namespace}__{raw_name}"
    call.__doc__ = description

    return Tool(
        call,
        {
            **({"outputSchema": dict(output_schema)} if declares_output else {}),
            "name": f"{namespace}__{raw_name}",
            "description": description,
            "parameters": dict(parameters),
        },
        lazy=lazy,
    )


__all__ = [
    "mcp_content_to_result",
    "mcp_prompt_to_messages",
    "mcp_tool_to_tool",
    "sanitize_namespace",
]
