"""A real MCP server over stdio, for tests that should not mock the protocol.

Small enough to read in one sitting and real enough to be worth running: it
speaks newline-delimited JSON-RPC on stdin/stdout and implements the handshake,
`tools/list` (paginated), `tools/call`, resources, prompts and notifications.

Mocking a transport would prove the client can talk to our idea of a server.
This proves it can talk to one -- through a pipe, across a process boundary,
with real buffering and real interleaving.

Modes (argv[1]) exist so a test can ask for the awkward case instead of waiting
for it:

    normal    the ordinary server: eight tools, one page
    paged     `tools/list` answers in two pages, so pagination is exercised
    noisy     writes a banner and a log line to STDOUT before any message
    slow      delays every response past a short timeout
    erroring  `tools/call` answers with `isError`, which is not an exception
    modern    speaks the 2026-07-28 wire: `server/discover`, no handshake
    stubborn  ignores EOF on stdin, so closing the pipe does not end it
    listless  connects fine and then refuses `tools/list`
    caching   sends the 2026 `ttlMs` hint on `tools/list`

Every mode answers `server/discover` with a method-not-found error except
`modern`, so a connection to any of the others exercises the fallback path the
auto negotiator takes against every server that exists today.

`modern` also speaks the three subsystems that only exist at 2026-07-28 --
`subscriptions/listen`, tasks, and `input_required` -- because none of them can
be exercised on the handshake wire at all.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

MODE = sys.argv[1] if len(sys.argv) > 1 else "normal"

PROTOCOL_VERSION = "2025-11-25"
MODERN_VERSION = "2026-07-28"

METHOD_NOT_FOUND = -32601

#: Every reply the client sent to one of THIS server's requests, readable back
#: through `fixture/replies`.
REPLIES: list[dict[str, Any]] = []

#: Open `subscriptions/listen` requests, by their JSON-RPC id. A listen request
#: is never answered while the subscription lives -- its response IS the
#: end-of-stream signal -- so the id is held here until `fixture/end_listen`.
LISTENERS: dict[Any, dict[str, Any]] = {}

#: Tasks by id, for the `tasks/*` methods.
TASKS: dict[str, dict[str, Any]] = {}

#: How many `tasks/get` polls each task still needs before it finishes. A task
#: that completed on the first poll would let a broken polling loop pass.
TASK_POLLS_LEFT: dict[str, int] = {}

#: How many pings this server has answered. A keep-alive is invisible from the
#: client side -- it succeeds silently -- so the only place to observe one is
#: the far end.
PINGS = [0]

#: How many times each resource has actually been read off the wire. The only
#: way to prove a cached read did NOT reach the server: the answer is identical
#: either way, so the client cannot tell them apart from the outside.
READS: dict[str, int] = {}

#: `requestState` values this server has handed out, so a test can prove the
#: token came back byte-exact rather than being rebuilt.
STATES_SEEN: list[str] = []

#: Enough tools that declaring them all is visibly the wrong default, which is
#: the point the lazy-loading example is making.
TOOLS: list[dict[str, Any]] = [
    {
        "name": "add",
        "description": "Add two numbers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
    {
        "name": "echo",
        "description": "Return the text it was given.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "search_docs",
        "description": "Search the documentation for a phrase.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from the workspace.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "list_issues",
        "description": "List open issues in the tracker.",
        "inputSchema": {"type": "object", "properties": {"label": {"type": "string"}}},
    },
    {
        "name": "picture",
        "description": "Return an image, to exercise a non-text content block.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "structured",
        "description": "Return a structured result beside its text.",
        "inputSchema": {"type": "object", "properties": {}},
        "outputSchema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
    },
    {
        "name": "no_such_thing",
        "description": "Always fails, so a tool-level error can be observed.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

#: Declares that it MUST be run as a task. Only offered on the modern wire,
#: since tasks do not exist before it.
TASK_ONLY_TOOLS = [
    {
        "name": "long_job",
        "description": "Runs too long for one request, so it has to be a task.",
        "inputSchema": {"type": "object", "properties": {}},
        "execution": {"taskSupport": "required"},
    },
    {
        "name": "doomed_job",
        "description": "A task that always fails, so the unhappy path is real.",
        "inputSchema": {"type": "object", "properties": {}},
        "execution": {"taskSupport": "required"},
    },
]


def send(message: dict[str, Any]) -> None:
    """Write one message. Binary, so no platform rewrites the delimiter."""
    sys.stdout.buffer.write(json.dumps(message).encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()


def reply(request_id: Any, result: Any) -> None:
    if MODE == "slow":
        time.sleep(1.5)
    send({"jsonrpc": "2.0", "id": request_id, "result": result})


def fail(request_id: Any, code: int, message: str, data: Any = None) -> None:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    send({"jsonrpc": "2.0", "id": request_id, "error": error})


def input_required_result(
    name: str, params: dict[str, Any]
) -> dict[str, Any] | None:
    """The `input_required` leg for a tool that asks before it answers.

    None when this call is not one -- either the tool never asks, or the answers
    have already come back and the call can now be completed.
    """
    responses = params.get("inputResponses")
    state = params.get("requestState")
    if state is not None:
        STATES_SEEN.append(str(state))

    if name == "needs_input":
        if responses:
            answer: Any = next(iter(responses.values()), {})
            text = ""
            if isinstance(answer, dict):
                content = answer.get("content")
                text = content.get("text", "") if isinstance(content, dict) else ""
            return {"content": [{"type": "text", "text": f"the model said: {text}"}]}
        return {
            "resultType": "input_required",
            "requestState": "state-token-1",
            "inputRequests": {
                "q1": {
                    "method": "sampling/createMessage",
                    "params": {
                        "messages": [
                            {"role": "user", "content": {"type": "text", "text": "guess"}}
                        ],
                        "maxTokens": 16,
                    },
                }
            },
        }

    if name == "slow_work":
        # State-only legs: questions come later, or never. The server is saying
        # "still working, ask again", which is what the driver backs off on.
        round_number = int(str(state or "round-0").rsplit("-", 1)[-1])
        if round_number >= 2:
            return {"content": [{"type": "text", "text": f"done after {round_number} rounds"}]}
        return {"resultType": "input_required", "requestState": f"round-{round_number + 1}"}

    if name == "never_satisfied":
        # Always asks again, however it is answered -- the loop that would run
        # forever without a round cap.
        return {
            "resultType": "input_required",
            "requestState": "endless",
            "inputRequests": {"q": {"method": "sampling/createMessage", "params": {}}},
        }
    return None


def call_result(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if MODE == "erroring" or name == "no_such_thing":
        return {"content": [{"type": "text", "text": f"{name} is unavailable"}], "isError": True}
    if name == "add":
        total = arguments.get("a", 0) + arguments.get("b", 0)
        return {"content": [{"type": "text", "text": str(total)}]}
    if name == "echo":
        return {"content": [{"type": "text", "text": str(arguments.get("text", ""))}]}
    if name == "picture":
        return {
            "content": [
                {"type": "text", "text": "here it is"},
                {"type": "image", "mimeType": "image/png", "data": "iVBORw0KGgo="},
            ]
        }
    if name == "structured":
        return {
            "content": [{"type": "text", "text": '{"ok": true}'}],
            "structuredContent": {"ok": True},
        }
    return {"content": [{"type": "text", "text": f"{name}({json.dumps(arguments)})"}]}


def handle(message: dict[str, Any]) -> None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}

    if method == "server/discover":
        if MODE != "modern":
            fail(request_id, METHOD_NOT_FOUND, "server/discover is not supported")
            return
        reply(
            request_id,
            {
                "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                "instructions": "the modern fixture",
                "_meta": {
                    "io.modelcontextprotocol/serverInfo": {
                        "name": "fixture",
                        "version": "0.0.1",
                    }
                },
            },
        )
        return

    if method == "initialize":
        reply(
            request_id,
            {
                "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                "capabilities": {"tools": {}, "resources": {}, "prompts": {}, "logging": {}},
                "serverInfo": {"name": "fixture", "version": "0.0.1"},
            },
        )
        return

    if method == "notifications/initialized":
        return

    if method == "tools/list":
        if MODE == "listless":
            # Connects, then refuses the list -- so the failure lands AFTER the
            # handshake, which is the path where a child gets orphaned.
            fail(request_id, -32603, "the tool index is offline")
            return
        if MODE == "paged":
            cursor = params.get("cursor")
            if not cursor:
                reply(request_id, {"tools": TOOLS[:3], "nextCursor": "page-2"})
            else:
                reply(request_id, {"tools": TOOLS[3:]})
            return
        if MODE == "caching":
            reply(request_id, {"tools": TOOLS, "ttlMs": 60000, "cacheScope": "private"})
            return
        reply(
            request_id,
            {"tools": [*TOOLS, *TASK_ONLY_TOOLS] if MODE == "modern" else TOOLS},
        )
        return

    if method == "tools/call":
        name = str(params.get("name"))
        if MODE == "modern" and "task" in params:
            if name == "task_less":
                # Asked for a task, answers without one. There is then nothing
                # to poll, and a client that shrugs leaves the caller holding a
                # result that never arrives.
                reply(request_id, {})
                return
            task_id = f"task-{len(TASKS) + 1}"
            TASKS[task_id] = {
                "taskId": task_id,
                "status": "working",
                "ttl": params.get("task", {}).get("ttl"),
                "createdAt": "2026-07-28T00:00:00Z",
                "lastUpdatedAt": "2026-07-28T00:00:00Z",
                # Milliseconds on the wire. Small so the test is not a sleep,
                # but non-zero so a client that ignores it is still correct.
                "pollInterval": 10,
                "_name": name,
                "_arguments": dict(params.get("arguments") or {}),
            }
            # `never_satisfied` never leaves `working`, so a client that
            # polls without a deadline hangs -- which is the thing the timeout
            # exists to turn into an error.
            TASK_POLLS_LEFT[task_id] = 10**9 if name == "never_satisfied" else 2
            reply(request_id, {"task": {k: v for k, v in TASKS[task_id].items()
                                        if not k.startswith("_")}})
            return
        if MODE == "modern":
            asked = input_required_result(name, params)
            if asked is not None:
                reply(request_id, asked)
                return
        reply(request_id, call_result(name, dict(params.get("arguments") or {})))
        return

    if method in ("resources/subscribe", "resources/unsubscribe"):
        reply(request_id, {})
        return

    if method == "completion/complete":
        if (params.get("ref") or {}).get("name") == "silent":
            # No `completion` key at all -- which is not the same shape as an
            # empty list, and is what a server sends when it has nothing.
            reply(request_id, {})
            return
        value = str((params.get("argument") or {}).get("value", ""))
        reply(
            request_id,
            {"completion": {"values": [f"{value}-one", f"{value}-two"], "hasMore": False}},
        )
        return

    # `subscriptions/listen` is answered only when the subscription ENDS, so the
    # id is parked and the ack goes out as a stream frame instead.
    if method == "subscriptions/listen":
        wanted = dict(params.get("notifications") or {})
        LISTENERS[request_id] = wanted
        # Deliberately NARROWER than any request that asked for prompts: the
        # honoured subset can be smaller than the ask, and a client that assumes
        # otherwise waits forever for events that were never granted.
        honored = {k: v for k, v in wanted.items() if k != "promptsListChanged"}
        send(
            {
                "jsonrpc": "2.0",
                "method": "notifications/subscriptions/acknowledged",
                "params": {
                    "notifications": honored,
                    "_meta": {"io.modelcontextprotocol/subscriptionId": request_id},
                },
            }
        )
        return

    if method == "tasks/get":
        task_id = str(params.get("taskId"))
        task = TASKS.get(task_id)
        if task is None:
            fail(request_id, -32602, f"no such task: {task_id}")
            return
        left = TASK_POLLS_LEFT.get(task_id, 0)
        if left > 0:
            TASK_POLLS_LEFT[task_id] = left - 1
        elif task["status"] == "working":
            if task["_name"] == "doomed_job":
                task["status"] = "failed"
                task["statusMessage"] = "the worker died"
            else:
                task["status"] = "completed"
        reply(request_id, {k: v for k, v in task.items() if not k.startswith("_")})
        return

    if method == "tasks/result":
        task_id = str(params.get("taskId"))
        task = TASKS.get(task_id)
        if task is None:
            fail(request_id, -32602, f"no such task: {task_id}")
            return
        reply(request_id, call_result(str(task["_name"]), dict(task["_arguments"])))
        return

    if method == "tasks/list":
        reply(
            request_id,
            {
                "tasks": [
                    {k: v for k, v in t.items() if not k.startswith("_")}
                    for t in TASKS.values()
                ]
            },
        )
        return

    if method == "tasks/cancel":
        task_id = str(params.get("taskId"))
        if task_id in TASKS:
            TASKS[task_id]["status"] = "cancelled"
            TASK_POLLS_LEFT[task_id] = 0
        reply(request_id, {})
        return

    if method == "resources/list":
        reply(
            request_id,
            {"resources": [{"uri": "mem://greeting", "name": "greeting", "mimeType": "text/plain"}]},
        )
        return

    if method == "resources/read":
        uri = str(params.get("uri"))
        READS[uri] = READS.get(uri, 0) + 1
        body: dict[str, Any] = {
            "contents": [
                {
                    "uri": params.get("uri"),
                    "mimeType": "text/plain",
                    "text": f"hello from {params.get('uri')}",
                }
            ]
        }
        if MODE == "caching":
            # The 2026 freshness hint. Only sent in this mode, so every other
            # mode still proves the "no hint, no caching" rule.
            body["ttlMs"] = 60000
            body["cacheScope"] = "private"
        reply(request_id, body)
        return

    if method == "prompts/list":
        reply(request_id, {"prompts": [{"name": "greet", "description": "Say hello."}]})
        return

    if method == "prompts/get":
        reply(
            request_id,
            {
                "messages": [
                    {"role": "user", "content": {"type": "text", "text": "Say hello politely."}}
                ]
            },
        )
        return

    if method == "ping":
        PINGS[0] += 1
        reply(request_id, {})
        return

    if method == "fixture/pings":
        reply(request_id, {"pings": PINGS[0]})
        return

    # `fixture/notify` asks the server to push a notification, so the client's
    # server->client path can be observed without waiting for a real event.
    if method == "fixture/notify":
        send(
            {
                "jsonrpc": "2.0",
                "method": "notifications/message",
                "params": {"level": "info", "data": params.get("data", "hello")},
            }
        )
        reply(request_id, {})
        return

    # `fixture/ask` makes the server send US a request, which is the direction
    # that only a real duplex connection can exercise.
    if method == "fixture/ask":
        send({"jsonrpc": "2.0", "id": 9001, "method": "roots/list", "params": {}})
        reply(request_id, {})
        return

    # What came back to the server's own requests. Without this a test can see
    # that we SENT a server request but not whether the client ever answered it
    # -- and "answered" is the whole property: a dropped one leaves the server
    # blocked forever.
    if method == "fixture/replies":
        reply(request_id, {"replies": REPLIES})
        return

    # Push one listen-stream frame, stamped with the subscription it belongs to.
    # Without the stamp a client cannot attribute a frame at all, which is
    # exactly the failure a second open subscription would expose.
    if method == "fixture/push_event":
        # Naming a subscription sends exactly ONE frame, to that one. Otherwise
        # every open subscription gets its own -- and the difference is what
        # lets a test prove a frame did NOT reach the other stream.
        named = params.get("subscriptionId")
        targets = [named] if named is not None else list(LISTENERS)
        for target in targets:
            send(
                {
                    "jsonrpc": "2.0",
                    "method": str(params.get("event", "notifications/tools/list_changed")),
                    "params": {
                        **({"uri": params["uri"]} if "uri" in params else {}),
                        "_meta": {"io.modelcontextprotocol/subscriptionId": target},
                    },
                }
            )
        reply(request_id, {})
        return

    # End every open subscription by answering its parked listen request. This
    # is the ONLY thing that ends a stream cleanly, and a client that ignores it
    # keeps a handle that will never deliver again.
    if method == "fixture/end_listen":
        for listen_id in list(LISTENERS):
            if params.get("error"):
                fail(listen_id, -32000, str(params["error"]))
            else:
                send({"jsonrpc": "2.0", "id": listen_id, "result": {}})
            del LISTENERS[listen_id]
        reply(request_id, {})
        return

    # The `requestState` tokens this server handed out, so a test can prove they
    # came back byte-exact rather than being reconstructed.
    if method == "fixture/reads":
        reply(request_id, {"reads": READS})
        return

    if method == "fixture/states":
        reply(request_id, {"states": STATES_SEEN})
        return

    # What the child can actually see of the parent's environment. Asserting on
    # `safe_env()` in-process only checks the function; this checks the process.
    if method == "fixture/env":
        reply(request_id, {"names": sorted(os.environ)})
        return

    if request_id is not None:
        fail(request_id, METHOD_NOT_FOUND, f"unknown method: {method}")


def main() -> None:
    if MODE == "noisy":
        # A banner on stdout is a real thing servers do, and a client that
        # cannot survive it fails on a server that works everywhere else.
        sys.stdout.buffer.write(b"fixture server starting...\n")
        sys.stdout.buffer.write(b"[warn] not JSON\n")
        sys.stdout.buffer.flush()

    buffer = b""
    while True:
        # `os.read`, not `.read(n)`: a buffered read waits for the full count
        # or EOF, which on a request/response pipe is a deadlock.
        chunk = os.read(sys.stdin.fileno(), 4096)
        if not chunk:
            # `stubborn` ignores EOF, the way a server with its own event loop
            # and no stdin handling does. Closing stdin is not enough to end one
            # of those, which is what makes the escalation in `close()` real.
            if MODE == "stubborn":
                time.sleep(60)
                continue
            return
        buffer += chunk
        while True:
            index = buffer.find(b"\n")
            if index < 0:
                break
            line, buffer = buffer[:index], buffer[index + 1 :]
            text = line.decode("utf-8", "replace").strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except ValueError:
                continue
            if isinstance(message, dict):
                # A reply to one of OUR requests carries an id and no method.
                if message.get("method") is None and message.get("id") is not None:
                    REPLIES.append(
                        {
                            "id": message.get("id"),
                            "ok": "result" in message,
                            "error": (message.get("error") or {}).get("message"),
                        }
                    )
                    continue
                handle(message)


if __name__ == "__main__":
    main()
