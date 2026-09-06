"""Lazy loading where it actually pays: a server with a lot of tools.

An MCP server is the usual source of a large tool block, and declaring sixty
tools spends sixty declarations of context on every turn -- most of them for
tools the model will never call. `lazy=True` registers them all and declares
none; the model finds what it needs through `tool_search`.

The mechanic itself is `lazy_tools.py`. What this shows is that it applies to
tools that came from a server exactly as it does to local ones, because by the
time they reach the agent they are the same objects.

Deterministic -- the server is a local fixture, no key and no network.
"""

import sys
from pathlib import Path

from _check import check, report

from combycode_llm_sdk import connect_mcp

FIXTURE = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "mcp_server.py"

with connect_mcp(command=sys.executable, args=[str(FIXTURE), "normal"], lazy=True) as mcp:
    lazy_tools = mcp.tools()
    check(bool(lazy_tools), "the server's tools must be registered")
    check(all(t.lazy for t in lazy_tools), "lazy=True must mark every tool lazy")
    # Registered, not declared: the wire form carries the flag the agent reads.
    check(all(t.to_wire().get("lazy") for t in lazy_tools), "the flag must reach the tool wire form")

    eager = mcp.tools(lazy=False)
    check(not any(t.lazy for t in eager), "lazy is a per-call override too")

    names = sorted(t.name for t in lazy_tools)

report(tools=names, lazy=True, server="fixture")
