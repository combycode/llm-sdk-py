"""The protocol underneath the tool bridge.

Listing is PAGED -- a server with 18 tools works either way, one with 120
silently offers a third of them if the cursor is not followed.

Listing is also CACHED, and the cache is invalidated by the SERVER saying so
rather than by a clock, because only the server knows when its tool set moved.
That is not hypothetical: `server-everything` announces
`notifications/tools/list_changed` on its own, so this run usually observes a
real invalidation rather than a simulated one. Asserting "the second list is
cached" would therefore be asserting that the server stayed still -- which is
exactly the assumption a time-based cache makes and gets wrong.
"""

from _bench import bench

from combycode_llm_sdk import connect_mcp

SERVER = ["-y", "@modelcontextprotocol/server-everything"]


def main() -> str:
    # No model and no key: this exercises the protocol, not a provider.
    announced: list[str] = []
    with connect_mcp(command="npx", args=SERVER,
                     on_notification=lambda m, _p: announced.append(m)) as mcp:
        tools = mcp.list_tools()

        # A tool that fails comes back as a RESULT carrying isError, not as an
        # exception: the model asked for it, and being told is what lets it try
        # something else.
        echoed = mcp.call_tool("echo", {"message": "hi"})
        ok = not echoed.get("isError")

        # Dropping the cache forces the next list back onto the wire.
        mcp.invalidate_cache("tools/list")
        refetched = mcp.list_tools() is not tools

        name = mcp.server_info.get("serverInfo", {}).get("name", "?")

    return f"{name}|tools:{len(tools)}|echo:{ok}|refetched:{refetched}|notified:{len(announced)}"


bench(main)
