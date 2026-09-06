"""A remote MCP server's tools, handed to a model like any others.

The point of the bridge is that there is nothing to learn: `mcp.tools()` returns
the same objects `@tool` produces, so they go into `tools=` exactly as local
ones do -- and nothing downstream knows the tool runs on someone else's server.

DeepWiki speaks Streamable HTTP, which is where the comparison gets interesting.
The official SDKs split three ways here: OpenAI, Anthropic and xAI take the URL
and call DeepWiki THEMSELVES, so the model provider becomes the MCP client;
Google needs a separately installed MCP SDK wrapped by hand; OpenRouter has no
hosted MCP at all and needs the whole tool loop written out. Here it is one
`connect_mcp()` with no extra dependency, and the same five lines for all five
providers -- which is the claim this corpus exists to check.

The stdio transport and the protocol underneath it are exercised by 29b.
"""

from _bench import api_key, bench, model

from combycode_llm_sdk import complete, connect_mcp

#: Public and unauthenticated. The catalog names this server so both corpora
#: ask the same question of the same thing.
DEEPWIKI = "https://mcp.deepwiki.com/mcp"


def main() -> str:
    with connect_mcp(url=DEEPWIKI, timeout=120) as mcp:
        result = complete(
            model=model(),
            api_key=api_key(),
            prompt=(
                "What transport protocols does the modelcontextprotocol/typescript-sdk "
                "support? Use the DeepWiki MCP server."
            ),
            tools=mcp.tools(),
            max_tokens=1024,
        )
    return result.text.strip()


bench(main)
