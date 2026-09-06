"""`lazy=True` registers a tool without spending context declaring it.

Tool definitions go out in full on every request. Three MCP servers and the tool
block dominates the context before the conversation starts -- a cost paid every
turn, and eventually a context-window failure rather than a bill.

A lazy tool is registered but not declared. The model finds it through the
built-in `tool_search`, which returns the full schema as data, and then calls it
normally. The saving is the tool block; the cost is one extra round trip when a
lazy tool is actually needed.

Deterministic: inspects what WOULD be sent, sends nothing.
"""

from _check import check, report

from combycode_llm_sdk import Agent, tool


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return "sunny"


@tool(lazy=True)
def rebuild_search_index(namespace: str, force: bool = False) -> str:
    """Rebuild the search index for a namespace. Slow and rarely needed."""
    return "queued"


agent = Agent(model="openai/gpt-4.1", api_key="k", tools=[get_weather, rebuild_search_index])

declared = agent.declared_tools()
names = {t["name"] for t in declared}

# The eager tool is declared; the lazy one is not.
check("get_weather" in names, "an eager tool must be declared")
check("rebuild_search_index" not in names, "a lazy tool must NOT be declared")

# ...but the model is given a way to find it, or lazy would just mean "missing".
check("tool_search" in names, "tool_search must be declared when any tool is lazy")

# And it is still callable once discovered -- registered, merely undeclared.
check(agent.has_tool("rebuild_search_index"), "a lazy tool must remain callable")

report(declared=sorted(names), lazy=["rebuild_search_index"])
