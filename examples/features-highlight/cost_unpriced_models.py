"""A cost of $0.00 that means "I could not price this".

A model the catalog does not know still CALLS perfectly well -- a fine-tune, a
private deployment, a model released this morning. What it cannot do is be
priced. If that entry were summed at zero like any other, the report would read
$0.00 and look like a cheap run rather than an unpriced one, and a budget built
on that total would never fire.

(The original trigger for this was an ALIAS the catalog did not carry:
`anthropic/claude-haiku-4-5` called fine while the catalog keyed it
`claude-haiku-4.5`. That specific gap was closed in 3.1.0 -- the alias now
prices -- so this example uses a genuinely unknown model instead. The shape of
the bug is what matters, not the one id that first exposed it.)

Not hypothetical. A live benchmark reported $0.00000 per task across every
Anthropic run and looked like the cheapest arm in the table until someone
checked; a client later reported 72k tokens billed at $0.00.

So `CostSummary` counts what it could not price and NAMES the models, and the
collector warns once per model -- not once per request, which trains you to
ignore it. Genuinely free calls are not counted: they are priced at zero with a
note, which is a different thing and stays quiet.

In Python the API makes it harder still to misread: `entry.cost` is None when
unpriced, never 0.0.

No API key and no network: a stub transport returns a fixed Anthropic response,
so the cost pipeline runs exactly as it would live.
"""

from _check import check, report

from combycode_llm_sdk import Engine, TransportResponse, complete

STUB_BODY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000},
}


def stub(request):
    return TransportResponse(status=200, body=STUB_BODY)


warnings: list[object] = []
engine = Engine(catalog="defaults", api_keys={"anthropic": "unused-by-the-stub"}, transport=stub)


@engine.on_warning
def collect(ctx) -> None:
    warnings.append(ctx)


# A catalog key: priced normally.
complete(model="anthropic/claude-haiku-4.5", prompt="Hello", engine=engine)

# Reaches the provider, but is NOT in the catalog. Twice, to show the warning
# does not repeat per request.
complete(model="anthropic/claude-private-tune", prompt="Hello", engine=engine)
complete(model="anthropic/claude-private-tune", prompt="Hello", engine=engine)

costs = engine.cost.total()

# The total is real -- it is simply not complete, and `unpriced` says so.
check(costs.entries == 3, f"expected 3 entries, got {costs.entries}")
check(costs.unpriced == 2, f"expected 2 unpriced entries, got {costs.unpriced}")
check(
    any("claude-private-tune" in m for m in costs.unpriced_models),
    f"the unpriced model should be named, got {costs.unpriced_models}",
)
check(costs.total > 0, "the priced call must still contribute to the total")

unpriced_warnings = [w for w in warnings if w.code == "unpriced_model"]
check(len(unpriced_warnings) == 1, f"expected 1 warning, got {len(unpriced_warnings)}")

# The check worth copying into a real application.
if costs.unpriced:
    print(f"WARNING: {costs.unpriced} entries unpriced ({', '.join(costs.unpriced_models)})")

report(
    entries=costs.entries,
    unpriced=costs.unpriced,
    unpriced_models=costs.unpriced_models,
    total=round(costs.total, 6),
)
