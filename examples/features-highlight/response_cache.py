"""A cached answer belongs to a request AND to where that request was sent.

`Cache` is content-agnostic: it stores a body under `(cache_name, cache_key)`
with a TTL. `request_cache_key` is the half that turns a request into that key,
and it takes `provider` as a REQUIRED argument with no default.

That is the part worth reading twice. A normalized request holds wire fields
only -- by the time it exists the provider prefix has been stripped off the
model id -- so `openai/gpt-5` and `openrouter/openai/gpt-5` arrive here
indistinguishable. They are different services, with different moderation and
different answers. Without the route in the key they hash the same, and one
caller is handed the other service's completion: no error, no warning, just a
wrong answer that looks exactly like a right one. `base_url` is the same
argument one level down, for the same model behind an Azure deployment.

Deterministic: a counting stub transport, no API key, no network. The on-disk
half runs in a temporary directory.
"""

import tempfile

from _check import check, report

from combycode_llm_sdk import (
    LLM,
    Cache,
    FilePersistence,
    MemoryCacheStore,
    PersistentCacheStore,
    TransportResponse,
)
from combycode_llm_sdk.cache import CacheEntry, make_storage_key, request_cache_key

MESSAGES = [{"role": "user", "content": "What is the capital of France?"}]

#: The request as it goes on the wire: no provider anywhere in it.
REQUEST = {"model": "gpt-5.4-nano", "input": MESSAGES, "temperature": 0}

MODELS = {"openai": "openai/gpt-5.4-nano", "openrouter": "openrouter/openai/gpt-5.4-nano"}
AZURE_URL = "https://contoso.openai.azure.com"

RESPONSES_BODY = {
    "id": "resp_1",
    "output": [{"type": "message", "content": [{"type": "output_text", "text": "Paris."}]}],
    "usage": {"input_tokens": 5, "output_tokens": 2},
}
CHAT_BODY = {
    "id": "chatcmpl_1",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Paris, France."}}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
}

calls: list[str] = []


def stub(request):
    calls.append(request.url)
    body = CHAT_BODY if "openrouter" in request.url else RESPONSES_BODY
    return TransportResponse(status=200, body=body)


cache = Cache(store=MemoryCacheStore())


def ask(provider: str) -> str:
    """Complete through the cache, keyed by the request and by the route."""
    key = request_cache_key(REQUEST, provider=provider)
    hit = cache.get(None, key)
    if hit is not None:
        return hit
    text = LLM(model=MODELS[provider], api_key="k", transport=stub).complete(MESSAGES).text
    cache.set(None, key, text)
    return text


direct = request_cache_key(REQUEST, provider="openai")
resold = request_cache_key(REQUEST, provider="openrouter")
azure = request_cache_key(REQUEST, provider="openai", base_url=AZURE_URL)

check(direct != resold, "the same request to two services must not share one key")
check(direct != azure, "nor the same service at two base urls")

reordered = {"temperature": 0, "input": MESSAGES, "model": REQUEST["model"]}
check(
    request_cache_key(reordered, provider="openai") == direct,
    "the same request written in another order is the same request",
)

first = ask("openai")
repeated = ask("openai")
other_service = ask("openrouter")

check(repeated == first, f"an identical request must be served from the cache, got {repeated!r}")
check(len(calls) == 2, f"the cached call must not reach the transport, saw {len(calls)} calls")
# The one that costs something: with the route missing from the key this is
# `first`, returned from the cache, and nothing anywhere says it is wrong.
check(other_service != first, f"another service must get its own answer, got {other_service!r}")

# Expiry is checked on READ and the entry dropped there, so backdating one entry
# is the whole of it -- no sweeper thread, and nothing here has to sleep.
stale = Cache(store=MemoryCacheStore())
stale.store.set(
    make_storage_key("default", direct),
    CacheEntry(body="answered in 1970", stored_at=0.0, ttl_ms=60_000, cache_name="default"),
)
check(stale.get(None, direct) is None, "an entry past its TTL must not be served")
check(stale.store.keys() == [], "and the read that found it expired must drop it")

with tempfile.TemporaryDirectory() as directory:
    Cache(store=PersistentCacheStore(FilePersistence(directory))).set("tenant-7", direct, first)

    # A new process: new objects, same directory.
    reopened = Cache(store=PersistentCacheStore(FilePersistence(directory)))
    check(reopened.get("tenant-7", direct) == first, "an entry on disk outlives the process")
    check(reopened.get("tenant-9", direct) is None, "another namespace must not see it")
    check(reopened.invalidate(cache_name="tenant-9") == 0, "invalidate is scoped to its namespace")
    check(reopened.invalidate(cache_name="tenant-7") == 1, "and drops the entries that are its own")

report(transport_calls=len(calls), openai=first, openrouter=other_service, key=direct[:16])
