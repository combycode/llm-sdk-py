"""Hosted code execution produces files. They arrive as descriptors, not bytes.

A chart or a CSV can be large, and a provider may hand back an id or a URL
rather than inline data. Eagerly downloading every file would make an innocuous
call slow and occasionally enormous, so `result.files` carries descriptors and
`.read()` is the explicit step that fetches.

`result.files` is always a list -- empty when nothing was produced -- so callers
never None-check the ordinary case.

Deterministic: stub transport.
"""

from _check import check, report

from combycode_llm_sdk import TransportResponse, complete

BODY = {
    "id": "resp_1",
    "output": [
        {"type": "message", "content": [{"type": "output_text", "text": "saved"}]},
        {"type": "code_interpreter_call", "outputs": [
            {"type": "file", "file_id": "file_abc", "filename": "fib.txt", "mime_type": "text/plain"}
        ]},
    ],
    "usage": {"input_tokens": 10, "output_tokens": 3},
}


def stub(request):
    if request.url.endswith("/files/file_abc/content"):
        return TransportResponse(status=200, body=b"1 1 2 3 5")
    return TransportResponse(status=200, body=BODY)


result = complete(
    model="openai/gpt-5.4-nano",
    api_key="k",
    prompt="Compute fibonacci and save it",
    builtin_tools=["code_interpreter"],
    transport=stub,
)

check(len(result.files) == 1, f"expected one produced file, got {len(result.files)}")

descriptor = result.files[0]
check(descriptor.filename == "fib.txt", "the descriptor names the file")
check(descriptor.mime_type == "text/plain", "the descriptor carries its type")

# The fetch is explicit, and only now does a byte cross the network.
content = descriptor.read()
check(content == b"1 1 2 3 5", "read() fetches the actual bytes")

# Saving is the common case, so it is one call rather than an open/write dance.
check(hasattr(descriptor, "save"), "a descriptor should be savable in one call")

report(files=len(result.files), filename=descriptor.filename, bytes=len(content))
