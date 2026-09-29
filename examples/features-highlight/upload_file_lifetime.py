"""Asking the provider to delete an uploaded file for you.

Uploaded files do not clean themselves up. OpenAI states that everything but
`purpose=batch` persists until manually deleted, so an agent that attaches a
document per turn grows an unbounded pile on the customer's account -- and
nothing in the happy path ever mentions it.

`upload_lifetime_seconds` is set once, on the registry, and every upload carries
it. The three providers that accept one each want it in a different shape, which
the adapters own; the fourth cannot take one at all and says so rather than
pretending. That is the part worth seeing here: a unified option that silently
does nothing on one provider is worse than one that is honest about where it
applies.

Deterministic: the adapter is a stub that records what the registry handed it.
"""

from __future__ import annotations

from typing import Any

from _check import check, report

from combycode_llm_sdk import Engine
from combycode_llm_sdk.files import (
    Base64Content,
    FilesRegistry,
    FileUploadOptions,
    FileUploadResult,
)

engine = Engine(catalog="defaults", api_keys={"openai": "k"}, register_as_default=False)


class RecordingAdapter:
    """A file store that records the options rather than talking to anyone."""

    name = "stub"
    expires_after_ms = None
    max_file_size = 100_000_000
    supported_types = None

    def __init__(self, seen: dict[str, Any]) -> None:
        self._seen = seen

    def upload(
        self, file: Any, fetch: Any, opts: FileUploadOptions | None = None
    ) -> FileUploadResult:
        self._seen["opts"] = opts
        # A provider that cannot honour the request says so through `warn`, which
        # the registry turns into an on_warning with code `request_adjusted`.
        # Google is the real case: its `expiration_time` is "Output only".
        if opts is not None and opts.lifetime_seconds is not None and opts.warn is not None:
            opts.warn(
                f"this store cannot set a lifetime; {opts.lifetime_seconds}s was not sent",
                {"requestedLifetimeSeconds": opts.lifetime_seconds},
            )
        return FileUploadResult(remote_id="file_stub_1")

    def delete(self, remote_id: str, fetch: Any) -> None: ...

    def get_info(self, remote_id: str, fetch: Any) -> None:
        return None

    def list(self, fetch: Any) -> list[Any]:
        return []


def upload_with(lifetime: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seen: dict[str, Any] = {}
    adjusted: list[dict[str, Any]] = []

    def on_warning(w: Any) -> None:
        if dict(w).get("code") == "request_adjusted":
            adjusted.append(dict(w))

    off = engine.hooks.on("onWarning", on_warning)
    files = FilesRegistry(
        hooks=engine.hooks,
        catalog=engine.catalog,
        fetch=engine.fetch,
        upload_lifetime_seconds=lifetime,
    )
    files.register_provider("stub", RecordingAdapter(seen))
    attachment = files.add(
        filename="report.txt",
        mime_type="text/plain",
        content=Base64Content(mime_type="text/plain", data="aGVsbG8="),
    )
    files.upload(attachment.id, "stub")
    off()
    files.close()
    return seen, adjusted


# -- 1. configured once, carried on every upload -----------------------------
an_hour_seen, an_hour_adjusted = upload_with(3600)
check(an_hour_seen["opts"].lifetime_seconds == 3600,
      "the configured lifetime must reach the adapter")

# -- 2. off by default, which is the providers' own default ------------------
# An uploaded file otherwise lives until something deletes it -- so NOT asking
# has to mean not sending the field, never sending a zero or a default.
none_seen, none_adjusted = upload_with()
check(none_seen["opts"].lifetime_seconds is None,
      "an unconfigured registry must not invent a lifetime")
check(not none_adjusted, "nothing was asked for, so nothing was adjusted")

# -- 3. what cannot be honoured is said out loud -----------------------------
check(len(an_hour_adjusted) == 1, "a store that cannot honour it must warn exactly once")
warning = an_hour_adjusted[0]["message"] if an_hour_adjusted else ""
check("3600" in warning, "the warning should name the lifetime that was dropped")

report(
    configured=an_hour_seen["opts"].lifetime_seconds,
    unconfigured=none_seen["opts"].lifetime_seconds,
    warned_when_asked=len(an_hour_adjusted),
    warned_when_not_asked=len(none_adjusted),
    warning=warning,
)
