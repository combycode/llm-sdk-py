"""Runtime detection.

PARTIAL transposition of `unified-library-ts/src/runtime/runtime.ts`, and
deliberately so: that file is one detection function plus six lazy loaders for
`node:fs`, `node:path` and `node:child_process`. The loaders exist because a
browser bundler cannot resolve `node:*` builtins, so no module on a reachable
import path may statically import one. Python has no browser target and no
bundler, and `os`, `pathlib` and `subprocess` are always importable -- so the
loaders have nothing to defer and are not ported. Their call sites import the
stdlib directly.

`is_browser` IS ported, because it is not really a bundling concern: it decides
whether the Anthropic adapter sends `anthropic-dangerous-direct-browser-access`,
and that decision has to exist in every port so the header logic reads the same
everywhere.
"""

from __future__ import annotations


def is_browser() -> bool:
    """True when running inside a browser (DOM present).

    Always False here: CPython has no DOM. It is a function rather than a
    constant so the shape matches the TypeScript and a call site reads
    identically across the ports -- and so that this is the ONE place to change
    if the library is ever targeted at an in-browser Python runtime, where
    whether requests go through the browser's CORS-checked `fetch` would have to
    be established rather than assumed.
    """
    return False


__all__ = ["is_browser"]
