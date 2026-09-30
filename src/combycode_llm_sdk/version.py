"""This library's version, as it identifies itself to a server.

The ONE place the number is written. `pyproject.toml` declares the version
dynamic and hatchling reads it from here, and the package re-exports it as
`__version__`, so the wheel, the package and anything that reports a version
cannot disagree about what they are.

It lives in its own module rather than in `__init__.py` so that code deep in
the package can read it without importing the package root, which would be a
cycle. That mattered as soon as something needed it: the MCP client told every
server it ever spoke to that it was version `"0"`, hard-coded, because the real
number was not reachable from there.
"""

from __future__ import annotations

__version__ = "0.1.1"

#: The same value under the name the TypeScript library uses, so the two ports
#: read alike at the call sites that report a version.
SDK_VERSION = __version__
