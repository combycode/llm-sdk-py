"""Makes `tests` a package, so `tests.golden_waivers` is the module's ONE name.

Without this file mypy can derive two names for the same file -- `golden_waivers`
from its path and `tests.golden_waivers` from the imports in
`tests/unit/**` -- and refuses to continue:

    error: Source file found twice under different module names

CI runs `mypy src tests`, so that error fails the build. It was invisible locally
for a while because a desktop run of `mypy src` never looks at this directory.

Only files directly in `tests/` are affected: pytest walks up from a test file
while `__init__.py` exists, and the subdirectories deliberately have none, so
module naming under `tests/unit/**` is unchanged.
"""
