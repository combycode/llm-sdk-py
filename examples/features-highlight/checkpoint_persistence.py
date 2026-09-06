"""One key-value interface; where the bytes land is a swap, not a rewrite.

Checkpointing a run, a cache that outlives the process, a schedule of deferred
tasks: all three want `get`/`set`/`delete`/`list`/`has` and differ only in where
the bytes end up. So `Persistence` is a runtime-checkable Protocol -- structural,
not a base class -- and an application's own Redis or SQL store satisfies it
WITHOUT importing anything from this library or inheriting from anything in it.
`RedisLikeStore` below never mentions `Persistence` and is still one.

What that buys: a resume path exercised against `MemoryPersistence` in a test is
the same code that resumes from `FilePersistence` in production. Without it, the
only backend the resume was ever proved against is the one it was written for,
and the first restart in production is the first real run.

Which is also why `MemoryPersistence` copies values in and out. A caller
mutating what it read would otherwise silently edit the checkpoint -- something
the file backend cannot do -- and two backends that disagree about that are not
interchangeable.

Deterministic: no network. The on-disk backend runs in a temporary directory.
"""

import tempfile

from _check import check, report

from combycode_llm_sdk import FilePersistence, MemoryPersistence, Persistence

STEPS = ("fetch", "extract", "summarise")

#: A colon is how keys are namespaced here ("task:...", "cache:default:...") and
#: is not a legal filename character on Windows, so the file backend escapes it.
RUN_KEY = "run:2026-08-30:quarterly-report"


class RedisLikeStore:
    """An application's own backend, standing in for its Redis client.

    Written against nothing: no import from this library, no base class. It is a
    `Persistence` because it has the five methods.
    """

    def __init__(self) -> None:
        self._rows: dict[str, object] = {}

    def get(self, key: str) -> object | None:
        return self._rows.get(key)

    def set(self, key: str, value: object) -> None:
        self._rows[key] = value

    def delete(self, key: str) -> None:
        self._rows.pop(key, None)

    def list(self, prefix: str | None = None) -> list[str]:
        return [k for k in self._rows if prefix is None or k.startswith(prefix)]

    def has(self, key: str) -> bool:
        return key in self._rows


def run_steps(store: Persistence, key: str, *, dies_at: str | None = None) -> list[str]:
    """Do the steps that are not done yet, checkpointing after each one."""
    done = list(store.get(key) or [])
    for step in STEPS:
        if step in done:
            continue
        if step == dies_at:
            raise RuntimeError(f"the process died during {step}")
        done.append(step)
        store.set(key, done)
    return done


def crash_and_resume(store: Persistence) -> tuple[list[str], list[str]]:
    """Run until it dies, then run again against the same store."""
    try:
        run_steps(store, RUN_KEY, dies_at="summarise")
    except RuntimeError:
        pass  # the point is what survives it
    return list(store.get(RUN_KEY)), run_steps(store, RUN_KEY)


with tempfile.TemporaryDirectory() as directory:
    backends = {
        "memory": MemoryPersistence(),
        "file": FilePersistence(directory),
        "application": RedisLikeStore(),
    }

    for name, store in backends.items():
        check(isinstance(store, Persistence), f"{name} must satisfy the protocol structurally")
        partial, resumed = crash_and_resume(store)
        check(partial == ["fetch", "extract"], f"{name} kept the wrong checkpoint: {partial}")
        check(resumed == list(STEPS), f"{name} resumed wrong: {resumed}")
        # Not re-run: the resume above skipped what the checkpoint already had,
        # which is the only reason to write one.
        check(store.has(RUN_KEY), f"{name} must still hold the finished run")

    on_disk = backends["file"]
    check(on_disk.list("run:") == [RUN_KEY], "a key with colons round-trips through a filename")
    on_disk.delete(RUN_KEY)
    check(not on_disk.has(RUN_KEY), "and the same escaping finds the file again to delete it")
    check(on_disk.get(RUN_KEY) is None, "a key that was never written reads as None, not an error")

memory = MemoryPersistence()
memory.set("draft", {"steps": ["fetch"]})
memory.get("draft")["steps"].append("extract")
check(
    memory.get("draft") == {"steps": ["fetch"]},
    "mutating what was read must not edit the checkpoint -- the file backend cannot",
)

report(backends=sorted(backends), steps=list(STEPS), key=RUN_KEY)
