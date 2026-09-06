"""Deferred work, with no timer and no thread.

The TypeScript `Scheduler` owns `setTimeout` handles and fires tasks itself.
This one does not, and the difference is deliberate: a library that starts a
background thread has started one in every process that imports it, whether or
not that process wanted one. It cannot be shut down by a caller who does not
know it exists, and it cannot be tested without sleeping.

So this scheduler ANSWERS QUESTIONS. `due()` says what should have fired by now
and `run_due()` fires it; whoever owns the program's loop decides when to ask.
A caller who wants a thread can start one in four lines and own it.

The schedule is DATA in a persistence store rather than state on the instance,
which is what lets a process that dies with an hour left on a task pick it up on
the next start. Handlers are bound by NAME because a name is all a persisted
task can carry -- a function cannot be written to disk.

Transposed from `unified-library-ts/src/plugins/scheduler/scheduler.ts`.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

#: The prefix every task key carries in the store, so one store can hold more
#: than the schedule.
TASK_PREFIX = "task:"

_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h|d)\s*$")

_UNITS_MS = {"ms": 1.0, "s": 1_000.0, "m": 60_000.0, "h": 3_600_000.0, "d": 86_400_000.0}


def parse_duration(value: str | float) -> float:
    """`"30s"`, `"5m"`, `"1h"`, `"2d"` or a raw number, as MILLISECONDS.

    Milliseconds because that is what the store holds and what `Date.now()`
    gives the TypeScript; a scheduler whose durations and whose clock disagreed
    about units would be wrong by a factor of a thousand and still look
    plausible in every test that used only one of them.

    A number goes through `str()` like everything else. Special-casing it read
    as a fast path and was in fact indistinguishable -- `float(str(250))` is
    `250.0` -- except for `True`, which `isinstance(x, int)` accepts and which
    would then have meant one millisecond.
    """
    match = _DURATION.match(str(value))
    if match:
        return float(match.group(1)) * _UNITS_MS[match.group(2)]
    try:
        return float(str(value).strip())
    except ValueError:
        raise ValueError(
            f"invalid duration {value!r}. Use 30s, 5m, 1h, 2d, or a number of milliseconds."
        ) from None


@dataclass
class ScheduledTask:
    """One entry in the schedule, exactly as it is stored."""

    id: str
    name: str
    args: Mapping[str, Any] = field(default_factory=dict)
    fire_at: float = 0.0
    #: `"once"` is deleted after it runs; `"periodic"` is rescheduled.
    type: str = "once"
    interval: float | None = None

    def as_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "args": dict(self.args),
            "fireAt": self.fire_at,
            "type": self.type,
            "interval": self.interval,
        }

    @staticmethod
    def of(row: Mapping[str, Any]) -> ScheduledTask:
        return ScheduledTask(
            id=str(row.get("id") or ""),
            name=str(row.get("name") or ""),
            args=dict(row.get("args") or {}),
            fire_at=float(row.get("fireAt") or 0.0),
            type=str(row.get("type") or "once"),
            interval=(
                float(row["interval"]) if row.get("interval") not in (None, "") else None
            ),
        )


def _now_ms() -> float:
    """Wall-clock milliseconds -- the unit the store holds."""
    return time.time() * 1000


class Scheduler:
    """A schedule kept in a store, and asked rather than awaited."""

    def __init__(
        self,
        persistence: Any,
        *,
        clock: Callable[[], float] = _now_ms,
        prefix: str = TASK_PREFIX,
    ) -> None:
        self._store = persistence
        self._clock = clock
        self._prefix = prefix
        # Handlers live on the INSTANCE and the schedule lives in the store, so
        # a restart re-registers the functions and finds the schedule waiting.
        self._handlers: dict[str, Callable[[Mapping[str, Any]], Any]] = {}

    # -- registration --------------------------------------------------------

    def register(self, name: str, handler: Callable[[Mapping[str, Any]], Any]) -> None:
        """Bind a handler to the name tasks refer to it by."""
        self._handlers[name] = handler

    def unregister(self, name: str) -> bool:
        return self._handlers.pop(name, None) is not None

    def handlers(self) -> list[str]:
        return list(self._handlers)

    # -- scheduling ----------------------------------------------------------

    def after(self, duration: str | float, name: str, args: Mapping[str, Any] | None = None) -> str:
        """Run `name` once, this long from now."""
        return self._add(self._clock() + parse_duration(duration), name, args, "once", None)

    def at(self, when: float, name: str, args: Mapping[str, Any] | None = None) -> str:
        """Run `name` once, at this timestamp (in the clock's own units)."""
        return self._add(float(when), name, args, "once", None)

    def every(
        self, interval: str | float, name: str, args: Mapping[str, Any] | None = None
    ) -> str:
        """Run `name` repeatedly, this far apart. First run is one interval away."""
        every = parse_duration(interval)
        return self._add(self._clock() + every, name, args, "periodic", every)

    def cancel(self, task_id: str) -> None:
        self._store.delete(self._key(task_id))

    def _add(
        self,
        fire_at: float,
        name: str,
        args: Mapping[str, Any] | None,
        task_type: str,
        interval: float | None,
    ) -> str:
        task = ScheduledTask(
            id=f"task_{uuid.uuid4().hex[:8]}",
            name=name,
            args=dict(args or {}),
            fire_at=fire_at,
            type=task_type,
            interval=interval,
        )
        self._store.set(self._key(task.id), task.as_row())
        return task.id

    # -- asking --------------------------------------------------------------

    def pending(self) -> list[ScheduledTask]:
        """Everything scheduled, due or not, ordered by when it fires."""
        tasks = [
            ScheduledTask.of(row)
            for row in (self._store.get(key) for key in self._store.list(self._prefix))
            if isinstance(row, Mapping)
        ]
        return sorted(tasks, key=lambda t: t.fire_at)

    def due(self) -> list[ScheduledTask]:
        """What should have fired by now. Reading this changes nothing."""
        now = self._clock()
        return [task for task in self.pending() if task.fire_at <= now]

    def run_due(self) -> int:
        """Fire everything due, and return how many ran.

        A handler that raises does NOT stop the rest. The tasks are unrelated,
        and one failing report must not delay every other job in the schedule --
        which is what letting the exception out mid-loop would do, one tick at a
        time, for as long as it kept failing.

        Nor is it swallowed: every failure is collected and raised together at
        the end, so a caller sees all of them and none of them silently. The
        failing task is still retired, because a periodic task that re-fires
        instantly on failure is a hot loop.
        """
        ran = 0
        failures: list[Exception] = []
        for task in self.due():
            handler = self._handlers.get(task.name)
            if handler is None:
                # No handler REGISTERED is not the same as no handler existing:
                # this process may simply not be the one that runs this task, so
                # the entry is left in the store for one that does.
                continue
            try:
                handler(dict(task.args))
            except Exception as exc:  # noqa: BLE001 -- a handler is caller code and
                # may raise anything; collected so the next task still runs.
                failures.append(exc)
            finally:
                self._retire(task)
            ran += 1
        if failures:
            raise ExceptionGroup(
                f"{len(failures)} scheduled task(s) failed", failures
            )
        return ran

    def _retire(self, task: ScheduledTask) -> None:
        if task.type == "periodic" and task.interval:
            task.fire_at = self._clock() + task.interval
            self._store.set(self._key(task.id), task.as_row())
        else:
            self._store.delete(self._key(task.id))

    def _key(self, task_id: str) -> str:
        return f"{self._prefix}{task_id}"

    def __len__(self) -> int:
        return len(self._store.list(self._prefix))

    def __repr__(self) -> str:
        return f"<Scheduler {len(self)} pending, {len(self._handlers)} handler(s)>"


__all__ = ["TASK_PREFIX", "ScheduledTask", "Scheduler", "parse_duration"]
