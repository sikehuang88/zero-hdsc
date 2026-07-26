"""Clock abstraction — business code never reads system time directly.

Pipeline §10.3: `Clock` protocol + `SystemClock` (production) and
`FrozenClock` (tests). Time is always expressed in UTC milliseconds
to avoid timezone drift inside the database.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """A monotonic-ish source of UTC wall-clock milliseconds."""

    def now_ms(self) -> int:
        """Return the current UTC time as epoch milliseconds."""
        ...


class SystemClock:
    """Production clock backed by `time.time_ns()`."""

    def now_ms(self) -> int:
        return time.time_ns() // 1_000_000


class FrozenClock:
    """Test clock. Time only moves when explicitly advanced.

    Starting value defaults to a non-zero sentinel so that
    `created_at_ms = 0` can be used as an "unset" marker if needed.
    """

    __slots__ = ("_ms",)

    def __init__(self, start_ms: int = 1_700_000_000_000) -> None:
        if start_ms < 0:
            raise ValueError("FrozenClock start_ms must be non-negative")
        self._ms = start_ms

    def now_ms(self) -> int:
        return self._ms

    def advance_ms(self, delta_ms: int) -> int:
        """Move the clock forward by `delta_ms` and return the new value."""
        if delta_ms < 0:
            raise ValueError("advance_ms delta must be non-negative")
        self._ms += delta_ms
        return self._ms

    def set_ms(self, value_ms: int) -> None:
        """Hard-set the clock. Useful for fixture seeding."""
        if value_ms < 0:
            raise ValueError("set_ms value must be non-negative")
        self._ms = value_ms


__all__ = ["Clock", "FrozenClock", "SystemClock"]
