"""ID generation — deterministic in tests, UUID4 in production.

Pipeline §10.3: `IdGenerator` protocol + production/stub impls.
IDs are strings to keep SQLite schemas and JSON exports uniform.
"""

from __future__ import annotations

import uuid
from itertools import count
from typing import Protocol, runtime_checkable


@runtime_checkable
class IdGenerator(Protocol):
    """Produces unique string IDs."""

    def new(self) -> str:
        ...


class UuidIdGenerator:
    """Production generator backed by UUID4."""

    def new(self) -> str:
        return uuid.uuid4().hex


class SequentialIdGenerator:
    """Test generator producing predictable, ordered IDs.

    Output format: `seq-000001`, `seq-000002`, ...
    The prefix is configurable to keep different generators distinguishable
    in multi-fixture scenarios.
    """

    __slots__ = ("_counter", "_prefix")

    def __init__(self, prefix: str = "seq") -> None:
        self._counter = count(start=1)
        self._prefix = prefix

    def new(self) -> str:
        n = next(self._counter)
        return f"{self._prefix}-{n:06d}"


__all__ = ["IdGenerator", "SequentialIdGenerator", "UuidIdGenerator"]
