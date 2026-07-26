"""State repository — persists OrganismState snapshots with optimistic locking.

Pipeline §11.2: versioned snapshots with append_if_version.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ssa.domain.state import OrganismState


class StateRepository:
    """Versioned state snapshot repository (pipeline §8.2 state_snapshots)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def latest(self) -> OrganismState | None:
        """Return the latest state snapshot, or None if none exists."""
        row = self._conn.execute(
            "SELECT * FROM state_snapshots ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return self._row_to_state(row)

    def append_if_version(
        self, expected_version: int, state: OrganismState, cause_event_id: str
    ) -> OrganismState:
        """Append a new snapshot only if the current version matches expected.

        Pipeline §11.2: optimistic locking.
        """
        current = self.latest()
        current_version = current.version if current else 0

        if current_version != expected_version:
            raise StateVersionConflict(
                f"Expected version {expected_version}, but current is {current_version}"
            )

        import uuid
        snapshot_id = uuid.uuid4().hex
        state_json = json.dumps(state.model_dump(mode="json"))

        self._conn.execute(
            """
            INSERT INTO state_snapshots
                (id, previous_id, cause_event_id, version, state_json, created_at_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                None,  # TODO: link to previous snapshot ID
                cause_event_id,
                state.version,
                state_json,
                state.updated_at_ms,
            ),
        )
        return state

    @staticmethod
    def _row_to_state(row: sqlite3.Row | dict[str, Any]) -> OrganismState:
        data = dict(row)
        state_data = json.loads(data["state_json"])
        return OrganismState(**state_data)


class StateVersionConflict(RuntimeError):
    """Raised when optimistic lock check fails."""


__all__ = ["StateRepository", "StateVersionConflict"]
