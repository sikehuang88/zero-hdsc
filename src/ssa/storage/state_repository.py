"""State repository — persists OrganismState snapshots with optimistic locking.

Pipeline §11.2: versioned snapshots with append_if_version.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ssa.domain.state import OrganismState
from ssa.ids import IdGenerator


class StateRepository:
    """Versioned state snapshot repository (pipeline §8.2 state_snapshots)."""

    def __init__(self, conn: sqlite3.Connection, ids: IdGenerator) -> None:
        self._conn = conn
        self._ids = ids

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
        required_version = expected_version + 1
        if state.version != required_version:
            raise ValueError(f"State version must be {required_version}, got {state.version}")

        snapshot_id = self._ids.new()
        state_json = json.dumps(state.model_dump(mode="json"))

        cursor = self._conn.execute(
            """
            INSERT INTO state_snapshots
                (id, previous_id, cause_event_id, version, state_json, created_at_ms)
            SELECT
                ?,
                (SELECT id FROM state_snapshots ORDER BY version DESC LIMIT 1),
                ?, ?, ?, ?
            WHERE COALESCE(
                (SELECT MAX(version) FROM state_snapshots),
                0
            ) = ?
            """,
            (
                snapshot_id,
                cause_event_id,
                state.version,
                state_json,
                state.updated_at_ms,
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            current = self.latest()
            current_version = current.version if current else 0
            raise StateVersionConflict(
                f"Expected version {expected_version}, but current is {current_version}"
            )
        return state

    def get_by_version(self, version: int) -> OrganismState | None:
        row = self._conn.execute(
            "SELECT * FROM state_snapshots WHERE version = ?", (version,)
        ).fetchone()
        return self._row_to_state(row) if row is not None else None

    @staticmethod
    def _row_to_state(row: sqlite3.Row | dict[str, Any]) -> OrganismState:
        data = dict(row)
        state_data = json.loads(data["state_json"])
        return OrganismState(**state_data)


class StateVersionConflict(RuntimeError):
    """Raised when optimistic lock check fails."""


__all__ = ["StateRepository", "StateVersionConflict"]
