"""Persistence for versioned relationship snapshots."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ssa.domain.relationship import RelationshipState
from ssa.ids import IdGenerator


class RelationshipRepository:
    """SQLite relationship snapshots with atomic optimistic locking."""

    def __init__(self, conn: sqlite3.Connection, ids: IdGenerator) -> None:
        self._conn = conn
        self._ids = ids

    def latest(self) -> RelationshipState | None:
        """Return the highest relationship version, if one exists."""
        row = self._conn.execute(
            "SELECT * FROM relationship_snapshots ORDER BY version DESC LIMIT 1"
        ).fetchone()
        return self._row_to_relationship(row) if row is not None else None

    def get_by_version(self, version: int) -> RelationshipState | None:
        """Return one exact version, if present."""
        row = self._conn.execute(
            "SELECT * FROM relationship_snapshots WHERE version = ?", (version,)
        ).fetchone()
        return self._row_to_relationship(row) if row is not None else None

    def append_if_version(
        self,
        expected_version: int,
        relationship: RelationshipState,
        cause_event_id: str,
    ) -> RelationshipState:
        """Append only when the current version equals ``expected_version``.

        The version check, predecessor lookup, and insert are one SQLite
        statement. Competing writers therefore cannot both append the same
        version after checking stale state in application code.
        """
        required_version = expected_version + 1
        if expected_version < 0:
            raise ValueError("expected_version must be non-negative")
        if relationship.version != required_version:
            raise ValueError(
                f"Relationship version must be {required_version}, got {relationship.version}"
            )
        if not cause_event_id or not cause_event_id.strip():
            raise ValueError("cause_event_id must not be empty")

        snapshot_id = self._ids.new()
        payload = relationship.model_dump_json()
        cursor = self._conn.execute(
            """
            INSERT INTO relationship_snapshots (
                id, previous_id, cause_event_id, version,
                relationship_json, created_at_ms
            )
            SELECT
                ?,
                (SELECT id FROM relationship_snapshots ORDER BY version DESC LIMIT 1),
                ?, ?, ?, ?
            WHERE COALESCE(
                (SELECT MAX(version) FROM relationship_snapshots),
                0
            ) = ?
            """,
            (
                snapshot_id,
                cause_event_id,
                relationship.version,
                payload,
                relationship.updated_at_ms,
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            current = self.latest()
            current_version = current.version if current is not None else 0
            raise RelationshipVersionConflict(
                f"Expected version {expected_version}, but current is {current_version}"
            )
        return relationship

    @staticmethod
    def _row_to_relationship(
        row: sqlite3.Row | dict[str, Any],
    ) -> RelationshipState:
        data = dict(row)
        payload = json.loads(str(data["relationship_json"]))
        return RelationshipState.model_validate(payload)


class RelationshipVersionConflict(RuntimeError):
    """Raised when a relationship snapshot optimistic lock is stale."""


# Both names are exported so callers can be explicit about the backend while
# preserving the concise repository naming used by StateRepository.
SqliteRelationshipRepository = RelationshipRepository


__all__ = [
    "RelationshipRepository",
    "RelationshipVersionConflict",
    "SqliteRelationshipRepository",
]
