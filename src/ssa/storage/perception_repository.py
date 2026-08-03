"""SQLite persistence for immutable environment-perception snapshots."""

from __future__ import annotations

import json
import sqlite3

from ssa.domain.perception import SituationPerception, StoredPerception


class SqlitePerceptionRepository:
    """Store and reconstruct versioned perception audit records."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def insert(self, perception: StoredPerception) -> StoredPerception:
        self._connection.execute(
            """
            INSERT INTO perception_snapshots (
                id, correlation_id, conversation_id, query_event_id,
                model_id, operator_version, result_json, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                perception.id,
                perception.correlation_id,
                perception.conversation_id,
                perception.query_event_id,
                perception.result.model_id,
                perception.result.operator_version,
                perception.result.model_dump_json(),
                perception.created_at_ms,
            ),
        )
        return perception

    def find_by_query_event(self, query_event_id: str) -> StoredPerception | None:
        row = self._connection.execute(
            "SELECT * FROM perception_snapshots WHERE query_event_id = ?",
            (query_event_id,),
        ).fetchone()
        return self._row_to_perception(row) if row is not None else None

    def latest_by_conversation(self, conversation_id: str) -> StoredPerception | None:
        row = self._connection.execute(
            """
            SELECT * FROM perception_snapshots
            WHERE conversation_id = ?
            ORDER BY created_at_ms DESC, rowid DESC
            LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()
        return self._row_to_perception(row) if row is not None else None

    @staticmethod
    def _row_to_perception(row: sqlite3.Row) -> StoredPerception:
        result = SituationPerception.model_validate(json.loads(str(row["result_json"])))
        return StoredPerception(
            id=str(row["id"]),
            correlation_id=str(row["correlation_id"]),
            conversation_id=str(row["conversation_id"]),
            query_event_id=str(row["query_event_id"]),
            result=result,
            created_at_ms=int(row["created_at_ms"]),
        )


__all__ = ["SqlitePerceptionRepository"]
