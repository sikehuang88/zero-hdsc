"""Persistence for structured appraisal records."""

from __future__ import annotations

import json
import sqlite3

from ssa.domain.appraisal import AppraisalResult, StoredAppraisal


class SqliteAppraisalRepository:
    """Stores and retrieves immutable appraisal audit records."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert(self, appraisal: StoredAppraisal) -> StoredAppraisal:
        self._conn.execute(
            """
            INSERT INTO appraisals (
                id, correlation_id, cause_event_id, result_json,
                provider, model, prompt_version, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                appraisal.id,
                appraisal.correlation_id,
                appraisal.cause_event_id,
                appraisal.result.model_dump_json(),
                appraisal.provider,
                appraisal.model,
                appraisal.prompt_version,
                appraisal.created_at_ms,
            ),
        )
        return appraisal

    def get(self, appraisal_id: str) -> StoredAppraisal | None:
        row = self._conn.execute(
            "SELECT * FROM appraisals WHERE id = ?", (appraisal_id,)
        ).fetchone()
        return self._row_to_appraisal(row) if row is not None else None

    def find_by_correlation(self, correlation_id: str) -> list[StoredAppraisal]:
        rows = self._conn.execute(
            "SELECT * FROM appraisals WHERE correlation_id = ? ORDER BY created_at_ms, id",
            (correlation_id,),
        ).fetchall()
        return [self._row_to_appraisal(row) for row in rows]

    @staticmethod
    def _row_to_appraisal(row: sqlite3.Row) -> StoredAppraisal:
        payload = json.loads(str(row["result_json"]))
        return StoredAppraisal(
            id=str(row["id"]),
            correlation_id=str(row["correlation_id"]),
            cause_event_id=str(row["cause_event_id"]),
            result=AppraisalResult.model_validate(payload),
            provider=str(row["provider"]),
            model=str(row["model"]),
            prompt_version=str(row["prompt_version"]),
            created_at_ms=int(row["created_at_ms"]),
        )


__all__ = ["SqliteAppraisalRepository"]
