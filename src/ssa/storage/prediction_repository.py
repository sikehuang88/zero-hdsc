"""SQLite persistence for machine-verifiable grounded predictions."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ssa.domain.predictions import GroundedPrediction, PredictionStatus


class PredictionRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert(self, prediction: GroundedPrediction) -> GroundedPrediction:
        existing = self.find_by_dedup(prediction.dedup_key)
        if existing is not None:
            return existing
        self._conn.execute(
            """
            INSERT INTO predictions (
                id, conversation_id, dedup_key, claim_text, claim_kind,
                verifier_kind, verifier_spec_json, stated_confidence,
                base_rate_prior, source_event_ids_json, source_trace_ids_json,
                created_at_ms, resolve_after_ms, expires_at_ms, status,
                resolved_at_ms, resolved_by_event_id, brier_contribution
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            self._values(prediction),
        )
        return prediction

    def get(self, prediction_id: str) -> GroundedPrediction | None:
        row = self._conn.execute(
            "SELECT * FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        return self._row_to_prediction(row) if row is not None else None

    def find_by_dedup(self, dedup_key: str) -> GroundedPrediction | None:
        row = self._conn.execute(
            "SELECT * FROM predictions WHERE dedup_key = ?",
            (dedup_key,),
        ).fetchone()
        return self._row_to_prediction(row) if row is not None else None

    def list_due(
        self,
        now_ms: int,
        *,
        conversation_id: str | None = None,
        limit: int = 100,
    ) -> list[GroundedPrediction]:
        if limit < 1:
            raise ValueError("prediction limit must be positive")
        if conversation_id is None:
            rows = self._conn.execute(
                """
                SELECT * FROM predictions
                WHERE status = 'pending' AND resolve_after_ms <= ?
                ORDER BY resolve_after_ms, created_at_ms, rowid
                LIMIT ?
                """,
                (now_ms, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT * FROM predictions
                WHERE conversation_id = ? AND status = 'pending' AND resolve_after_ms <= ?
                ORDER BY resolve_after_ms, created_at_ms, rowid
                LIMIT ?
                """,
                (conversation_id, now_ms, limit),
            ).fetchall()
        return [self._row_to_prediction(row) for row in rows]

    def recent(self, conversation_id: str, *, limit: int = 50) -> list[GroundedPrediction]:
        if limit < 1:
            raise ValueError("prediction limit must be positive")
        rows = self._conn.execute(
            """
            SELECT * FROM predictions
            WHERE conversation_id = ?
            ORDER BY created_at_ms DESC, rowid DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        return [self._row_to_prediction(row) for row in rows]

    def resolved(self, conversation_id: str) -> list[GroundedPrediction]:
        rows = self._conn.execute(
            """
            SELECT * FROM predictions
            WHERE conversation_id = ? AND status IN ('resolved_true', 'resolved_false')
            ORDER BY resolved_at_ms, rowid
            """,
            (conversation_id,),
        ).fetchall()
        return [self._row_to_prediction(row) for row in rows]

    def status_counts(self, conversation_id: str) -> dict[PredictionStatus, int]:
        rows = self._conn.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM predictions
            WHERE conversation_id = ?
            GROUP BY status
            """,
            (conversation_id,),
        ).fetchall()
        return {PredictionStatus(row["status"]): int(row["count"]) for row in rows}

    def resolve(
        self,
        prediction: GroundedPrediction,
        *,
        status: PredictionStatus,
        resolved_at_ms: int,
        resolved_by_event_id: str | None = None,
    ) -> GroundedPrediction:
        if prediction.status != PredictionStatus.PENDING:
            return prediction
        if status not in {
            PredictionStatus.RESOLVED_TRUE,
            PredictionStatus.RESOLVED_FALSE,
            PredictionStatus.EXPIRED,
            PredictionStatus.UNVERIFIABLE,
        }:
            raise ValueError("prediction resolution requires a terminal status")
        outcome = 1.0 if status == PredictionStatus.RESOLVED_TRUE else 0.0
        brier = (
            (prediction.stated_confidence - outcome) ** 2
            if status in {PredictionStatus.RESOLVED_TRUE, PredictionStatus.RESOLVED_FALSE}
            else None
        )
        cursor = self._conn.execute(
            """
            UPDATE predictions SET
                status = ?, resolved_at_ms = ?, resolved_by_event_id = ?,
                brier_contribution = ?
            WHERE id = ? AND status = 'pending'
            """,
            (status.value, resolved_at_ms, resolved_by_event_id, brier, prediction.id),
        )
        current = self.get(prediction.id)
        if current is None:
            raise RuntimeError(f"prediction {prediction.id!r} disappeared during resolution")
        if cursor.rowcount == 0 and current.status == PredictionStatus.PENDING:
            raise RuntimeError(f"prediction {prediction.id!r} changed concurrently")
        return current

    @staticmethod
    def _values(prediction: GroundedPrediction) -> tuple[object, ...]:
        return (
            prediction.id,
            prediction.conversation_id,
            prediction.dedup_key,
            prediction.claim_text,
            prediction.claim_kind.value,
            prediction.verifier_kind.value,
            json.dumps(prediction.verifier_spec, ensure_ascii=False, sort_keys=True),
            prediction.stated_confidence,
            prediction.base_rate_prior,
            json.dumps(prediction.source_event_ids, ensure_ascii=False),
            json.dumps(prediction.source_trace_ids, ensure_ascii=False),
            prediction.created_at_ms,
            prediction.resolve_after_ms,
            prediction.expires_at_ms,
            prediction.status.value,
            prediction.resolved_at_ms,
            prediction.resolved_by_event_id,
            prediction.brier_contribution,
        )

    @staticmethod
    def _row_to_prediction(row: sqlite3.Row | dict[str, Any]) -> GroundedPrediction:
        data = dict(row)
        return GroundedPrediction(
            id=data["id"],
            conversation_id=data["conversation_id"],
            dedup_key=data["dedup_key"],
            claim_text=data["claim_text"],
            claim_kind=data["claim_kind"],
            verifier_kind=data["verifier_kind"],
            verifier_spec=json.loads(data["verifier_spec_json"]),
            stated_confidence=data["stated_confidence"],
            base_rate_prior=data["base_rate_prior"],
            source_event_ids=json.loads(data["source_event_ids_json"]),
            source_trace_ids=json.loads(data["source_trace_ids_json"]),
            created_at_ms=data["created_at_ms"],
            resolve_after_ms=data["resolve_after_ms"],
            expires_at_ms=data["expires_at_ms"],
            status=data["status"],
            resolved_at_ms=data["resolved_at_ms"],
            resolved_by_event_id=data["resolved_by_event_id"],
            brier_contribution=data["brier_contribution"],
        )


__all__ = ["PredictionRepository"]
