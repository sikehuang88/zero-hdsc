"""SQLite persistence for append-only self-belief version chains."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ssa.domain.self_belief import (
    SelfBelief,
    SelfBeliefEvidence,
    SelfBeliefEvidenceRelation,
    SelfBeliefStatus,
    validate_self_belief_transition,
)


class SqliteSelfBeliefRepository:
    """Stores immutable belief versions with atomic optimistic locking."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert_initial(self, belief: SelfBelief) -> SelfBelief:
        """Create a candidate lineage at version one."""
        self._validate_initial(belief)
        with self._write_savepoint():
            cursor = self._conn.execute(
                f"""
                INSERT INTO self_beliefs ({self._column_names()})
                SELECT {self._value_placeholders()}
                WHERE NOT EXISTS (
                    SELECT 1 FROM self_beliefs WHERE lineage_id = ?
                )
                """,
                (*self._record_values(belief), belief.lineage_id),
            )
            if cursor.rowcount != 1:
                raise SelfBeliefVersionConflict(
                    f"Self-belief lineage {belief.lineage_id!r} already exists"
                )
            self._insert_evidence(belief)
        return belief

    def append_if_version(
        self,
        expected_version: int,
        belief: SelfBelief,
    ) -> SelfBelief:
        """Append only when ``expected_version`` is still the lineage head."""
        required_version = expected_version + 1
        if belief.version != required_version:
            raise ValueError(
                f"Self-belief version must be {required_version}, got {belief.version}"
            )

        previous = self.latest(belief.lineage_id)
        if previous is None or previous.version != expected_version:
            current_version = previous.version if previous is not None else 0
            raise SelfBeliefVersionConflict(
                f"Expected version {expected_version}, but current is {current_version}"
            )
        self._validate_successor(previous, belief)

        with self._write_savepoint():
            cursor = self._conn.execute(
                f"""
                INSERT INTO self_beliefs ({self._column_names()})
                SELECT {self._value_placeholders()}
                FROM self_beliefs AS current
                WHERE current.id = ?
                  AND current.lineage_id = ?
                  AND current.version = ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM self_beliefs AS newer
                      WHERE newer.lineage_id = current.lineage_id
                        AND newer.version > current.version
                  )
                """,
                (
                    *self._record_values(belief),
                    previous.id,
                    belief.lineage_id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                current = self.latest(belief.lineage_id)
                current_version = current.version if current is not None else 0
                raise SelfBeliefVersionConflict(
                    f"Expected version {expected_version}, but current is {current_version}"
                )
            self._insert_evidence(belief)
        return belief

    def get(self, belief_version_id: str) -> SelfBelief | None:
        row = self._conn.execute(
            "SELECT * FROM self_beliefs WHERE id = ?",
            (belief_version_id,),
        ).fetchone()
        return self._row_to_belief(row) if row is not None else None

    def latest(self, lineage_id: str) -> SelfBelief | None:
        row = self._conn.execute(
            """
            SELECT *
            FROM self_beliefs
            WHERE lineage_id = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (lineage_id,),
        ).fetchone()
        return self._row_to_belief(row) if row is not None else None

    def get_by_version(self, lineage_id: str, version: int) -> SelfBelief | None:
        row = self._conn.execute(
            """
            SELECT * FROM self_beliefs
            WHERE lineage_id = ? AND version = ?
            """,
            (lineage_id, version),
        ).fetchone()
        return self._row_to_belief(row) if row is not None else None

    def history(self, lineage_id: str) -> list[SelfBelief]:
        rows = self._conn.execute(
            """
            SELECT * FROM self_beliefs
            WHERE lineage_id = ?
            ORDER BY version
            """,
            (lineage_id,),
        ).fetchall()
        return [self._row_to_belief(row) for row in rows]

    def list_current(
        self,
        statuses: set[SelfBeliefStatus] | None = None,
        *,
        limit: int | None = None,
    ) -> list[SelfBelief]:
        """List lineage heads, optionally filtered by lifecycle status."""
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")

        sql = """
            SELECT belief.*
            FROM self_beliefs AS belief
            JOIN (
                SELECT lineage_id, MAX(version) AS version
                FROM self_beliefs
                WHERE lineage_id IS NOT NULL
                GROUP BY lineage_id
            ) AS heads
              ON heads.lineage_id = belief.lineage_id
             AND heads.version = belief.version
        """
        params: list[object] = []
        if statuses:
            ordered_statuses = sorted(status.value for status in statuses)
            placeholders = ", ".join("?" for _ in ordered_statuses)
            sql += f" WHERE belief.status IN ({placeholders})"
            params.extend(ordered_statuses)
        sql += " ORDER BY belief.updated_at_ms DESC, belief.id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_belief(row) for row in rows]

    def find_current_by_claim(self, claim: str) -> SelfBelief | None:
        normalized = claim.strip().casefold()
        for belief in self.list_current():
            if belief.claim.casefold() == normalized:
                return belief
        return None

    def evidence_for_version(self, belief_version_id: str) -> list[SelfBeliefEvidence]:
        rows = self._conn.execute(
            """
            SELECT belief_version_id, event_id, relation, recorded_at_ms
            FROM self_belief_evidence
            WHERE belief_version_id = ?
            ORDER BY relation, event_id
            """,
            (belief_version_id,),
        ).fetchall()
        return [
            SelfBeliefEvidence(
                belief_version_id=str(row["belief_version_id"]),
                event_id=str(row["event_id"]),
                relation=SelfBeliefEvidenceRelation(str(row["relation"])),
                recorded_at_ms=int(row["recorded_at_ms"]),
            )
            for row in rows
        ]

    @staticmethod
    def _validate_initial(belief: SelfBelief) -> None:
        if belief.version != 1:
            raise ValueError("an initial self belief must have version 1")
        if belief.id != belief.lineage_id:
            raise ValueError("the initial version ID must equal its lineage ID")
        if belief.previous_id is not None:
            raise ValueError("an initial self belief must not have a predecessor")
        if belief.status != SelfBeliefStatus.CANDIDATE:
            raise ValueError("a self belief must start as candidate")
        if belief.confidence > 0.6:
            raise ValueError("initial self-belief confidence must not exceed 0.6")
        if len(belief.evidence_event_ids) < 2:
            raise ValueError("a self belief needs at least two independent events")
        SqliteSelfBeliefRepository._validate_audit_fields(belief)

    @staticmethod
    def _validate_successor(previous: SelfBelief, belief: SelfBelief) -> None:
        if belief.lineage_id != previous.lineage_id:
            raise ValueError("a successor must remain in the same lineage")
        if belief.previous_id != previous.id:
            raise ValueError("a successor must reference the current version ID")
        if belief.updated_at_ms < previous.updated_at_ms:
            raise ValueError("a successor timestamp must not move backwards")
        validate_self_belief_transition(previous.status, belief.status)

        old_support = set(previous.evidence_event_ids)
        old_counter = set(previous.counterevidence_event_ids)
        if not old_support.issubset(belief.evidence_event_ids):
            raise ValueError("supporting evidence is append-only")
        if not old_counter.issubset(belief.counterevidence_event_ids):
            raise ValueError("counterevidence is append-only")
        if belief.claim != previous.claim:
            if not (
                previous.status == SelfBeliefStatus.CHALLENGED
                and belief.status == SelfBeliefStatus.REVISED
            ):
                raise ValueError("a claim can change only when challenged belief is revised")
        elif (
            previous.status == SelfBeliefStatus.CHALLENGED
            and belief.status == SelfBeliefStatus.REVISED
        ):
            raise ValueError("a revised belief must change its claim")
        SqliteSelfBeliefRepository._validate_audit_fields(belief)

    @staticmethod
    def _validate_audit_fields(belief: SelfBelief) -> None:
        if belief.cause_event_id is None:
            raise ValueError("new self-belief versions require a cause event")
        if belief.model is None or belief.prompt_version is None:
            raise ValueError("new self-belief versions require model and prompt metadata")

    def _insert_evidence(self, belief: SelfBelief) -> None:
        rows = [
            (
                belief.id,
                event_id,
                SelfBeliefEvidenceRelation.SUPPORTS.value,
                belief.updated_at_ms,
            )
            for event_id in belief.evidence_event_ids
        ]
        rows.extend(
            (
                belief.id,
                event_id,
                SelfBeliefEvidenceRelation.CONTRADICTS.value,
                belief.updated_at_ms,
            )
            for event_id in belief.counterevidence_event_ids
        )
        self._conn.executemany(
            """
            INSERT INTO self_belief_evidence (
                belief_version_id, event_id, relation, recorded_at_ms
            ) VALUES (?, ?, ?, ?)
            """,
            rows,
        )

    @contextmanager
    def _write_savepoint(self) -> Iterator[None]:
        self._conn.execute("SAVEPOINT self_belief_write")
        try:
            yield
            self._conn.execute("RELEASE SAVEPOINT self_belief_write")
        except Exception:
            self._conn.execute("ROLLBACK TO SAVEPOINT self_belief_write")
            self._conn.execute("RELEASE SAVEPOINT self_belief_write")
            raise

    @staticmethod
    def _column_names() -> str:
        return """
            id, lineage_id, claim, confidence, status, version,
            evidence_json, counterevidence_json, previous_id, cause_event_id,
            change_reason, model, prompt_version, candidate_since_ms,
            activated_at_ms, created_at_ms, updated_at_ms
        """

    @staticmethod
    def _value_placeholders() -> str:
        return ", ".join("?" for _ in range(17))

    @staticmethod
    def _record_values(belief: SelfBelief) -> tuple[object, ...]:
        return (
            belief.id,
            belief.lineage_id,
            belief.claim,
            belief.confidence,
            belief.status.value,
            belief.version,
            json.dumps(belief.evidence_event_ids, ensure_ascii=False),
            json.dumps(belief.counterevidence_event_ids, ensure_ascii=False),
            belief.previous_id,
            belief.cause_event_id,
            belief.change_reason,
            belief.model,
            belief.prompt_version,
            belief.candidate_since_ms,
            belief.activated_at_ms,
            belief.created_at_ms,
            belief.updated_at_ms,
        )

    @staticmethod
    def _row_to_belief(row: sqlite3.Row | dict[str, Any]) -> SelfBelief:
        data = dict(row)
        belief_id = str(data["id"])
        return SelfBelief(
            id=belief_id,
            lineage_id=str(data.get("lineage_id") or belief_id),
            claim=str(data["claim"]),
            confidence=float(data["confidence"]),
            status=SelfBeliefStatus(str(data["status"])),
            version=int(data["version"]),
            evidence_event_ids=json.loads(str(data["evidence_json"])),
            counterevidence_event_ids=json.loads(str(data["counterevidence_json"])),
            previous_id=data.get("previous_id"),
            cause_event_id=data.get("cause_event_id"),
            change_reason=str(data.get("change_reason") or "legacy migration"),
            model=data.get("model"),
            prompt_version=data.get("prompt_version"),
            candidate_since_ms=int(data.get("candidate_since_ms") or data["created_at_ms"]),
            activated_at_ms=data.get("activated_at_ms"),
            created_at_ms=int(data["created_at_ms"]),
            updated_at_ms=int(data["updated_at_ms"]),
        )


class SelfBeliefVersionConflict(RuntimeError):
    """Raised when another writer advanced a belief lineage first."""


__all__ = ["SelfBeliefVersionConflict", "SqliteSelfBeliefRepository"]
