"""SQLite persistence for the cement-seal state and its transition ledger."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

from ssa.domain.cement_seal import CementSealState, CementSealTransition


class CementSealRepository:
    def __init__(self, database_path: str, *, busy_timeout_ms: int = 5_000) -> None:
        self._database_path = Path(database_path)
        self._busy_timeout_ms = busy_timeout_ms

    def get(self) -> CementSealState:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT phase, integrity, seal_count, sealed_at_ms,
                       last_transition_ms, source_event_ids, version
                FROM cement_seal_state
                WHERE profile_id = 'primary'
                """
            ).fetchone()
        if row is None:
            return CementSealState()
        payload = dict(row)
        payload["source_event_ids"] = json.loads(payload["source_event_ids"] or "[]")
        return CementSealState.model_validate(payload)

    def save(self, state: CementSealState) -> CementSealState:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO cement_seal_state (
                    profile_id, phase, integrity, seal_count, sealed_at_ms,
                    last_transition_ms, source_event_ids, version
                ) VALUES ('primary', ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id) DO UPDATE SET
                    phase = excluded.phase,
                    integrity = excluded.integrity,
                    seal_count = excluded.seal_count,
                    sealed_at_ms = excluded.sealed_at_ms,
                    last_transition_ms = excluded.last_transition_ms,
                    source_event_ids = excluded.source_event_ids,
                    version = excluded.version
                """,
                (
                    state.phase.value,
                    state.integrity,
                    state.seal_count,
                    state.sealed_at_ms,
                    state.last_transition_ms,
                    json.dumps(state.source_event_ids, ensure_ascii=False),
                    state.version,
                ),
            )
        return state

    def append_transition(self, transition: CementSealTransition) -> str:
        transition_id = f"cement:{uuid.uuid4().hex}"
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO cement_seal_transitions (
                    transition_id, previous_phase, phase, trigger,
                    integrity_before, integrity_after, seal_count, toughness,
                    reason, source_event_ids, occurred_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transition_id,
                    transition.previous_phase.value,
                    transition.phase.value,
                    transition.trigger.value,
                    transition.integrity_before,
                    transition.integrity_after,
                    transition.seal_count,
                    transition.toughness,
                    transition.reason,
                    json.dumps(transition.source_event_ids, ensure_ascii=False),
                    transition.occurred_at_ms,
                ),
            )
        return transition_id

    def recent_transitions(self, limit: int = 10) -> list[CementSealTransition]:
        if limit < 1:
            raise ValueError("cement seal transition limit must be positive")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT previous_phase, phase, trigger, integrity_before, integrity_after,
                       seal_count, toughness, reason, source_event_ids, occurred_at_ms
                FROM cement_seal_transitions
                ORDER BY occurred_at_ms DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        transitions: list[CementSealTransition] = []
        for row in rows:
            payload = dict(row)
            payload["source_event_ids"] = json.loads(payload["source_event_ids"] or "[]")
            transitions.append(CementSealTransition.model_validate(payload))
        return transitions

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        return connection


__all__ = ["CementSealRepository"]
