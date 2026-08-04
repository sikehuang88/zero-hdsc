"""SQLite persistence for user-controlled relationship preferences."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from pathlib import Path

from ssa.domain.relationship_preferences import RelationshipPreferences


class RelationshipPreferencesRepository:
    def __init__(self, database_path: str, *, busy_timeout_ms: int = 5_000) -> None:
        self._database_path = Path(database_path)
        self._busy_timeout_ms = busy_timeout_ms

    def get(self) -> RelationshipPreferences:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT venom_intensity, support_mode, proactive_frequency,
                       quiet_hours_enabled, quiet_hours_start, quiet_hours_end, updated_at_ms
                FROM relationship_preferences
                WHERE profile_id = 'primary'
                """
            ).fetchone()
        if row is None:
            return RelationshipPreferences()
        return RelationshipPreferences.model_validate(dict(row))

    def save(self, preferences: RelationshipPreferences) -> RelationshipPreferences:
        saved = preferences.model_copy(update={"updated_at_ms": int(time.time() * 1_000)})
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO relationship_preferences (
                    profile_id, venom_intensity, support_mode, proactive_frequency,
                    quiet_hours_enabled, quiet_hours_start, quiet_hours_end, updated_at_ms
                ) VALUES ('primary', ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id) DO UPDATE SET
                    venom_intensity = excluded.venom_intensity,
                    support_mode = excluded.support_mode,
                    proactive_frequency = excluded.proactive_frequency,
                    quiet_hours_enabled = excluded.quiet_hours_enabled,
                    quiet_hours_start = excluded.quiet_hours_start,
                    quiet_hours_end = excluded.quiet_hours_end,
                    updated_at_ms = excluded.updated_at_ms
                """,
                (
                    saved.venom_intensity,
                    saved.support_mode.value,
                    saved.proactive_frequency.value,
                    int(saved.quiet_hours_enabled),
                    saved.quiet_hours_start,
                    saved.quiet_hours_end,
                    saved.updated_at_ms,
                ),
            )
        return saved

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self._database_path),
            timeout=self._busy_timeout_ms / 1_000,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        return connection


__all__ = ["RelationshipPreferencesRepository"]
