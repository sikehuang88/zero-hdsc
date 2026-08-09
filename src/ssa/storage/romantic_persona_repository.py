"""SQLite persistence for the primary romantic persona card."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from pathlib import Path

from ssa.domain.romantic_persona import RomanticPersonaProfile


class RomanticPersonaRepository:
    def __init__(self, database_path: str, *, busy_timeout_ms: int = 5_000) -> None:
        self._database_path = Path(database_path)
        self._busy_timeout_ms = busy_timeout_ms

    def get(self) -> RomanticPersonaProfile:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT character_name, owner_address, relationship_stage,
                       affection_style, care_style, conflict_style,
                       teasing_intensity, vulnerability_openness, custom_notes,
                       version, updated_at_ms
                FROM romantic_persona_profiles
                WHERE profile_id = 'primary'
                """
            ).fetchone()
        if row is None:
            return RomanticPersonaProfile()
        return RomanticPersonaProfile.model_validate(dict(row))

    def save(self, profile: RomanticPersonaProfile) -> RomanticPersonaProfile:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT version FROM romantic_persona_profiles WHERE profile_id = 'primary'"
            ).fetchone()
            persisted_version = int(row["version"]) if row is not None else 0
            saved = profile.model_copy(
                update={
                    "version": max(persisted_version, profile.version) + 1,
                    "updated_at_ms": int(time.time() * 1_000),
                }
            )
            connection.execute(
                """
                INSERT INTO romantic_persona_profiles (
                    profile_id, character_name, owner_address, relationship_stage,
                    affection_style, care_style, conflict_style, teasing_intensity,
                    vulnerability_openness, custom_notes, version, updated_at_ms
                ) VALUES ('primary', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id) DO UPDATE SET
                    character_name = excluded.character_name,
                    owner_address = excluded.owner_address,
                    relationship_stage = excluded.relationship_stage,
                    affection_style = excluded.affection_style,
                    care_style = excluded.care_style,
                    conflict_style = excluded.conflict_style,
                    teasing_intensity = excluded.teasing_intensity,
                    vulnerability_openness = excluded.vulnerability_openness,
                    custom_notes = excluded.custom_notes,
                    version = excluded.version,
                    updated_at_ms = excluded.updated_at_ms
                """,
                (
                    saved.character_name,
                    saved.owner_address,
                    saved.relationship_stage.value,
                    saved.affection_style.value,
                    saved.care_style.value,
                    saved.conflict_style.value,
                    saved.teasing_intensity,
                    saved.vulnerability_openness,
                    saved.custom_notes,
                    saved.version,
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


__all__ = ["RomanticPersonaRepository"]
