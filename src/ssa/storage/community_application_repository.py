"""Persistence for public developer recruitment and private-beta reservations."""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

CommunityTrack = Literal["developer", "tester"]


@dataclass(frozen=True)
class CommunityApplicationInput:
    track: CommunityTrack
    email: str
    display_name: str
    focus: str
    profile_url: str | None = None
    platform: str | None = None
    notes: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class CommunityApplicationReceipt:
    application_code: str
    track: CommunityTrack
    queue_position: int
    created: bool


class CommunityApplicationRepository:
    """Small connection-per-call repository for the public reservation API."""

    def __init__(self, database_path: str, *, busy_timeout_ms: int = 5_000) -> None:
        self._database_path = Path(database_path)
        self._busy_timeout_ms = busy_timeout_ms

    def submit(self, application: CommunityApplicationInput) -> CommunityApplicationReceipt:
        now_ms = int(time.time() * 1_000)
        email = application.email.strip().lower()
        with closing(self._connect()) as connection, connection:
            existing = connection.execute(
                """
                SELECT id, application_code, track, created_at_ms
                FROM community_applications
                WHERE track = ? AND email = ?
                """,
                (application.track, email),
            ).fetchone()
            if existing is not None:
                return self._receipt(connection, existing, created=False)

            for _ in range(5):
                application_id = str(uuid.uuid4())
                application_code = self._new_code(application.track)
                try:
                    connection.execute(
                        """
                        INSERT INTO community_applications (
                            id, application_code, track, email, display_name, focus,
                            profile_url, platform, notes, metadata_json, status,
                            created_at_ms, updated_at_ms
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'reserved', ?, ?)
                        """,
                        (
                            application_id,
                            application_code,
                            application.track,
                            email,
                            application.display_name.strip(),
                            application.focus.strip(),
                            application.profile_url,
                            application.platform,
                            application.notes.strip(),
                            json.dumps(application.metadata or {}, ensure_ascii=False),
                            now_ms,
                            now_ms,
                        ),
                    )
                    row = connection.execute(
                        """
                        SELECT id, application_code, track, created_at_ms
                        FROM community_applications
                        WHERE id = ?
                        """,
                        (application_id,),
                    ).fetchone()
                    if row is None:
                        raise RuntimeError("community application insert was not visible")
                    return self._receipt(connection, row, created=True)
                except sqlite3.IntegrityError:
                    duplicate = connection.execute(
                        """
                        SELECT id, application_code, track, created_at_ms
                        FROM community_applications
                        WHERE track = ? AND email = ?
                        """,
                        (application.track, email),
                    ).fetchone()
                    if duplicate is not None:
                        return self._receipt(connection, duplicate, created=False)
            raise RuntimeError("community application code allocation exhausted")

    def counts(self) -> dict[CommunityTrack, int]:
        counts: dict[CommunityTrack, int] = {"developer": 0, "tester": 0}
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """
                SELECT track, COUNT(*) AS total
                FROM community_applications
                GROUP BY track
                """
            ).fetchall()
        for row in rows:
            track = str(row["track"])
            if track in counts:
                counts[track] = int(row["total"])
        return counts

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self._database_path),
            timeout=self._busy_timeout_ms / 1_000,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        return connection

    @staticmethod
    def _new_code(track: CommunityTrack) -> str:
        prefix = "DEV" if track == "developer" else "BETA"
        return f"ZERO-{prefix}-{secrets.token_hex(3).upper()}"

    @staticmethod
    def _receipt(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        created: bool,
    ) -> CommunityApplicationReceipt:
        queue_row = connection.execute(
            """
            SELECT COUNT(*) AS position
            FROM community_applications
            WHERE track = ?
              AND (created_at_ms < ? OR (created_at_ms = ? AND id <= ?))
            """,
            (row["track"], row["created_at_ms"], row["created_at_ms"], row["id"]),
        ).fetchone()
        return CommunityApplicationReceipt(
            application_code=str(row["application_code"]),
            track=cast(CommunityTrack, str(row["track"])),
            queue_position=int(queue_row["position"]) if queue_row is not None else 1,
            created=created,
        )


__all__ = [
    "CommunityApplicationInput",
    "CommunityApplicationReceipt",
    "CommunityApplicationRepository",
    "CommunityTrack",
]
