"""Event repository — the only code that reads/writes the `events` table.

Pipeline §11.2: Services don't execute SQL directly; they go through
repositories. This repository returns domain `Event` objects, not raw
`sqlite3.Row`.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event


class SqliteEventRepository:
    """Concrete EventRepository backed by SQLite.

    Implements pipeline §11.2 contract:
    - `append`: idempotent on (channel, channel_message_id).
    - `get`: by event ID.
    - `find_by_channel_message`: dedup lookup.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def append(self, event: Event) -> Event:
        """Insert an event. Raises if the (channel, message_id) pair already exists.

        Pipeline §11.5: duplicate channel messages must produce only one Event.
        We rely on the UNIQUE(channel, channel_message_id) constraint.
        """
        metadata_json = json.dumps(event.metadata, ensure_ascii=False, sort_keys=True)
        try:
            self._conn.execute(
                """
                INSERT INTO events (
                    id, correlation_id, conversation_id, actor, event_type,
                    source_kind, content, parent_event_id, channel,
                    channel_message_id, content_hash, metadata_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.correlation_id,
                    event.conversation_id,
                    event.actor.value,
                    event.event_type,
                    event.source_kind.value,
                    event.content,
                    event.parent_event_id,
                    event.channel,
                    event.channel_message_id,
                    event.content_hash,
                    metadata_json,
                    event.created_at_ms,
                ),
            )
        except sqlite3.IntegrityError as exc:
            if "UNIQUE constraint failed: events.channel" in str(exc) or (
                "channel_message_id" in str(exc)
            ):
                raise DuplicateEventError(
                    f"Event with channel={event.channel!r} "
                    f"message_id={event.channel_message_id!r} already exists"
                ) from exc
            raise
        return event

    def get(self, event_id: str) -> Event | None:
        row = self._conn.execute(
            "SELECT * FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_event(row)

    def find_by_channel_message(self, channel: str, message_id: str) -> Event | None:
        row = self._conn.execute(
            "SELECT * FROM events WHERE channel = ? AND channel_message_id = ?",
            (channel, message_id),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_event(row)

    def find_by_correlation(self, correlation_id: str) -> list[Event]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE correlation_id = ? ORDER BY created_at_ms",
            (correlation_id,),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as c FROM events").fetchone()
        return int(row["c"]) if row else 0

    def recent(self, limit: int = 10) -> list[Event]:
        rows = self._conn.execute(
            "SELECT * FROM events ORDER BY created_at_ms DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def recent_by_conversation(
        self,
        conversation_id: str,
        *,
        limit: int = 30,
    ) -> list[Event]:
        """Return the latest conversation events in chronological order."""
        if not conversation_id.strip():
            raise ValueError("conversation_id must not be empty")
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = self._conn.execute(
            """
            SELECT * FROM events
            WHERE conversation_id = ?
            ORDER BY created_at_ms DESC, rowid DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        return [self._row_to_event(row) for row in reversed(rows)]

    def latest_by_actor(self, conversation_id: str, actor: Actor) -> Event | None:
        row = self._conn.execute(
            """
            SELECT * FROM events
            WHERE conversation_id = ? AND actor = ?
            ORDER BY created_at_ms DESC, rowid DESC LIMIT 1
            """,
            (conversation_id, actor.value),
        ).fetchone()
        return self._row_to_event(row) if row is not None else None

    def between_by_conversation(
        self,
        conversation_id: str,
        *,
        start_ms: int,
        end_ms: int,
    ) -> list[Event]:
        """Return immutable events inside an inclusive verification window."""
        if not conversation_id.strip():
            raise ValueError("conversation_id must not be empty")
        if start_ms < 0 or end_ms < start_ms:
            raise ValueError("event window must satisfy 0 <= start_ms <= end_ms")
        rows = self._conn.execute(
            """
            SELECT * FROM events
            WHERE conversation_id = ? AND created_at_ms BETWEEN ? AND ?
            ORDER BY created_at_ms, rowid
            """,
            (conversation_id, start_ms, end_ms),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_event(row: sqlite3.Row | dict[str, Any]) -> Event:
        """Map a DB row to a domain Event."""
        data = dict(row)

        metadata_raw = data.get("metadata_json") or "{}"
        metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else dict(metadata_raw)

        return Event(
            id=data["id"],
            correlation_id=data["correlation_id"],
            conversation_id=data["conversation_id"],
            actor=Actor(data["actor"]),
            event_type=data["event_type"],
            source_kind=SourceKind(data["source_kind"]),
            content=data["content"],
            parent_event_id=data.get("parent_event_id"),
            channel=data.get("channel"),
            channel_message_id=data.get("channel_message_id"),
            content_hash=data["content_hash"],
            metadata=metadata,
            created_at_ms=data["created_at_ms"],
        )


class DuplicateEventError(RuntimeError):
    """Raised when a channel+message_id collision is detected."""


__all__ = ["DuplicateEventError", "SqliteEventRepository"]
