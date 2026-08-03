"""SQLite persistence for the dedicated emotional-memory library."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ssa.domain.emotional_memory import EmotionalMemory, EmotionalMemoryStatus
from ssa.domain.lifecycle import EmotionType


class EmotionalMemoryRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert(self, memory: EmotionalMemory) -> EmotionalMemory:
        existing = self.find_by_origin_trace(memory.origin_trace_id)
        if existing is not None:
            return existing
        self._conn.execute(
            """
            INSERT INTO emotional_memories (
                id, conversation_id, correlation_id, origin_trace_id,
                emotion_type, target, trigger_summary, felt_summary,
                valence, arousal, intensity, action_tendency,
                source_event_ids_json, source_trace_ids_json, status,
                recall_count, last_recalled_at_ms, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.id,
                memory.conversation_id,
                memory.correlation_id,
                memory.origin_trace_id,
                memory.emotion_type.value,
                memory.target,
                memory.trigger_summary,
                memory.felt_summary,
                memory.valence,
                memory.arousal,
                memory.intensity,
                memory.action_tendency,
                json.dumps(memory.source_event_ids, ensure_ascii=False),
                json.dumps(memory.source_trace_ids, ensure_ascii=False),
                memory.status.value,
                memory.recall_count,
                memory.last_recalled_at_ms,
                memory.created_at_ms,
                memory.updated_at_ms,
            ),
        )
        return memory

    def find_by_origin_trace(self, trace_id: str) -> EmotionalMemory | None:
        row = self._conn.execute(
            "SELECT * FROM emotional_memories WHERE origin_trace_id = ?",
            (trace_id,),
        ).fetchone()
        return self._row_to_memory(row) if row is not None else None

    def candidates(self, conversation_id: str, *, limit: int = 240) -> list[EmotionalMemory]:
        rows = self._conn.execute(
            """
            SELECT * FROM emotional_memories
            WHERE conversation_id = ? AND status IN ('active', 'settled')
            ORDER BY intensity DESC, updated_at_ms DESC, rowid DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def recent(self, conversation_id: str, *, limit: int = 20) -> list[EmotionalMemory]:
        rows = self._conn.execute(
            """
            SELECT * FROM emotional_memories
            WHERE conversation_id = ? AND status != 'archived'
            ORDER BY updated_at_ms DESC, rowid DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def mark_recalled(self, memory_ids: list[str], now_ms: int) -> None:
        if not memory_ids:
            return
        placeholders = ", ".join("?" for _ in memory_ids)
        self._conn.execute(
            f"""
            UPDATE emotional_memories
            SET recall_count = recall_count + 1,
                last_recalled_at_ms = ?
            WHERE id IN ({placeholders})
            """,
            (now_ms, *memory_ids),
        )

    def count(self, conversation_id: str | None = None) -> int:
        if conversation_id is None:
            row = self._conn.execute("SELECT COUNT(*) AS c FROM emotional_memories").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM emotional_memories WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return int(row["c"]) if row is not None else 0

    @staticmethod
    def _row_to_memory(row: sqlite3.Row | dict[str, Any]) -> EmotionalMemory:
        data = dict(row)
        return EmotionalMemory(
            id=data["id"],
            conversation_id=data["conversation_id"],
            correlation_id=data["correlation_id"],
            origin_trace_id=data["origin_trace_id"],
            emotion_type=EmotionType(data["emotion_type"]),
            target=data["target"],
            trigger_summary=data["trigger_summary"],
            felt_summary=data["felt_summary"],
            valence=data["valence"],
            arousal=data["arousal"],
            intensity=data["intensity"],
            action_tendency=data["action_tendency"],
            source_event_ids=json.loads(data["source_event_ids_json"] or "[]"),
            source_trace_ids=json.loads(data["source_trace_ids_json"] or "[]"),
            status=EmotionalMemoryStatus(data["status"]),
            recall_count=data["recall_count"],
            last_recalled_at_ms=data["last_recalled_at_ms"],
            created_at_ms=data["created_at_ms"],
            updated_at_ms=data["updated_at_ms"],
        )


__all__ = ["EmotionalMemoryRepository"]
