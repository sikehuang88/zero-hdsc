"""Memory repository — the only code that reads/writes the `memories` table.

Pipeline §11.2: Services don't execute SQL directly.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

from ssa.domain.enums import MemoryType, SourceKind
from ssa.domain.memories import Memory


class SqliteMemoryRepository:
    """Concrete memory repository backed by SQLite."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert(self, memory: Memory, embedding_bytes: bytes) -> Memory:
        """Insert a memory and its embedding vector."""
        # vec0 rowid must be INTEGER — use a stable hash of the memory ID.
        vec_rowid = int(hashlib.sha256(memory.id.encode("utf-8")).hexdigest()[:15], 16)
        self._conn.execute(
            """
            INSERT INTO memories (
                id, memory_type, source_kind, content, summary,
                confidence, importance, valence, arousal,
                embedding_model, embedding_dim, content_hash,
                derived_by_model, prompt_version, status,
                access_count, last_accessed_at_ms, vec_rowid,
                created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.id,
                memory.memory_type.value,
                memory.source_kind.value,
                memory.content,
                memory.summary,
                memory.confidence,
                memory.importance,
                memory.valence,
                memory.arousal,
                memory.embedding_model,
                memory.embedding_dim,
                memory.content_hash,
                memory.derived_by_model,
                memory.prompt_version,
                memory.status,
                memory.access_count,
                memory.last_accessed_at_ms,
                vec_rowid,
                memory.created_at_ms,
                memory.updated_at_ms,
            ),
        )
        # Insert embedding into vec table if it exists.
        with suppress(sqlite3.OperationalError):
            self._conn.execute(
                "INSERT INTO memory_vec (rowid, embedding) VALUES (?, ?)",
                (vec_rowid, embedding_bytes),
            )
        return memory

    def insert_bundle(
        self,
        memory: Memory,
        embedding_bytes: bytes,
        evidence_event_ids: list[str],
        links: list[tuple[str, str, float, int]],
    ) -> Memory:
        """Atomically persist one memory with its evidence and semantic links."""
        if not evidence_event_ids:
            raise ValueError("memory persistence requires evidence")
        missing = [
            event_id
            for event_id in evidence_event_ids
            if self._conn.execute("SELECT 1 FROM events WHERE id = ?", (event_id,)).fetchone()
            is None
        ]
        if missing:
            raise ValueError(f"unknown memory evidence events: {missing}")
        with self._write_savepoint():
            self.insert(memory, embedding_bytes)
            for event_id in evidence_event_ids:
                self.add_evidence(memory.id, event_id, "supports")
            for target_id, link_type, weight, now_ms in links:
                self.add_link(memory.id, target_id, link_type, weight, now_ms)
        return memory

    def get(self, memory_id: str) -> Memory | None:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def find_by_content_hash(self, content_hash: str) -> Memory | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE content_hash = ? AND status = 'active'",
            (content_hash,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def find_similar(
        self,
        embedding_bytes: bytes,
        memory_type: MemoryType | None = None,
        limit: int = 10,
    ) -> list[tuple[Memory, float]]:
        """Find similar memories by vector similarity.

        Returns (memory, similarity) pairs. Similarity is in [0, 1].
        """
        try:
            rows = self._conn.execute(
                "SELECT rowid, distance FROM memory_vec WHERE embedding MATCH ? "
                "ORDER BY distance LIMIT ?",
                (embedding_bytes, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        results: list[tuple[Memory, float]] = []
        for row in rows:
            vec_rowid = row["rowid"]
            distance = float(row["distance"])
            mem_row = self._conn.execute(
                "SELECT * FROM memories WHERE vec_rowid = ?",
                (vec_rowid,),
            ).fetchone()
            if mem_row is None:
                continue
            mem = self._row_to_memory(mem_row)
            if memory_type is not None and mem.memory_type != memory_type:
                continue
            if mem.status != "active":
                continue
            similarity = max(0.0, 1.0 - distance)
            results.append((mem, similarity))
        return results

    def add_evidence(self, memory_id: str, event_id: str, relation: str = "supports") -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO memory_evidence (memory_id, event_id, relation) "
            "VALUES (?, ?, ?)",
            (memory_id, event_id, relation),
        )

    def get_evidence(self, memory_id: str) -> list[tuple[str, str]]:
        rows = self._conn.execute(
            "SELECT event_id, relation FROM memory_evidence WHERE memory_id = ?",
            (memory_id,),
        ).fetchall()
        return [(r["event_id"], r["relation"]) for r in rows]

    def add_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str,
        weight: float,
        now_ms: int,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO memory_links "
            "(source_memory_id, target_memory_id, link_type, weight, created_at_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (source_id, target_id, link_type, weight, now_ms),
        )

    def get_links(self, memory_id: str) -> list[tuple[str, str, float]]:
        rows = self._conn.execute(
            "SELECT target_memory_id, link_type, weight FROM memory_links "
            "WHERE source_memory_id = ?",
            (memory_id,),
        ).fetchall()
        return [(r["target_memory_id"], r["link_type"], r["weight"]) for r in rows]

    def update_status(self, memory_id: str, status: str, now_ms: int) -> None:
        self._conn.execute(
            "UPDATE memories SET status = ?, updated_at_ms = ? WHERE id = ?",
            (status, now_ms, memory_id),
        )

    def increment_access(self, memory_id: str, now_ms: int) -> None:
        self._conn.execute(
            "UPDATE memories SET access_count = access_count + 1, "
            "last_accessed_at_ms = ? WHERE id = ?",
            (now_ms, memory_id),
        )

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()
        return int(row["c"]) if row else 0

    def count_by_type(self, memory_type: MemoryType) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) as c FROM memories WHERE memory_type = ?",
            (memory_type.value,),
        ).fetchone()
        return int(row["c"]) if row else 0

    def all_active(self, limit: int = 100) -> list[Memory]:
        """Return active memories from most recently updated to oldest."""
        rows = self._conn.execute(
            """
            SELECT * FROM memories
            WHERE status = 'active'
            ORDER BY updated_at_ms DESC, rowid DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    @contextmanager
    def _write_savepoint(self) -> Iterator[None]:
        self._conn.execute("SAVEPOINT memory_bundle_write")
        try:
            yield
            self._conn.execute("RELEASE SAVEPOINT memory_bundle_write")
        except Exception:
            self._conn.execute("ROLLBACK TO SAVEPOINT memory_bundle_write")
            self._conn.execute("RELEASE SAVEPOINT memory_bundle_write")
            raise

    @staticmethod
    def _row_to_memory(row: sqlite3.Row | dict[str, Any]) -> Memory:
        data = dict(row)
        return Memory(
            id=data["id"],
            memory_type=MemoryType(data["memory_type"]),
            source_kind=SourceKind(data["source_kind"]),
            content=data["content"],
            summary=data.get("summary"),
            confidence=data["confidence"],
            importance=data["importance"],
            valence=data["valence"],
            arousal=data["arousal"],
            embedding_model=data["embedding_model"],
            embedding_dim=data["embedding_dim"],
            content_hash=data["content_hash"],
            derived_by_model=data.get("derived_by_model"),
            prompt_version=data.get("prompt_version"),
            status=data.get("status", "active"),
            access_count=data.get("access_count", 0),
            last_accessed_at_ms=data.get("last_accessed_at_ms"),
            created_at_ms=data["created_at_ms"],
            updated_at_ms=data["updated_at_ms"],
        )


__all__ = ["SqliteMemoryRepository"]
