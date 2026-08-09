"""SQLite persistence for append-only traces, links, and activation audits."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import suppress
from typing import Any, Literal

import numpy as np

from ssa.domain.enums import SourceKind
from ssa.domain.traces import (
    ActivatedTrace,
    ResonanceRecallAudit,
    Trace,
    TraceLink,
    TraceLinkType,
)


def trace_vec_rowid(trace_id: str) -> int:
    """Return a stable positive rowid for a textual trace identifier."""
    return int(hashlib.sha256(trace_id.encode("utf-8")).hexdigest()[:15], 16)


class SqliteTraceRepository:
    """Repository boundary for the immutable trace-space substrate."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert(self, trace: Trace, embedding_bytes: bytes) -> Trace:
        self._conn.execute(
            """
            INSERT INTO traces (
                id, conversation_id, correlation_id, input_event_id,
                output_event_id, content, content_type, source_kind,
                importance, valence, arousal, is_internal, tension,
                embedding_model, embedding_dim, vec_rowid, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace.id,
                trace.conversation_id,
                trace.correlation_id,
                trace.input_event_id,
                trace.output_event_id,
                trace.content,
                trace.content_type,
                trace.source_kind.value,
                trace.importance,
                trace.valence,
                trace.arousal,
                int(trace.is_internal),
                trace.tension,
                trace.embedding_model,
                trace.embedding_dim,
                trace.vec_rowid,
                trace.created_at_ms,
            ),
        )
        with suppress(sqlite3.OperationalError):
            self._conn.execute(
                "INSERT INTO trace_vec (rowid, embedding) VALUES (?, ?)",
                (trace.vec_rowid, embedding_bytes),
            )
        return trace

    def get(self, trace_id: str) -> Trace | None:
        row = self._conn.execute(
            "SELECT * FROM traces WHERE id = ?",
            (trace_id,),
        ).fetchone()
        return self._row_to_trace(row) if row is not None else None

    def find_by_input_event(self, input_event_id: str) -> Trace | None:
        row = self._conn.execute(
            "SELECT * FROM traces WHERE input_event_id = ? LIMIT 1",
            (input_event_id,),
        ).fetchone()
        return self._row_to_trace(row) if row is not None else None

    def find_by_event(self, event_id: str) -> Trace | None:
        row = self._conn.execute(
            """
            SELECT * FROM traces
            WHERE input_event_id = ? OR output_event_id = ?
            ORDER BY created_at_ms DESC, rowid DESC
            LIMIT 1
            """,
            (event_id, event_id),
        ).fetchone()
        return self._row_to_trace(row) if row is not None else None

    def count(self, conversation_id: str | None = None) -> int:
        if conversation_id is None:
            row = self._conn.execute("SELECT COUNT(*) AS c FROM traces").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM traces WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return int(row["c"]) if row is not None else 0

    def recent(self, conversation_id: str, *, limit: int = 120) -> list[Trace]:
        rows = self._conn.execute(
            """
            SELECT * FROM traces
            WHERE conversation_id = ?
            ORDER BY created_at_ms DESC, rowid DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        return [self._row_to_trace(row) for row in reversed(rows)]

    def all_with_embeddings(self, conversation_id: str) -> list[tuple[Trace, list[float]]]:
        try:
            rows = self._conn.execute(
                """
                SELECT trace.*, vector.embedding AS vector_embedding
                FROM traces AS trace
                JOIN trace_vec AS vector ON vector.rowid = trace.vec_rowid
                WHERE trace.conversation_id = ?
                ORDER BY trace.created_at_ms, trace.rowid
                """,
                (conversation_id,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        results: list[tuple[Trace, list[float]]] = []
        for row in rows:
            vector = self._decode_embedding(row["vector_embedding"])
            if vector is not None:
                results.append((self._row_to_trace(row), vector))
        return results

    def find_similar(
        self,
        embedding_bytes: bytes,
        *,
        conversation_id: str,
        limit: int = 50,
    ) -> list[tuple[Trace, float]]:
        try:
            rows = self._conn.execute(
                """
                SELECT rowid, distance
                FROM trace_vec
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
                """,
                (embedding_bytes, limit * 3),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        results: list[tuple[Trace, float]] = []
        for row in rows:
            trace_row = self._conn.execute(
                "SELECT * FROM traces WHERE vec_rowid = ? AND conversation_id = ?",
                (int(row["rowid"]), conversation_id),
            ).fetchone()
            if trace_row is None:
                continue
            distance = max(0.0, float(row["distance"]))
            similarity = max(0.0, min(1.0, 1.0 - (distance * distance / 2.0)))
            results.append((self._row_to_trace(trace_row), similarity))
            if len(results) >= limit:
                break
        return results

    def add_link(self, link: TraceLink) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO trace_links (
                source_trace_id, target_trace_id, link_type, weight, created_at_ms
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                link.source_trace_id,
                link.target_trace_id,
                link.link_type.value,
                link.weight,
                link.created_at_ms,
            ),
        )

    def links_from(
        self,
        trace_id: str,
        *,
        link_types: tuple[str, ...] | None = None,
    ) -> list[TraceLink]:
        if link_types:
            placeholders = ", ".join("?" for _ in link_types)
            rows = self._conn.execute(
                f"""
                SELECT * FROM trace_links
                WHERE source_trace_id = ?
                  AND link_type IN ({placeholders})
                ORDER BY weight DESC, target_trace_id
                """,
                (trace_id, *link_types),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT * FROM trace_links
                WHERE source_trace_id = ?
                ORDER BY weight DESC, target_trace_id
                """,
                (trace_id,),
            ).fetchall()
        return [self._row_to_link(row) for row in rows]

    def links_among(self, trace_ids: list[str]) -> list[TraceLink]:
        if not trace_ids:
            return []
        placeholders = ", ".join("?" for _ in trace_ids)
        rows = self._conn.execute(
            f"""
            SELECT * FROM trace_links
            WHERE source_trace_id IN ({placeholders})
              AND target_trace_id IN ({placeholders})
            ORDER BY source_trace_id, weight DESC
            """,
            (*trace_ids, *trace_ids),
        ).fetchall()
        return [self._row_to_link(row) for row in rows]

    def record_activations(
        self,
        *,
        activation_ids: list[str],
        conversation_id: str,
        correlation_id: str,
        query_event_id: str,
        activations: list[ActivatedTrace],
        activated_at_ms: int,
    ) -> None:
        if len(activation_ids) != len(activations):
            raise ValueError("each trace activation requires one audit ID")
        self._conn.executemany(
            """
            INSERT INTO trace_activations (
                id, conversation_id, correlation_id, query_event_id,
                trace_id, rank, score, semantic_similarity, freshness,
                importance_factor, activation_kind, activated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    activation_id,
                    conversation_id,
                    correlation_id,
                    query_event_id,
                    activation.trace.id,
                    activation.rank,
                    activation.score,
                    activation.semantic_similarity,
                    activation.freshness,
                    activation.importance_factor,
                    activation.activation_kind,
                    activated_at_ms,
                )
                for activation_id, activation in zip(
                    activation_ids,
                    activations,
                    strict=True,
                )
            ],
        )

    def latest_activations(self, conversation_id: str) -> list[ActivatedTrace]:
        latest = self._conn.execute(
            """
            SELECT correlation_id FROM trace_activations
            WHERE conversation_id = ?
            ORDER BY activated_at_ms DESC, rowid DESC
            LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()
        if latest is None:
            return []
        rows = self._conn.execute(
            """
            SELECT * FROM trace_activations
            WHERE correlation_id = ?
            ORDER BY rank
            """,
            (str(latest["correlation_id"]),),
        ).fetchall()
        results: list[ActivatedTrace] = []
        for row in rows:
            trace = self.get(str(row["trace_id"]))
            if trace is None:
                continue
            raw_kind = str(row["activation_kind"])
            if raw_kind not in {"main", "radiation", "resonance"}:
                continue
            activation_kind: Literal["main", "radiation", "resonance"] = (
                "main"
                if raw_kind == "main"
                else "radiation"
                if raw_kind == "radiation"
                else "resonance"
            )
            results.append(
                ActivatedTrace(
                    trace=trace,
                    rank=int(row["rank"]),
                    score=float(row["score"]),
                    semantic_similarity=float(row["semantic_similarity"]),
                    freshness=float(row["freshness"]),
                    importance_factor=float(row["importance_factor"]),
                    activation_kind=activation_kind,
                )
            )
        return results

    def insert_resonance_audit(self, audit: ResonanceRecallAudit) -> ResonanceRecallAudit:
        self._conn.execute(
            """
            INSERT INTO resonance_recall_audits (
                id, conversation_id, query_digest, situation_mode, hop_budget,
                edge_gains_json, candidate_audits_json, selected_trace_ids_json,
                background_median, emergence_ratio, null_mass,
                conservation_residual, emerged, created_at_ms, warped_shadow_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit.id,
                audit.conversation_id,
                audit.query_digest,
                audit.situation_mode,
                audit.hop_budget,
                json.dumps(
                    {key.value: value for key, value in audit.edge_gains.items()},
                    sort_keys=True,
                ),
                json.dumps(
                    [item.model_dump(mode="json") for item in audit.candidates],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                json.dumps(audit.selected_trace_ids),
                audit.background_median,
                audit.emergence_ratio,
                audit.null_mass,
                audit.conservation_residual,
                int(audit.emerged),
                audit.created_at_ms,
                (
                    json.dumps(
                        audit.warped_shadow.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    if audit.warped_shadow is not None
                    else None
                ),
            ),
        )
        return audit

    def latest_resonance_audit(self, conversation_id: str) -> ResonanceRecallAudit | None:
        row = self._conn.execute(
            """
            SELECT * FROM resonance_recall_audits
            WHERE conversation_id = ?
            ORDER BY created_at_ms DESC, rowid DESC
            LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()
        if row is None:
            return None
        return ResonanceRecallAudit(
            id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            query_digest=str(row["query_digest"]),
            situation_mode=str(row["situation_mode"]),
            hop_budget=int(row["hop_budget"]),
            edge_gains=json.loads(str(row["edge_gains_json"])),
            candidates=tuple(json.loads(str(row["candidate_audits_json"]))),
            selected_trace_ids=tuple(json.loads(str(row["selected_trace_ids_json"]))),
            background_median=float(row["background_median"]),
            emergence_ratio=float(row["emergence_ratio"]),
            null_mass=float(row["null_mass"]),
            conservation_residual=float(row["conservation_residual"]),
            emerged=bool(row["emerged"]),
            created_at_ms=int(row["created_at_ms"]),
            warped_shadow=(
                json.loads(str(row["warped_shadow_json"]))
                if row["warped_shadow_json"] is not None
                else None
            ),
        )

    def embedding(self, trace: Trace) -> list[float] | None:
        try:
            row = self._conn.execute(
                "SELECT embedding FROM trace_vec WHERE rowid = ?",
                (trace.vec_rowid,),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        if row is None:
            return None
        return self._decode_embedding(row["embedding"])

    @staticmethod
    def _decode_embedding(raw: object) -> list[float] | None:
        if isinstance(raw, str):
            decoded = json.loads(raw)
            return [float(value) for value in decoded]
        if isinstance(raw, memoryview):
            raw = raw.tobytes()
        if isinstance(raw, bytes):
            return [float(value) for value in np.frombuffer(raw, dtype=np.float32)]
        return None

    @staticmethod
    def _row_to_trace(row: sqlite3.Row | dict[str, Any]) -> Trace:
        data = dict(row)
        return Trace(
            id=str(data["id"]),
            conversation_id=str(data["conversation_id"]),
            correlation_id=str(data["correlation_id"]),
            input_event_id=str(data["input_event_id"]),
            output_event_id=(
                str(data["output_event_id"]) if data.get("output_event_id") is not None else None
            ),
            content=str(data["content"]),
            content_type=str(data["content_type"]),
            source_kind=SourceKind(str(data["source_kind"])),
            importance=float(data["importance"]),
            valence=float(data["valence"]),
            arousal=float(data["arousal"]),
            tension=float(data["tension"]),
            is_internal=bool(data["is_internal"]),
            embedding_model=str(data["embedding_model"]),
            embedding_dim=int(data["embedding_dim"]),
            vec_rowid=int(data["vec_rowid"]),
            created_at_ms=int(data["created_at_ms"]),
        )

    @staticmethod
    def _row_to_link(row: sqlite3.Row | dict[str, Any]) -> TraceLink:
        data = dict(row)
        return TraceLink(
            source_trace_id=str(data["source_trace_id"]),
            target_trace_id=str(data["target_trace_id"]),
            link_type=TraceLinkType(str(data["link_type"])),
            weight=float(data["weight"]),
            created_at_ms=int(data["created_at_ms"]),
        )


__all__ = ["SqliteTraceRepository", "trace_vec_rowid"]
