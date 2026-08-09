"""SQLite graph store for ENGRAM's append-only typed directed topology."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from typing import Any

from ssa.domain.engram import EngramEdge, EngramNode, EngramRelation
from ssa.domain.traces import Trace, TraceLink, TraceLinkType


class EngramRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def insert_node(self, node: EngramNode) -> EngramNode:
        self._connection.execute(
            """
            INSERT OR IGNORE INTO engram_nodes (node_id, node_type, conversation_id, created_at_ms)
            VALUES (?, ?, ?, ?)
            """,
            (node.node_id, node.node_type, node.conversation_id, node.created_at_ms),
        )
        return node

    def insert_edge(self, edge: EngramEdge, *, conversation_id: str) -> EngramEdge:
        self._connection.execute(
            """
            INSERT OR IGNORE INTO engram_edges (
                edge_id, src, dst, rel_type, support, action, valid_from_ms, valid_to_ms,
                source_event_id, trust_weight, likelihood_ratio
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge.edge_id,
                edge.src,
                edge.dst,
                edge.rel_type.value,
                edge.support,
                edge.action,
                edge.valid_from_ms,
                edge.valid_to_ms,
                edge.source_event_id,
                edge.trust_weight,
                edge.likelihood_ratio,
            ),
        )
        del conversation_id
        return edge

    def sync_trace_subgraph(
        self,
        traces: Sequence[Trace],
        links: Sequence[TraceLink],
    ) -> tuple[int, int]:
        """Idempotently project existing trace/link records into ENGRAM."""
        trace_ids = {trace.id for trace in traces}
        nodes = {
            trace.id: EngramNode(
                node_id=trace.id,
                node_type="trace",
                conversation_id=trace.conversation_id,
                created_at_ms=trace.created_at_ms,
            )
            for trace in traces
        }
        for node in nodes.values():
            self.insert_node(node)
        edge_count = 0
        for link in links:
            if link.source_trace_id not in trace_ids or link.target_trace_id not in trace_ids:
                continue
            relation = _relation_for_trace_link(link.link_type)
            edge = EngramEdge(
                edge_id=_edge_id(link, relation),
                src=link.source_trace_id,
                dst=link.target_trace_id,
                rel_type=relation,
                support=max(1e-9, float(link.weight)),
                valid_from_ms=link.created_at_ms,
                source_event_id=None,
            )
            self.insert_edge(edge, conversation_id=nodes[link.source_trace_id].conversation_id)
            edge_count += 1
        return len(nodes), edge_count

    def active_edges(
        self,
        conversation_id: str,
        node_ids: Sequence[str],
        *,
        now_ms: int,
    ) -> list[EngramEdge]:
        if not node_ids:
            return []
        placeholders = ", ".join("?" for _ in node_ids)
        rows = self._connection.execute(
            f"""
            SELECT edge.* FROM engram_edges AS edge
            JOIN engram_nodes AS source ON source.node_id = edge.src
            JOIN engram_nodes AS target ON target.node_id = edge.dst
            WHERE source.conversation_id = ?
              AND target.conversation_id = ?
              AND edge.src IN ({placeholders})
              AND edge.valid_from_ms <= ?
              AND (edge.valid_to_ms IS NULL OR edge.valid_to_ms > ?)
            ORDER BY edge.src, edge.dst, edge.rel_type, edge.edge_id
            """,
            (conversation_id, conversation_id, *node_ids, now_ms, now_ms),
        ).fetchall()
        return [self._row_to_edge(row) for row in rows]

    def count(self, conversation_id: str | None = None) -> int:
        if conversation_id is None:
            row = self._connection.execute("SELECT COUNT(*) AS count FROM engram_edges").fetchone()
        else:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM engram_edges AS edge
                JOIN engram_nodes AS source ON source.node_id = edge.src
                WHERE source.conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    @staticmethod
    def _row_to_edge(row: sqlite3.Row | dict[str, Any]) -> EngramEdge:
        data = dict(row)
        return EngramEdge(
            edge_id=str(data["edge_id"]),
            src=str(data["src"]),
            dst=str(data["dst"]),
            rel_type=EngramRelation(str(data["rel_type"])),
            support=float(data["support"]),
            action=str(data["action"]),
            valid_from_ms=int(data["valid_from_ms"]),
            valid_to_ms=(int(data["valid_to_ms"]) if data["valid_to_ms"] is not None else None),
            source_event_id=(
                str(data["source_event_id"]) if data["source_event_id"] is not None else None
            ),
            trust_weight=float(data["trust_weight"]),
            likelihood_ratio=float(data["likelihood_ratio"]),
        )


def _relation_for_trace_link(link_type: TraceLinkType) -> EngramRelation:
    return {
        TraceLinkType.SEMANTIC: EngramRelation.SEMANTIC,
        TraceLinkType.TEMPORAL: EngramRelation.TEMPORAL_FORWARD,
        TraceLinkType.AFFECTIVE: EngramRelation.EVIDENCE,
        TraceLinkType.ENTITY: EngramRelation.ENTITY,
        TraceLinkType.CAUSAL: EngramRelation.EVIDENCE,
    }[link_type]


def _edge_id(link: TraceLink, relation: EngramRelation) -> str:
    raw = "|".join(
        (
            link.source_trace_id,
            link.target_trace_id,
            relation.value,
            f"{link.created_at_ms}",
        )
    )
    return f"engram-edge:{hashlib.sha256(raw.encode()).hexdigest()[:32]}"


__all__ = ["EngramRepository"]
