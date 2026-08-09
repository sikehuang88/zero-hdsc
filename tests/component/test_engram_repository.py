"""SQLite persistence for typed directed ENGRAM edges."""

from __future__ import annotations

from pathlib import Path

from ssa.config import DatabaseConfig
from ssa.domain.engram import EngramEdge, EngramNode, EngramRelation
from ssa.storage.database import Database
from ssa.storage.engram_repository import EngramRepository


def test_engram_store_is_idempotent_directed_and_bitemporal(tmp_path: Path) -> None:
    database = Database(DatabaseConfig(path=str(tmp_path / "engram.db")))
    database.initialize()
    repository = EngramRepository(database.connection)
    try:
        for node_id in ("A", "B", "C"):
            repository.insert_node(
                EngramNode(
                    node_id=node_id,
                    conversation_id="conversation-1",
                    created_at_ms=10,
                )
            )
        forward = EngramEdge(
            edge_id="edge-ab",
            src="A",
            dst="B",
            rel_type=EngramRelation.SEMANTIC,
            support=0.8,
            valid_from_ms=10,
            source_event_id="event-ab",
        )
        expired = EngramEdge(
            edge_id="edge-bc-old",
            src="B",
            dst="C",
            rel_type=EngramRelation.REVISION,
            support=0.6,
            valid_from_ms=10,
            valid_to_ms=20,
            source_event_id="event-bc-old",
        )
        repository.insert_edge(forward, conversation_id="conversation-1")
        repository.insert_edge(forward, conversation_id="conversation-1")
        repository.insert_edge(expired, conversation_id="conversation-1")

        active = repository.active_edges(
            "conversation-1",
            ["A", "B", "C"],
            now_ms=25,
        )

        assert database.schema_version == 25
        assert repository.count("conversation-1") == 2
        assert active == [forward]
        assert repository.active_edges("conversation-1", ["B"], now_ms=15) == [expired]
        assert not repository.active_edges("conversation-1", ["B"], now_ms=25)
        assert not repository.active_edges("conversation-1", ["C"], now_ms=25)
    finally:
        database.close()
