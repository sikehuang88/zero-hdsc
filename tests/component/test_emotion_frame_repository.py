"""Persistence coverage for layered emotion frames and vocal trajectories."""

# Chinese fixtures are intentional.
# ruff: noqa: RUF001

from __future__ import annotations

from pathlib import Path

from ssa.clock import FrozenClock
from ssa.config import DatabaseConfig
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, compute_content_hash
from ssa.domain.relationship import RelationshipState
from ssa.domain.state import OrganismState
from ssa.services.realtime_emotion_service import RealtimeEmotionService
from ssa.storage.database import Database
from ssa.storage.emotion_frame_repository import EmotionFrameRepository
from ssa.storage.event_repository import SqliteEventRepository


def test_emotion_frame_roundtrip_keeps_voice_segments_atomic(tmp_path: Path) -> None:
    database = Database(DatabaseConfig(path=str(tmp_path / "emotion-frame.db")))
    database.initialize()
    clock = FrozenClock(1_900_000_000_000)
    event = Event(
        id="event-1",
        correlation_id="correlation-1",
        conversation_id="conversation-1",
        actor=Actor.USER,
        event_type="user.message",
        source_kind=SourceKind.USER_OBSERVED,
        content="你真走啊",
        content_hash=compute_content_hash("你真走啊"),
        created_at_ms=clock.now_ms(),
    )

    try:
        SqliteEventRepository(database.connection).append(event)
        frame = RealtimeEmotionService(clock).plan(
            event,
            OrganismState.initial(clock.now_ms()),
            RelationshipState.initial(clock.now_ms()),
            [],
            [],
        ).frame
        repository = EmotionFrameRepository(database.connection)

        stored = repository.insert(frame)
        loaded = repository.find_by_correlation(event.correlation_id)

        assert loaded == stored
        assert repository.recent(event.conversation_id) == [stored]
        rows = database.connection.execute(
            """
            SELECT sequence, recipe, performance_json
            FROM voice_performance_segments
            WHERE emotion_frame_id = ?
            ORDER BY sequence
            """,
            (frame.id,),
        ).fetchall()
        assert [row["sequence"] for row in rows] == list(range(len(frame.voice_segments)))
        assert len({row["recipe"] for row in rows}) >= 2
        assert all('"pitch_stability"' in row["performance_json"] for row in rows)
    finally:
        database.close()
