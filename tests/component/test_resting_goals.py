"""Background goal and resting-state behavior with no user-triggered turn."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssa.adapters.embedding import FakeEmbeddingService
from ssa.clock import FrozenClock
from ssa.config import DatabaseConfig, Settings
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, compute_content_hash
from ssa.domain.lifecycle import EmotionType, GoalOwner, GoalStatus
from ssa.domain.relationship import RelationshipState
from ssa.domain.state import OrganismState
from ssa.domain.traces import Trace, TraceLink
from ssa.ids import SequentialIdGenerator
from ssa.services.goal_service import GoalService
from ssa.services.resting_state_service import RestingStateService
from ssa.storage.database import Database
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.lifecycle_repository import EmotionEpisodeRepository, GoalRepository
from ssa.storage.trace_repository import SqliteTraceRepository, trace_vec_rowid


def _event(
    index: int,
    conversation_id: str,
    created_at_ms: int,
) -> Event:
    content = f"shared episode {index}"
    return Event(
        id=f"event-{index}",
        correlation_id=f"correlation-{index}",
        conversation_id=conversation_id,
        actor=Actor.USER,
        event_type="user.message",
        source_kind=SourceKind.USER_OBSERVED,
        content=content,
        content_hash=compute_content_hash(content),
        created_at_ms=created_at_ms,
    )


def _insert_trace(
    repository: SqliteTraceRepository,
    embedding: FakeEmbeddingService,
    event: Event,
    *,
    importance: float,
    arousal: float,
) -> Trace:
    trace_id = f"trace-{event.id}"
    trace = Trace(
        id=trace_id,
        conversation_id=event.conversation_id,
        correlation_id=event.correlation_id,
        input_event_id=event.id,
        content=event.content,
        source_kind=SourceKind.SYSTEM_DERIVED,
        importance=importance,
        valence=0.3,
        arousal=arousal,
        embedding_model=embedding.model_name,
        embedding_dim=embedding.dimension,
        vec_rowid=trace_vec_rowid(trace_id),
        created_at_ms=event.created_at_ms,
    )
    return repository.insert(trace, embedding.embed_one(trace.content).as_bytes())


def _setup(
    tmp_path: Path,
) -> tuple[
    Database,
    Settings,
    FrozenClock,
    SequentialIdGenerator,
    SqliteEventRepository,
    SqliteTraceRepository,
    GoalRepository,
    EmotionEpisodeRepository,
]:
    settings = Settings(database=DatabaseConfig(path=str(tmp_path / "resting.db")))
    database = Database(settings.database)
    database.initialize()
    return (
        database,
        settings,
        FrozenClock(1_900_000_000_000),
        SequentialIdGenerator("resting"),
        SqliteEventRepository(database.connection),
        SqliteTraceRepository(database.connection),
        GoalRepository(database.connection),
        EmotionEpisodeRepository(database.connection),
    )


def test_resting_state_uses_directed_radiation_and_bounded_mass(tmp_path: Path) -> None:
    database, settings, clock, ids, events, traces, _goals, emotions = _setup(tmp_path)
    embedding = FakeEmbeddingService(settings.embedding)
    conversation_id = "conversation-1"

    try:
        inserted: list[Trace] = []
        for index, (importance, arousal) in enumerate(
            [(1.0, 0.9), (0.05, 0.0), (0.7, 0.4), (0.65, 0.3)],
            1,
        ):
            event = events.append(_event(index, conversation_id, clock.now_ms() - index * 1_000))
            inserted.append(
                _insert_trace(
                    traces,
                    embedding,
                    event,
                    importance=importance,
                    arousal=arousal,
                )
            )
        traces.add_link(
            TraceLink(
                source_trace_id=inserted[0].id,
                target_trace_id=inserted[1].id,
                link_type="temporal",
                weight=1.0,
                created_at_ms=clock.now_ms(),
            )
        )
        service = RestingStateService(
            traces=traces,
            emotions=emotions,
            events=events,
            clock=clock,
            ids=ids,
            capacity=4,
        )
        organism = OrganismState.initial(clock.now_ms()).model_copy(
            update={"connection_need": 0.9, "curiosity": 0.9}
        )

        result = service.step(
            conversation_id=conversation_id,
            organism=organism,
            relationship=RelationshipState.initial(clock.now_ms()),
            goals=[],
        )

        assert result.event is not None
        assert result.event.event_type == "lifecycle.resting_activation"
        assert result.active_mass <= 1.0
        assert result.null_mass == pytest.approx(1.0 - result.active_mass)
        assert sum(item.score for item in result.activations) <= 1.0 + 1e-9
        target = next(item for item in result.activations if item.trace.id == inserted[1].id)
        assert target.activation_kind == "radiation"
        assert {episode.emotion_type for episode in result.emotions} == {
            EmotionType.ATTACHMENT,
            EmotionType.CURIOSITY,
        }
    finally:
        database.close()


def test_self_goal_advances_across_seven_virtual_days(tmp_path: Path) -> None:
    database, settings, clock, ids, events, traces, goals, _emotions = _setup(tmp_path)
    embedding = FakeEmbeddingService(settings.embedding)
    conversation_id = "conversation-1"
    service = GoalService(
        goals=goals,
        traces=traces,
        events=events,
        clock=clock,
        ids=ids,
        config=settings.goal,
    )

    try:
        for index in range(1, 8):
            event = events.append(_event(index, conversation_id, clock.now_ms() - index * 1_000))
            _insert_trace(
                traces,
                embedding,
                event,
                importance=0.5,
                arousal=0.3,
            )
        goal = service.ensure_self_project(conversation_id, OrganismState.initial(clock.now_ms()))
        assert goal is not None
        assert goal.owner == GoalOwner.SELF
        assert (
            service.ensure_self_project(conversation_id, OrganismState.initial(clock.now_ms()))
            == goal
        )

        current = goal
        for day in range(7):
            if day:
                clock.advance_ms(86_400_000)
            advanced = service.advance_due(conversation_id)
            assert advanced is not None
            current = advanced

        assert current.progress == 1.0
        assert current.status == GoalStatus.DONE
        assert len(goals.steps(current.id)) == 8
        assert (
            len(
                [
                    event
                    for event in events.recent_by_conversation(conversation_id, limit=30)
                    if event.event_type == "goal.step"
                ]
            )
            == 7
        )
    finally:
        database.close()
