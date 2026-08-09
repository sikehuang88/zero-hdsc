"""Dedicated emotional-memory capture, backfill, and recall coverage."""

from __future__ import annotations

from pathlib import Path

from ssa.clock import FrozenClock
from ssa.config import DatabaseConfig, Settings
from ssa.domain.appraisal import AppraisalResult
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, IncomingSignal, normalize_signal
from ssa.domain.lifecycle import EmotionType
from ssa.domain.traces import ActivatedTrace, Trace
from ssa.hdsc.resonance import RecallState
from ssa.ids import SequentialIdGenerator
from ssa.services.emotional_memory_service import EmotionalMemoryService
from ssa.storage.database import Database
from ssa.storage.emotional_memory_repository import EmotionalMemoryRepository
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.trace_repository import SqliteTraceRepository, trace_vec_rowid


def _event(
    events: SqliteEventRepository,
    ids: SequentialIdGenerator,
    clock: FrozenClock,
    *,
    actor: Actor,
    content: str,
    correlation_id: str,
) -> Event:
    event_id = ids.new()
    return events.append(
        normalize_signal(
            IncomingSignal(
                actor=actor,
                signal_type=f"{actor.value}.message",
                content=content,
                channel="fixture",
                channel_message_id=event_id,
                conversation_id="conversation-1",
            ),
            event_id=event_id,
            correlation_id=correlation_id,
            now_ms=clock.now_ms(),
            source_kind=(
                SourceKind.USER_OBSERVED if actor == Actor.USER else SourceKind.AGENT_OUTPUT
            ),
        )
    )


def _trace(user: Event, agent: Event, *, trace_id: str, valence: float = 0.8) -> Trace:
    return Trace(
        id=trace_id,
        conversation_id=user.conversation_id,
        correlation_id=user.correlation_id,
        input_event_id=user.id,
        output_event_id=agent.id,
        content=f"User: {user.content}\nAssistant: {agent.content}",
        source_kind=SourceKind.AGENT_OUTPUT,
        importance=0.8,
        valence=valence,
        arousal=0.75,
        embedding_model="fixture",
        embedding_dim=4,
        vec_rowid=trace_vec_rowid(trace_id),
        created_at_ms=user.created_at_ms,
    )


def test_emotional_library_captures_and_recalls_subjective_continuity(
    tmp_path: Path,
) -> None:
    settings = Settings(database=DatabaseConfig(path=str(tmp_path / "emotion-library.db")))
    database = Database(settings.database)
    database.initialize()
    clock = FrozenClock(1_900_000_000_000)
    ids = SequentialIdGenerator("emotion-library")
    events = SqliteEventRepository(database.connection)
    traces = SqliteTraceRepository(database.connection)
    repository = EmotionalMemoryRepository(database.connection)
    service = EmotionalMemoryService(repository, clock, ids, settings.emotion_library)

    try:
        correlation_id = ids.new()
        user = _event(
            events,
            ids,
            clock,
            actor=Actor.USER,
            content="我今天很想你，也希望你陪我一会儿。",
            correlation_id=correlation_id,
        )
        agent = _event(
            events,
            ids,
            clock,
            actor=Actor.AGENT,
            content="我在这里，想认真陪着你。",
            correlation_id=correlation_id,
        )
        trace = _trace(user, agent, trace_id="trace-attachment")
        traces.insert(trace, b"\x00" * 16)
        appraisal = AppraisalResult(
            novelty=0.7,
            goal_congruence=0.8,
            controllability=0.7,
            certainty=0.8,
            self_agency=0.3,
            user_agency=0.7,
            external_agency=0.0,
            relationship_relevance=0.95,
            urgency=0.2,
            valence_signal=0.8,
            arousal_signal=0.75,
            explanation="warm attachment",
        )

        captured = service.capture(user, agent, appraisal, trace)
        assert captured is not None
        assert captured.emotion_type == EmotionType.ATTACHMENT
        assert captured.target == "owner relationship"
        assert captured.source_event_ids == [user.id, agent.id]
        assert repository.count("conversation-1") == 1

        recalls = service.recall(
            "conversation-1",
            "刚才说到想你和陪伴",
            [
                ActivatedTrace(
                    trace=trace,
                    rank=1,
                    score=0.9,
                    semantic_similarity=0.9,
                    freshness=1.0,
                    importance_factor=0.8,
                    activation_kind="main",
                )
            ],
        )
        assert len(recalls) == 1
        assert recalls[0].memory.id == captured.id
        assert recalls[0].trace_score == 1.0
        recalled = repository.find_by_origin_trace(trace.id)
        assert recalled is not None
        assert recalled.recall_count == 1

        resonant = service.recall(
            "conversation-1",
            "刚才说到想你和陪伴",
            [
                ActivatedTrace(
                    trace=trace,
                    rank=1,
                    score=0.9,
                    semantic_similarity=0.9,
                    freshness=1.0,
                    importance_factor=0.8,
                    activation_kind="resonance",
                    coupling_mass=0.8,
                    detuning=0.1,
                )
            ],
            state=RecallState(
                valence=0.75,
                arousal=0.7,
                connection_need=0.9,
                situation_mode="reminisce",
            ),
        )
        assert len(resonant) == 1
        assert resonant[0].resonance_ratio > 1.35
        assert resonant[0].damping < 1.0
        assert service.recall(
            "conversation-1",
            "完全没有图传播证据的回忆",
            [],
            state=RecallState(situation_mode="reminisce"),
        ) == []
    finally:
        database.close()


def test_backfill_is_idempotent_and_keeps_low_salience_noise_out(tmp_path: Path) -> None:
    settings = Settings(database=DatabaseConfig(path=str(tmp_path / "emotion-backfill.db")))
    database = Database(settings.database)
    database.initialize()
    clock = FrozenClock(1_900_000_000_000)
    ids = SequentialIdGenerator("emotion-backfill")
    events = SqliteEventRepository(database.connection)
    traces = SqliteTraceRepository(database.connection)
    repository = EmotionalMemoryRepository(database.connection)
    service = EmotionalMemoryService(repository, clock, ids, settings.emotion_library)

    try:
        correlation_id = ids.new()
        user = _event(
            events,
            ids,
            clock,
            actor=Actor.USER,
            content="这次失败让我很难过。",
            correlation_id=correlation_id,
        )
        agent = _event(
            events,
            ids,
            clock,
            actor=Actor.AGENT,
            content="我会陪你重新整理。",
            correlation_id=correlation_id,
        )
        salient = _trace(user, agent, trace_id="trace-concern", valence=-0.8)
        traces.insert(salient, b"\x00" * 16)

        neutral_user = _event(
            events,
            ids,
            clock,
            actor=Actor.USER,
            content="收到。",
            correlation_id=ids.new(),
        )
        neutral_agent = _event(
            events,
            ids,
            clock,
            actor=Actor.AGENT,
            content="好的。",
            correlation_id=neutral_user.correlation_id,
        )
        neutral = _trace(neutral_user, neutral_agent, trace_id="trace-neutral", valence=0.0)
        neutral = neutral.model_copy(update={"importance": 0.1, "arousal": 0.1})
        traces.insert(neutral, b"\x00" * 16)

        assert service.backfill("conversation-1", [salient, neutral]) == 1
        assert service.backfill("conversation-1", [salient, neutral]) == 0
        assert repository.count("conversation-1") == 1
        stored = repository.find_by_origin_trace(salient.id)
        assert stored is not None
        assert stored.emotion_type == EmotionType.CONCERN
    finally:
        database.close()
