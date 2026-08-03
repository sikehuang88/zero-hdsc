"""Component coverage for deterministic environment perception."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ssa.config import DatabaseConfig, PerceptionConfig
from ssa.domain.appraisal import AppraisalResult
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, compute_content_hash
from ssa.domain.perception import ConversationGap, DayPhase, SituationMode, StoredPerception
from ssa.domain.relationship import RelationshipState
from ssa.domain.state import OrganismState
from ssa.services.perception_service import EnvironmentPerceptionService
from ssa.storage.database import Database
from ssa.storage.perception_repository import SqlitePerceptionRepository


def _service() -> EnvironmentPerceptionService:
    return EnvironmentPerceptionService(
        config=PerceptionConfig(),
        timezone_name="Asia/Shanghai",
        quiet_hours_start="23:00",
        quiet_hours_end="07:00",
    )


def _event(content: str, *, event_id: str = "user-1", now_ms: int = 1_900_000_000_000) -> Event:
    return Event(
        id=event_id,
        correlation_id=f"corr-{event_id}",
        conversation_id="perception-test",
        actor=Actor.USER,
        event_type="user.message",
        source_kind=SourceKind.USER_OBSERVED,
        content=content,
        content_hash=compute_content_hash(content),
        created_at_ms=now_ms,
    )


def test_human_clock_projects_utc_into_local_rhythm_and_gap() -> None:
    now_ms = int(datetime(2026, 7, 26, 15, 30, tzinfo=UTC).timestamp() * 1000)
    result = _service().perceive_time(
        now_ms,
        last_event_ms=now_ms - 5 * 60_000,
        last_user_ms=now_ms - 7 * 3_600_000,
    )

    assert result.local_iso.startswith("2026-07-26T23:30:00+08:00")
    assert result.day_phase == DayPhase.LATE_NIGHT
    assert result.is_quiet_hours is True
    assert result.conversation_gap == ConversationGap.LONG_GAP
    assert result.elapsed_since_last_event_ms == 5 * 60_000
    assert result.elapsed_since_last_user_ms == 7 * 3_600_000
    assert -1.0 <= result.circadian_sin <= 1.0
    assert -1.0 <= result.circadian_cos <= 1.0


def test_negative_self_disclosure_crosses_into_support_mode() -> None:
    event = _event("我现在很难过, 也很孤独, 不知道该怎么办?")
    appraisal = AppraisalResult.neutral().model_copy(
        update={
            "valence_signal": -0.9,
            "arousal_signal": 0.8,
            "urgency": 0.4,
            "certainty": 0.8,
            "relationship_relevance": 0.8,
        }
    )
    relationship = RelationshipState.initial(event.created_at_ms).model_copy(
        update={"closeness": 0.7, "tension": 0.1}
    )

    result = _service().perceive(
        event,
        appraisal,
        OrganismState.initial(event.created_at_ms),
        relationship,
        [],
        [event],
    )

    assert result.primary_mode == SituationMode.SUPPORT
    assert result.semantic.self_disclosure >= 0.7
    assert result.affect.negative == pytest.approx(0.9)
    assert result.cross_terms["semantic_affect"] > 0.5
    assert result.posture.warmth > result.posture.directness
    assert sum(result.mode_distribution.values()) == pytest.approx(1.0)


def test_explicit_task_crosses_into_action_mode_deterministically() -> None:
    event = _event("请帮我实现这个模块, 并开始写测试.")
    appraisal = AppraisalResult.neutral().model_copy(
        update={
            "novelty": 0.7,
            "controllability": 0.9,
            "certainty": 0.9,
            "urgency": 0.8,
            "valence_signal": 0.1,
            "arousal_signal": 0.4,
        }
    )
    organism = OrganismState.initial(event.created_at_ms)
    relationship = RelationshipState.initial(event.created_at_ms)

    first = _service().perceive(
        event,
        appraisal,
        organism,
        relationship,
        [],
        [event],
    )
    second = _service().perceive(
        event,
        appraisal,
        organism,
        relationship,
        [],
        [event],
    )

    assert first == second
    assert first.primary_mode == SituationMode.ACT
    assert first.semantic.directive >= 0.7
    assert first.cross_terms["action_readiness"] > 0.5
    assert first.posture.directness > first.posture.warmth


def test_perception_repository_round_trip(tmp_path: Path) -> None:
    database = Database(DatabaseConfig(path=str(tmp_path / "perception.db")))
    database.initialize()
    event = _event("请帮我看看现在的情况。")
    database.connection.execute(
        """
        INSERT INTO events (
            id, correlation_id, conversation_id, actor, event_type,
            source_kind, content, content_hash, metadata_json, created_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}', ?)
        """,
        (
            event.id,
            event.correlation_id,
            event.conversation_id,
            event.actor.value,
            event.event_type,
            event.source_kind.value,
            event.content,
            event.content_hash,
            event.created_at_ms,
        ),
    )
    result = _service().perceive(
        event,
        AppraisalResult.neutral(),
        OrganismState.initial(event.created_at_ms),
        RelationshipState.initial(event.created_at_ms),
        [],
        [event],
    )
    repository = SqlitePerceptionRepository(database.connection)

    try:
        stored = repository.insert(
            StoredPerception(
                id="perception-1",
                correlation_id=event.correlation_id,
                conversation_id=event.conversation_id,
                query_event_id=event.id,
                result=result,
                created_at_ms=event.created_at_ms,
            )
        )

        assert repository.find_by_query_event(event.id) == stored
        assert repository.latest_by_conversation(event.conversation_id) == stored
    finally:
        database.close()
