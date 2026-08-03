"""Twenty-four-hour virtual-clock acceptance for the autonomous runtime."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ssa.adapters.embedding import FakeEmbeddingService
from ssa.adapters.llm import FakeLLMAdapter
from ssa.clock import FrozenClock
from ssa.config import (
    DatabaseConfig,
    ExternalTruthConfig,
    InitiativeConfig,
    Settings,
    WorldConfig,
)
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import IncomingSignal, normalize_signal
from ssa.domain.learning import LearningProposalStatus, ReflectionRunStatus
from ssa.domain.lifecycle import Initiative, InitiativeStatus, OutboxMessage
from ssa.domain.relationship import RelationshipState
from ssa.domain.state import OrganismState
from ssa.domain.traces import Trace
from ssa.ids import SequentialIdGenerator
from ssa.runtime.autonomous import AutonomousRuntime
from ssa.storage.database import Database
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.lifecycle_repository import InitiativeRepository, OutboxRepository
from ssa.storage.trace_repository import SqliteTraceRepository, trace_vec_rowid


@pytest.mark.asyncio
async def test_autonomous_runtime_converges_across_24_virtual_hours(tmp_path: Path) -> None:
    start_ms = int(datetime(2030, 3, 5, 8, 0, tzinfo=UTC).timestamp() * 1000)
    settings = Settings(
        database=DatabaseConfig(path=str(tmp_path / "soak.db")),
        initiative=InitiativeConfig(
            daily_limit=3,
            cooldown_minutes=30,
            quiet_hours_start="00:00",
            quiet_hours_end="00:00",
            min_urgency=1.0,
            min_gap_hours=1_000,
        ),
        world=WorldConfig(observe_interval_seconds=300),
        external_truth=ExternalTruthConfig(enabled=False),
    )
    database = Database(settings.database)
    database.initialize()
    clock = FrozenClock(start_ms)
    seed_ids = SequentialIdGenerator("soak-seed")
    events = SqliteEventRepository(database.connection)
    traces = SqliteTraceRepository(database.connection)
    initiatives = InitiativeRepository(database.connection)
    outbox = OutboxRepository(database.connection)
    embedding = FakeEmbeddingService(settings.embedding)
    conversation_id = "conversation-soak"

    user_event = events.append(
        normalize_signal(
            IncomingSignal(
                actor=Actor.USER,
                signal_type="user.message",
                content="Keep our continuity journal alive while I am away.",
                channel="soak",
                channel_message_id=seed_ids.new(),
                conversation_id=conversation_id,
            ),
            event_id=seed_ids.new(),
            correlation_id=seed_ids.new(),
            now_ms=start_ms - 8 * 3_600_000,
            source_kind=SourceKind.USER_OBSERVED,
        )
    )
    trace_id = seed_ids.new()
    trace = Trace(
        id=trace_id,
        conversation_id=conversation_id,
        correlation_id=user_event.correlation_id,
        input_event_id=user_event.id,
        content=user_event.content,
        source_kind=SourceKind.SYSTEM_DERIVED,
        importance=0.9,
        valence=0.5,
        arousal=0.6,
        embedding_model=embedding.model_name,
        embedding_dim=embedding.dimension,
        vec_rowid=trace_vec_rowid(trace_id),
        created_at_ms=user_event.created_at_ms,
    )
    traces.insert(trace, embedding.embed_one(trace.content).as_bytes())

    opening = initiatives.insert(
        Initiative(
            id=seed_ids.new(),
            conversation_id=conversation_id,
            motive="continuity_check",
            intent="initiate",
            content_draft="I will keep a small light on for our journal.",
            urgency=0.9,
            decision_score=0.8,
            status=InitiativeStatus.QUEUED,
            earliest_send_at_ms=start_ms,
            correlation_id=seed_ids.new(),
            source_event_ids=[user_event.id],
            source_trace_ids=[trace.id],
            decision={"topic": "continuity journal"},
            channel="local",
            dedup_key="soak-opening",
            created_at_ms=start_ms,
            updated_at_ms=start_ms,
        )
    )
    outbox.enqueue(
        OutboxMessage(
            id=seed_ids.new(),
            correlation_id=opening.correlation_id or seed_ids.new(),
            channel="local",
            recipient=conversation_id,
            payload={"content": opening.content_draft},
            next_attempt_at_ms=start_ms,
            created_at_ms=start_ms,
            initiative_id=opening.id,
            dedup_key=f"outbox:{opening.id}",
        )
    )

    llm = FakeLLMAdapter()
    runtime = AutonomousRuntime(
        database=database,
        llm=llm,
        settings=settings,
        conversation_id=conversation_id,
        clock=clock,
        embedding=embedding,
    )
    try:
        runtime.bootstrap()
        first = await runtime.tick()
        assert len(first.proactive_events) == 1

        for _step in range(24 * 12):
            clock.advance_ms(5 * 60_000)
            tick = await runtime.tick()
            assert tick.lifecycle.failed == 0
            assert tick.lifecycle.dead_lettered == 0

        job_counts = database.connection.execute(
            """
            SELECT COUNT(*) AS total, COUNT(DISTINCT dedup_key) AS unique_keys
            FROM scheduled_jobs
            """
        ).fetchone()
        assert job_counts["total"] == 14
        assert job_counts["unique_keys"] == 14

        outbox_counts = database.connection.execute(
            """
            SELECT COUNT(*) AS total, COUNT(DISTINCT dedup_key) AS unique_keys,
                   SUM(status = 'delivered') AS delivered
            FROM outbox
            """
        ).fetchone()
        assert outbox_counts["total"] == outbox_counts["unique_keys"]
        assert outbox_counts["total"] == outbox_counts["delivered"] == 3
        proactive_count = database.connection.execute(
            "SELECT COUNT(*) AS c FROM events WHERE event_type = 'agent.initiative'"
        ).fetchone()["c"]
        assert proactive_count == outbox_counts["delivered"]

        latest_activations = traces.latest_activations(conversation_id)
        assert latest_activations
        assert sum(item.score for item in latest_activations) <= 1.0 + 1e-9

        offline = runtime.offline_episodes.recent_episodes(conversation_id)
        assert offline
        assert all(item.status.value == "completed" for item in offline)
        assert all(runtime.offline_episodes.artifacts_for_episode(item.id) for item in offline)
        offline_state_updates = database.connection.execute(
            """
            SELECT COUNT(*) AS c
            FROM state_snapshots AS state
            JOIN events AS event ON event.id = state.cause_event_id
            WHERE event.event_type = 'lifecycle.offline_episode'
            """
        ).fetchone()["c"]
        assert offline_state_updates >= 1

        reflection_runs = runtime.learning.recent_runs(
            conversation_id,
            set(ReflectionRunStatus),
            limit=100,
        )
        assert len(reflection_runs) == len(offline)
        assert all(run.status == ReflectionRunStatus.COMPLETED for run in reflection_runs)
        proposals = runtime.learning.list_proposals(
            conversation_id,
            set(LearningProposalStatus),
            limit=200,
        )
        assert proposals
        assert all(runtime.learning.evidence_for_proposal(item.id) for item in proposals)
        assert any(item.status == LearningProposalStatus.APPLIED for item in proposals)
        decision_count = database.connection.execute(
            "SELECT COUNT(*) AS c FROM consolidation_decisions"
        ).fetchone()["c"]
        assert decision_count >= len(reflection_runs)
        run_count_before_restart = len(reflection_runs)
        proposal_count_before_restart = len(proposals)

        goals = runtime.goals.list_open(conversation_id)
        assert len(goals) == 1
        assert goals[0].progress > 0.0
        assert runtime.contacts.active(conversation_id) == []

        restarted = AutonomousRuntime(
            database=database,
            llm=llm,
            settings=settings,
            conversation_id=conversation_id,
            clock=clock,
            embedding=embedding,
        )
        assert len(restarted.bootstrap()) == 14
        restart_job_count = database.connection.execute(
            "SELECT COUNT(*) AS c FROM scheduled_jobs"
        ).fetchone()["c"]
        assert restart_job_count == 14
        assert (
            restarted.reflective_service.schedule_reflections(
                conversation_id,
                restarted.states.latest() or OrganismState.initial(clock.now_ms()),
                restarted.relationships.latest() or RelationshipState.initial(clock.now_ms()),
                restarted.goals.list_open(conversation_id),
            ).runs
            == ()
        )
        restarted.reflective_service.consolidate(conversation_id)
        assert (
            len(
                restarted.learning.recent_runs(
                    conversation_id,
                    set(ReflectionRunStatus),
                    limit=100,
                )
            )
            == run_count_before_restart
        )
        assert (
            len(
                restarted.learning.list_proposals(
                    conversation_id,
                    set(LearningProposalStatus),
                    limit=200,
                )
            )
            == proposal_count_before_restart
        )
    finally:
        database.close()
