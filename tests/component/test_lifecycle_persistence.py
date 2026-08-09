"""Persistence contracts for the autonomous lifecycle kernel."""

from __future__ import annotations

from pathlib import Path

from ssa.config import DatabaseConfig
from ssa.domain.lifecycle import (
    CausalTraceRecord,
    ContactEpisode,
    EmotionEpisode,
    EmotionType,
    Goal,
    GoalOwner,
    GoalStatus,
    GoalStep,
    Initiative,
    InitiativeStatus,
    InnerLifeMode,
    InnerLoopState,
    JobStatus,
    OutboxMessage,
    OutboxStatus,
    ScheduledJob,
    WorldObservation,
)
from ssa.storage.database import Database
from ssa.storage.lifecycle_repository import (
    CausalTraceRepository,
    ContactEpisodeRepository,
    EmotionEpisodeRepository,
    GoalRepository,
    InitiativeRepository,
    InnerLoopRepository,
    OutboxRepository,
    ScheduledJobRepository,
    WorldObservationRepository,
)


def _database(tmp_path: Path) -> Database:
    database = Database(DatabaseConfig(path=str(tmp_path / "lifecycle.db")))
    database.initialize()
    return database


def test_lifecycle_migration_and_goal_repository(tmp_path: Path) -> None:
    database = _database(tmp_path)
    goals = GoalRepository(database.connection)
    goal = Goal(
        id="goal-1",
        conversation_id="conversation-1",
        owner=GoalOwner.SELF,
        title="Maintain a continuity journal",
        motive="Preserve meaningful shared episodes",
        success_criteria=["Record seven reviewed episodes"],
        priority=0.8,
        status=GoalStatus.ACTIVE,
        next_action_at_ms=1_000,
        created_at_ms=100,
        updated_at_ms=100,
    )

    try:
        assert database.schema_version == 22
        assert goals.insert(goal) == goal
        step = GoalStep(
            id="step-1",
            goal_id=goal.id,
            action="Review one episode",
            result="Reviewed a durable trace",
            metadata={"trace_ids": ["trace-1"]},
            created_at_ms=1_000,
        )
        goals.add_step(step)
        updated = goal.model_copy(update={"progress": 0.25, "updated_at_ms": 1_000})
        goals.update(updated)

        assert goals.get(goal.id) == updated
        assert goals.list_due("conversation-1", 1_000) == [updated]
        assert goals.steps(goal.id) == [step]
    finally:
        database.close()


def test_job_leasing_is_deduplicated_and_recovers_expired_lease(tmp_path: Path) -> None:
    database = _database(tmp_path)
    jobs = ScheduledJobRepository(database.connection)
    job = ScheduledJob(
        id="job-1",
        conversation_id="conversation-1",
        job_type="state.refresh",
        dedup_key="state.refresh:conversation-1",
        due_at_ms=1_000,
        created_at_ms=100,
        updated_at_ms=100,
    )

    try:
        assert jobs.enqueue(job) == job
        duplicate = job.model_copy(update={"id": "job-duplicate"})
        assert jobs.enqueue(duplicate).id == job.id

        leased = jobs.lease_due(1_000, lease_ms=100)
        assert len(leased) == 1
        assert leased[0].status == JobStatus.RUNNING
        assert leased[0].attempt_count == 1
        assert jobs.lease_due(1_050) == []

        recovered = jobs.lease_due(1_101, lease_ms=100)
        assert len(recovered) == 1
        assert recovered[0].attempt_count == 2
        pending = jobs.reschedule(job.id, 2_000, 1_102)
        assert pending.status == JobStatus.PENDING
        assert pending.due_at_ms == 2_000
    finally:
        database.close()


def test_inner_loop_state_upsert_survives_restart_boundary(tmp_path: Path) -> None:
    database = _database(tmp_path)
    states = InnerLoopRepository(database.connection)
    state = InnerLoopState(
        conversation_id="conversation-1",
        version=1,
        mode=InnerLifeMode.WAITING,
        reply_expectation=0.8,
        concern=0.3,
        curiosity=0.6,
        connection_pressure=0.5,
        uncertainty=0.4,
        offline_readiness=0.2,
        wait_started_at_ms=1_000,
        wait_deadline_at_ms=4_000,
        last_heartbeat_at_ms=1_000,
        transition_reason="context_wait",
        updated_at_ms=1_000,
    )
    try:
        assert states.upsert(state) == state
        offline = state.model_copy(
            update={
                "version": 2,
                "mode": InnerLifeMode.OFFLINE,
                "offline_readiness": 0.7,
                "last_heartbeat_at_ms": 4_001,
                "transition_reason": "context_wait_elapsed",
                "updated_at_ms": 4_001,
            }
        )
        states.upsert(offline)

        assert InnerLoopRepository(database.connection).get("conversation-1") == offline
    finally:
        database.close()


def test_initiative_outbox_emotion_contact_and_world_round_trip(tmp_path: Path) -> None:
    database = _database(tmp_path)
    initiatives = InitiativeRepository(database.connection)
    outbox = OutboxRepository(database.connection)
    emotions = EmotionEpisodeRepository(database.connection)
    contacts = ContactEpisodeRepository(database.connection)
    world = WorldObservationRepository(database.connection)

    initiative = Initiative(
        id="initiative-1",
        conversation_id="conversation-1",
        correlation_id="correlation-1",
        motive="share a recalled episode",
        intent="initiate",
        content_draft="I remembered our earlier conversation.",
        urgency=0.8,
        decision_score=0.72,
        status=InitiativeStatus.APPROVED,
        earliest_send_at_ms=1_000,
        expires_at_ms=10_000,
        source_trace_ids=["trace-1"],
        decision={"relevance": 0.9},
        dedup_key="initiative:trace-1",
        created_at_ms=100,
        updated_at_ms=100,
    )
    message = OutboxMessage(
        id="outbox-1",
        correlation_id="correlation-1",
        channel="local",
        recipient="conversation-1",
        payload={"text": initiative.content_draft},
        initiative_id=initiative.id,
        dedup_key="outbox:initiative-1",
        next_attempt_at_ms=1_000,
        created_at_ms=100,
    )
    emotion = EmotionEpisode(
        id="emotion-1",
        conversation_id="conversation-1",
        emotion_type=EmotionType.ATTACHMENT,
        target="owner",
        action_tendency="seek contact",
        intensity=0.75,
        inhibition=0.3,
        half_life_minutes=240,
        source_trace_ids=["trace-1"],
        created_at_ms=100,
        updated_at_ms=100,
    )
    contact = ContactEpisode(
        id="contact-1",
        conversation_id="conversation-1",
        correlation_id="correlation-1",
        motive=initiative.motive,
        topic="earlier conversation",
        hypotheses={"busy": 0.6, "unavailable": 0.3, "avoidance": 0.1},
        source_initiative_id=initiative.id,
        created_at_ms=100,
        updated_at_ms=100,
    )
    observation = WorldObservation(
        id="world-1",
        conversation_id="conversation-1",
        observation_type="clock",
        source="human_clock",
        summary="The local day phase changed to evening.",
        salience=0.3,
        dedup_key="clock:conversation-1:evening:2026-07-26",
        observed_at_ms=1_000,
    )

    try:
        assert initiatives.insert(initiative) == initiative
        assert initiatives.list_ready("conversation-1", 1_000) == [initiative]
        assert outbox.enqueue(message) == message
        claimed = outbox.claim_due(1_000)
        assert len(claimed) == 1
        assert claimed[0].status == OutboxStatus.DELIVERING
        assert emotions.insert(emotion) == emotion
        assert emotions.active("conversation-1") == [emotion]
        assert contacts.insert(contact) == contact
        assert contacts.active("conversation-1") == [contact]
        assert world.insert(observation) == observation
        assert world.insert(observation.model_copy(update={"id": "duplicate"})).id == observation.id
        assert world.recent("conversation-1") == [observation]
    finally:
        database.close()


def test_causal_trace_repository_uses_existing_event_contract(tmp_path: Path) -> None:
    database = _database(tmp_path)
    connection = database.connection
    connection.execute(
        """
        INSERT INTO events (
            id, correlation_id, conversation_id, actor, event_type, source_kind,
            content, content_hash, metadata_json, created_at_ms
        ) VALUES ('event-1', 'correlation-1', 'conversation-1', 'system',
                  'lifecycle.tick', 'system_derived', 'tick', 'hash', '{}', 100)
        """
    )
    traces = CausalTraceRepository(connection)
    trace = CausalTraceRecord(
        id="causal-1",
        correlation_id="correlation-1",
        input_event_id="event-1",
        goal_ids=["goal-1"],
        decision={"action": "wait"},
        created_at_ms=100,
    )

    try:
        assert traces.insert(trace) == trace
        row = connection.execute(
            "SELECT decision_json FROM causal_traces WHERE id = 'causal-1'"
        ).fetchone()
        assert row is not None
        assert '"action": "wait"' in row["decision_json"]
    finally:
        database.close()
