"""Reflection scheduling, consolidation, experiments, and outcome feedback."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssa.adapters.embedding import FakeEmbeddingService
from ssa.adapters.llm import FakeLLMAdapter
from ssa.clock import FrozenClock
from ssa.config import DatabaseConfig, ReflectiveLearningConfig, Settings
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, IncomingSignal, normalize_signal
from ssa.domain.learning import (
    EvidenceReferenceKind,
    EvidenceRelation,
    LearningProposalStatus,
    LearningProposalType,
    ProposalEvidence,
    ReflectionRun,
    ReflectionRunStatus,
    ReflectionTriggerKind,
)
from ssa.domain.lifecycle import (
    OfflineActionKind,
    OfflineArtifact,
    OfflineEpisode,
    OfflineEpisodeStatus,
)
from ssa.domain.relationship import RelationshipState
from ssa.domain.state import OrganismState
from ssa.domain.traces import Trace
from ssa.ids import SequentialIdGenerator
from ssa.services.memory_write_service import MemoryWriteService
from ssa.services.reflective_learning_service import ReflectiveLearningService
from ssa.storage.database import Database
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.learning_repository import LearningVersionConflict, SqliteLearningRepository
from ssa.storage.lifecycle_repository import OfflineAgencyRepository
from ssa.storage.memory_repository import SqliteMemoryRepository
from ssa.storage.trace_repository import SqliteTraceRepository, trace_vec_rowid


def _setup(tmp_path: Path) -> dict[str, object]:
    settings = Settings(
        database=DatabaseConfig(path=str(tmp_path / "reflective.db")),
        reflective_learning=ReflectiveLearningConfig(
            schedule_interval_minutes=1,
            consolidation_interval_minutes=1,
            outcome_interval_minutes=1,
            experiment_duration_hours=1,
        ),
    )
    database = Database(settings.database)
    database.initialize()
    clock = FrozenClock(1_900_000_000_000)
    ids = SequentialIdGenerator("learning")
    events = SqliteEventRepository(database.connection)
    traces = SqliteTraceRepository(database.connection)
    offline = OfflineAgencyRepository(database.connection)
    learning = SqliteLearningRepository(database.connection)
    memories = SqliteMemoryRepository(database.connection)
    embedding = FakeEmbeddingService(settings.embedding)
    memory_write = MemoryWriteService(
        FakeLLMAdapter(),
        embedding,
        memories,
        events.get,
        ids,
        clock,
        settings.retrieval,
        embedding_model_name=settings.embedding.model,
        embedding_dim=settings.embedding.dim,
    )
    service = ReflectiveLearningService(
        learning=learning,
        offline=offline,
        events=events,
        traces=traces,
        memory_write=memory_write,
        clock=clock,
        ids=ids,
        config=settings.reflective_learning,
    )
    return {
        "database": database,
        "settings": settings,
        "clock": clock,
        "ids": ids,
        "events": events,
        "traces": traces,
        "offline": offline,
        "learning": learning,
        "memories": memories,
        "embedding": embedding,
        "service": service,
    }


def _event(
    setup: dict[str, object],
    content: str,
    *,
    actor: Actor = Actor.USER,
    conversation_id: str = "conversation-1",
) -> Event:
    ids = setup["ids"]
    clock = setup["clock"]
    events = setup["events"]
    assert isinstance(ids, SequentialIdGenerator)
    assert isinstance(clock, FrozenClock)
    assert isinstance(events, SqliteEventRepository)
    return events.append(
        normalize_signal(
            IncomingSignal(
                actor=actor,
                signal_type=f"{actor.value}.message",
                content=content,
                channel="fixture",
                channel_message_id=ids.new(),
                conversation_id=conversation_id,
            ),
            event_id=ids.new(),
            correlation_id=ids.new(),
            now_ms=clock.now_ms(),
            source_kind=(
                SourceKind.USER_OBSERVED if actor == Actor.USER else SourceKind.AGENT_OUTPUT
            ),
        )
    )


def _seed_artifact(setup: dict[str, object]) -> tuple[OfflineEpisode, OfflineArtifact]:
    ids = setup["ids"]
    clock = setup["clock"]
    traces = setup["traces"]
    offline = setup["offline"]
    embedding = setup["embedding"]
    assert isinstance(ids, SequentialIdGenerator)
    assert isinstance(clock, FrozenClock)
    assert isinstance(traces, SqliteTraceRepository)
    assert isinstance(offline, OfflineAgencyRepository)
    assert isinstance(embedding, FakeEmbeddingService)
    user = _event(setup, "I value careful, evidence-linked continuity.")
    trace_id = ids.new()
    trace = traces.insert(
        Trace(
            id=trace_id,
            conversation_id=user.conversation_id,
            correlation_id=user.correlation_id,
            input_event_id=user.id,
            content=user.content,
            source_kind=SourceKind.SYSTEM_DERIVED,
            importance=0.9,
            valence=0.4,
            arousal=0.5,
            embedding_model=embedding.model_name,
            embedding_dim=embedding.dimension,
            vec_rowid=trace_vec_rowid(trace_id),
            created_at_ms=clock.now_ms(),
        ),
        embedding.embed_one(user.content).as_bytes(),
    )
    episode = offline.insert_episode(
        OfflineEpisode(
            id=ids.new(),
            conversation_id=user.conversation_id,
            correlation_id=user.correlation_id,
            dedup_key=f"offline:{trace.id}",
            action_kind=OfflineActionKind.TRACE_REFLECTION,
            motive="Advance evidence-grounded continuity",
            status=OfflineEpisodeStatus.COMPLETED,
            source_event_ids=[user.id],
            source_trace_ids=[trace.id],
            summary="Reviewed one continuity trace.",
            started_at_ms=clock.now_ms(),
            completed_at_ms=clock.now_ms(),
        )
    )
    artifact = offline.insert_artifact(
        OfflineArtifact(
            id=ids.new(),
            episode_id=episode.id,
            artifact_type="trace_reflection",
            title="Prefer evidence before identity claims",
            content=(
                "This reflection notes that future continuity claims should remain linked "
                "to inspectable evidence."
            ),
            evidence_event_ids=[user.id],
            evidence_trace_ids=[trace.id],
            metadata={
                "reflection_source": "fixture",
                "questions": ["Does later feedback support this approach?"],
                "next_action": "Cite a relevant shared trace when making continuity claims.",
            },
            created_at_ms=clock.now_ms(),
        )
    )
    return episode, artifact


def test_reflection_consolidation_is_evidence_linked_and_idempotent(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    database = setup["database"]
    service = setup["service"]
    learning = setup["learning"]
    memories = setup["memories"]
    clock = setup["clock"]
    assert isinstance(database, Database)
    assert isinstance(service, ReflectiveLearningService)
    assert isinstance(learning, SqliteLearningRepository)
    assert isinstance(memories, SqliteMemoryRepository)
    assert isinstance(clock, FrozenClock)
    try:
        _seed_artifact(setup)
        organism = OrganismState.initial(clock.now_ms())
        relationship = RelationshipState.initial(clock.now_ms())

        scheduled = service.schedule_reflections("conversation-1", organism, relationship, [])
        consolidated = service.consolidate("conversation-1")

        assert len(scheduled.runs) == 1
        assert scheduled.runs[0].status == ReflectionRunStatus.COMPLETED
        assert scheduled.runs[0].critic_score is not None
        assert len(consolidated.decisions) == 2
        proposals = learning.proposals_for_run(scheduled.runs[0].id)
        by_type = {proposal.proposal_type: proposal for proposal in proposals}
        memory = by_type[LearningProposalType.MEMORY_CANDIDATE]
        policy = by_type[LearningProposalType.POLICY_PROPOSAL]
        assert memory.status == LearningProposalStatus.APPLIED
        assert memory.target_kind == "memory"
        assert policy.status == LearningProposalStatus.UNDER_TEST
        assert policy.target_kind == "behavior_experiment"
        assert learning.evidence_for_proposal(memory.id)
        stored_memory = memories.get(memory.target_id or "")
        assert stored_memory is not None
        assert stored_memory.source_kind == SourceKind.MODEL_INFERENCE
        assert stored_memory.confidence <= 0.6
        assert "active experiment" in service.context_summary("conversation-1")

        assert service.schedule_reflections("conversation-1", organism, relationship, []).runs == ()
        service.consolidate("conversation-1")
        assert len(learning.proposals_for_run(scheduled.runs[0].id)) == 2
        assert memories.count() == 1
    finally:
        database.close()


@pytest.mark.parametrize(
    ("feedback", "expected_status"),
    [
        ("谢谢, 这个方法很好而且有用。", LearningProposalStatus.CONFIRMED),
        ("别这样, 这个方法没用。", LearningProposalStatus.ROLLED_BACK),
    ],
)
def test_policy_outcome_confirms_or_rolls_back(
    tmp_path: Path,
    feedback: str,
    expected_status: LearningProposalStatus,
) -> None:
    setup = _setup(tmp_path)
    database = setup["database"]
    service = setup["service"]
    learning = setup["learning"]
    clock = setup["clock"]
    assert isinstance(database, Database)
    assert isinstance(service, ReflectiveLearningService)
    assert isinstance(learning, SqliteLearningRepository)
    assert isinstance(clock, FrozenClock)
    try:
        _seed_artifact(setup)
        service.schedule_reflections(
            "conversation-1",
            OrganismState.initial(clock.now_ms()),
            RelationshipState.initial(clock.now_ms()),
            [],
        )
        service.consolidate("conversation-1")
        policy = learning.list_proposals(
            "conversation-1",
            {LearningProposalStatus.UNDER_TEST},
            proposal_type=LearningProposalType.POLICY_PROPOSAL,
        )[0]
        experiment = learning.get_experiment(policy.target_id or "")
        assert experiment is not None
        clock.set_ms(experiment.due_at_ms)
        _event(setup, feedback)

        evaluated = service.evaluate_outcomes("conversation-1")

        assert len(evaluated.outcomes) == 1
        assert evaluated.experiments[0].status.value == "completed"
        updated = learning.get_proposal(policy.id)
        assert updated is not None
        assert updated.status == expected_status
        assert len(learning.decisions_for_proposal(policy.id)) == 2
        context = service.context_summary("conversation-1")
        if expected_status == LearningProposalStatus.CONFIRMED:
            assert "confirmed proposal" in context
        else:
            assert policy.id not in context
    finally:
        database.close()


def test_repository_rejects_unfinished_run_and_cross_conversation_evidence(
    tmp_path: Path,
) -> None:
    setup = _setup(tmp_path)
    database = setup["database"]
    learning = setup["learning"]
    ids = setup["ids"]
    clock = setup["clock"]
    assert isinstance(database, Database)
    assert isinstance(learning, SqliteLearningRepository)
    assert isinstance(ids, SequentialIdGenerator)
    assert isinstance(clock, FrozenClock)
    try:
        run = learning.insert_run(
            ReflectionRun(
                id=ids.new(),
                conversation_id="conversation-1",
                correlation_id=ids.new(),
                dedup_key="planned-run",
                trigger_kind=ReflectionTriggerKind.SCHEDULED,
                priority_score=0.8,
                status=ReflectionRunStatus.PLANNED,
                started_at_ms=clock.now_ms(),
            )
        )
        from ssa.domain.learning import LearningProposal

        proposal = LearningProposal(
            id=ids.new(),
            run_id=run.id,
            conversation_id=run.conversation_id,
            dedup_key="invalid-proposal",
            proposal_type=LearningProposalType.MEMORY_CANDIDATE,
            title="Invalid",
            content="An unfinished run must not produce this.",
            rationale="fixture",
            confidence=0.5,
            critic_score=0.8,
            created_at_ms=clock.now_ms(),
            updated_at_ms=clock.now_ms(),
        )
        with pytest.raises(ValueError, match="completed reflection"):
            learning.insert_proposal(proposal)

        completed = learning.insert_run(
            ReflectionRun(
                id=ids.new(),
                conversation_id="conversation-1",
                correlation_id=ids.new(),
                dedup_key="completed-run",
                trigger_kind=ReflectionTriggerKind.SCHEDULED,
                priority_score=0.8,
                status=ReflectionRunStatus.COMPLETED,
                reflection_text="reviewed",
                critique_text="passed",
                critic_score=0.8,
                started_at_ms=clock.now_ms(),
                completed_at_ms=clock.now_ms(),
            )
        )
        valid = learning.insert_proposal(proposal.model_copy(update={"run_id": completed.id}))
        other = _event(setup, "other conversation", conversation_id="conversation-2")
        with pytest.raises(ValueError, match="unknown event evidence"):
            learning.add_evidence(
                ProposalEvidence(
                    proposal_id=valid.id,
                    reference_kind=EvidenceReferenceKind.EVENT,
                    reference_id=other.id,
                    relation=EvidenceRelation.SUPPORTS,
                    provenance_kind=SourceKind.USER_OBSERVED,
                    trust_weight=1.0,
                    likelihood_ratio=1.5,
                    content_hash=other.content_hash,
                    recorded_at_ms=clock.now_ms(),
                )
            )
        approved = valid.model_copy(
            update={
                "status": LearningProposalStatus.APPROVED,
                "version": 2,
                "updated_at_ms": clock.now_ms(),
            }
        )
        learning.update_proposal(1, approved)
        with pytest.raises(LearningVersionConflict):
            learning.update_proposal(1, approved)
    finally:
        database.close()
