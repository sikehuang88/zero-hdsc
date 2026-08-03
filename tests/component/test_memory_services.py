"""Component tests for memory write + retrieval services (M05 + M06).

Uses FakeEmbeddingService and FakeLLMAdapter to avoid network calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ssa.adapters.embedding import FakeEmbeddingService
from ssa.adapters.llm import FakeLLMAdapter
from ssa.clock import FrozenClock
from ssa.config import DatabaseConfig, EmbeddingConfig, RetrievalConfig, ThinkingMode
from ssa.domain.enums import Actor, MemoryType, SourceKind
from ssa.domain.events import Event, IncomingSignal, normalize_signal
from ssa.ids import SequentialIdGenerator
from ssa.services.memory_retrieval_service import MemoryRetrievalService
from ssa.services.memory_write_service import MemoryWriteService
from ssa.storage.database import Database
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.memory_repository import SqliteMemoryRepository

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def setup(tmp_path: Path):
    """Full test setup: db + repos + fake services."""
    db = Database(DatabaseConfig(path=str(tmp_path / "mem.db")))
    db.initialize()
    conn = db.connection

    event_repo = SqliteEventRepository(conn)
    memory_repo = SqliteMemoryRepository(conn)

    emb_config = EmbeddingConfig(model="fake", dim=512)
    emb = FakeEmbeddingService(emb_config)
    llm = FakeLLMAdapter()
    ids = SequentialIdGenerator(prefix="mem")
    clock = FrozenClock(start_ms=1_700_000_000_000)
    retrieval_config = RetrievalConfig()

    write_svc = MemoryWriteService(
        llm=llm,
        embedding=emb,
        memory_repo=memory_repo,
        event_lookup=event_repo.get,
        ids=ids,
        clock=clock,
        retrieval_config=retrieval_config,
        embedding_model_name="fake",
        embedding_dim=512,
    )
    retrieval_svc = MemoryRetrievalService(
        embedding=emb,
        memory_repo=memory_repo,
        clock=clock,
        config=retrieval_config,
    )

    yield {
        "db": db,
        "event_repo": event_repo,
        "memory_repo": memory_repo,
        "emb": emb,
        "llm": llm,
        "ids": ids,
        "clock": clock,
        "write_svc": write_svc,
        "retrieval_svc": retrieval_svc,
    }
    db.close()


def make_user_event(ids: SequentialIdGenerator, clock: FrozenClock, content: str) -> Event:
    signal = IncomingSignal(
        actor=Actor.USER,
        signal_type="user.message",
        content=content,
        channel="test",
        channel_message_id=f"m-{ids.new()}",
        conversation_id="c1",
    )
    return normalize_signal(
        signal,
        event_id=ids.new(),
        correlation_id="corr-1",
        now_ms=clock.now_ms(),
        source_kind=SourceKind.USER_OBSERVED,
    )


def make_agent_event(ids: SequentialIdGenerator, clock: FrozenClock, content: str) -> Event:
    signal = IncomingSignal(
        actor=Actor.AGENT,
        signal_type="agent.message",
        content=content,
        channel="test",
        channel_message_id=f"m-{ids.new()}",
        conversation_id="c1",
    )
    return normalize_signal(
        signal,
        event_id=ids.new(),
        correlation_id="corr-1",
        now_ms=clock.now_ms(),
        source_kind=SourceKind.AGENT_OUTPUT,
    )


# ---------------------------------------------------------------------------
# Memory write tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_candidates_from_user_event(setup):
    s = setup
    event = make_user_event(s["ids"], s["clock"], "my name is xiaoming, i like hotpot")
    s["event_repo"].append(event)

    s["llm"].set_response(
        "memory_extract",
        '{"candidates": [{"memory_type": "semantic", "content": "用户叫小明", "evidence_event_ids": ["'
        + event.id
        + '"], "confidence": 0.9, "importance": 0.8, "valence": 0.5, "arousal": 0.3}]}',
    )

    candidates = await s["write_svc"].extract_candidates(event)
    assert len(candidates) == 1
    assert candidates[0].content == "用户叫小明"
    assert candidates[0].source_kind == SourceKind.USER_OBSERVED
    request = s["llm"].calls[-1]
    assert request.model == "deepseek/deepseek-v4-flash"
    assert request.thinking == ThinkingMode.DISABLED
    assert request.reasoning_effort is None
    assert [message.role for message in request.messages] == ["system", "user"]
    assert event.content in (request.messages[-1].content or "")
    assert event.id in (request.messages[-1].content or "")


@pytest.mark.asyncio
async def test_extract_candidates_rejects_invalid_evidence(setup):
    s = setup
    user_event = make_user_event(s["ids"], s["clock"], "hello")
    s["event_repo"].append(user_event)
    agent_event = Event(
        id=s["ids"].new(),
        correlation_id="corr-1",
        conversation_id="c1",
        actor=Actor.AGENT,
        event_type="agent.message",
        source_kind=SourceKind.AGENT_OUTPUT,
        content="hi",
        content_hash="x",
        created_at_ms=s["clock"].now_ms(),
    )
    s["event_repo"].append(agent_event)

    s["llm"].set_response(
        "memory_extract",
        '{"candidates": [{"memory_type": "semantic", "content": "inferred fact", "evidence_event_ids": ["nonexistent-id"], "confidence": 0.95, "importance": 0.5, "valence": 0.0, "arousal": 0.3}]}',
    )

    candidates = await s["write_svc"].extract_candidates(user_event, agent_event)
    assert candidates == []


@pytest.mark.asyncio
async def test_extract_candidates_rejects_malformed_evidence_container(setup):
    s = setup
    user_event = make_user_event(s["ids"], s["clock"], "hello")
    s["event_repo"].append(user_event)
    s["llm"].set_response(
        "memory_extract",
        '{"candidates": [{"memory_type": "semantic", "content": "inferred fact", '
        '"evidence_event_ids": null, "confidence": 0.8, "importance": 0.5, '
        '"valence": 0.0, "arousal": 0.3}]}',
    )

    candidates = await s["write_svc"].extract_candidates(user_event)

    assert candidates == []


@pytest.mark.asyncio
async def test_extract_candidates_empty_evidence_uses_turn_as_inference(setup):
    s = setup
    user_event = make_user_event(s["ids"], s["clock"], "I prefer careful evidence")
    s["event_repo"].append(user_event)
    agent_event = Event(
        id=s["ids"].new(),
        correlation_id="corr-1",
        conversation_id="c1",
        actor=Actor.AGENT,
        event_type="agent.message",
        source_kind=SourceKind.AGENT_OUTPUT,
        content="I will remember that preference",
        content_hash="x",
        created_at_ms=s["clock"].now_ms(),
    )
    s["event_repo"].append(agent_event)
    s["llm"].set_response(
        "memory_extract",
        '{"candidates": [{"memory_type": "semantic", "content": "user prefers careful '
        'evidence", "evidence_event_ids": [], "confidence": 0.9, "importance": 0.8, '
        '"valence": 0.2, "arousal": 0.3}]}',
    )

    candidates = await s["write_svc"].extract_candidates(user_event, agent_event)

    assert len(candidates) == 1
    assert candidates[0].evidence_event_ids == [user_event.id, agent_event.id]
    assert candidates[0].source_kind == SourceKind.MODEL_INFERENCE
    assert candidates[0].confidence == 0.6


@pytest.mark.asyncio
async def test_extract_candidates_accepts_deepseek_compact_aliases(setup):
    s = setup
    event = make_user_event(s["ids"], s["clock"], "My name is Xiaoming")
    s["event_repo"].append(event)
    s["llm"].set_response(
        "memory_extract",
        '{"candidates": [{"memory_type": "user_preference", '
        '"fact": "The user is named Xiaoming", "event_ids": ["' + event.id + '"]}]}',
    )

    candidates = await s["write_svc"].extract_candidates(event)

    assert len(candidates) == 1
    assert candidates[0].memory_type == MemoryType.SEMANTIC
    assert candidates[0].content == "The user is named Xiaoming"
    assert candidates[0].evidence_event_ids == [event.id]


@pytest.mark.asyncio
async def test_extract_candidates_mixed_evidence_is_inference_and_capped(setup):
    s = setup
    user_event = make_user_event(s["ids"], s["clock"], "hello")
    s["event_repo"].append(user_event)
    agent_event = Event(
        id=s["ids"].new(),
        correlation_id="corr-1",
        conversation_id="c1",
        actor=Actor.AGENT,
        event_type="agent.message",
        source_kind=SourceKind.AGENT_OUTPUT,
        content="hi",
        content_hash="x",
        created_at_ms=s["clock"].now_ms(),
    )
    s["event_repo"].append(agent_event)
    evidence_json = f'["{user_event.id}", "{agent_event.id}"]'
    s["llm"].set_response(
        "memory_extract",
        '{"candidates": [{"memory_type": "semantic", "content": "inferred fact", '
        f'"evidence_event_ids": {evidence_json}, "confidence": 0.95, '
        '"importance": 0.5, "valence": 0.0, "arousal": 0.3}]}',
    )

    candidates = await s["write_svc"].extract_candidates(user_event, agent_event)
    assert len(candidates) == 1
    assert candidates[0].source_kind == SourceKind.MODEL_INFERENCE
    assert candidates[0].confidence == 0.6
    assert candidates[0].derived_by_model == "deepseek/deepseek-v4-flash"


def test_write_candidates_creates_memory(setup):
    s = setup
    event = make_user_event(s["ids"], s["clock"], "test content")
    s["event_repo"].append(event)

    from ssa.domain.memories import MemoryCandidate

    candidate = MemoryCandidate(
        memory_type=MemoryType.SEMANTIC,
        content="a fact",
        source_kind=SourceKind.USER_OBSERVED,
        evidence_event_ids=[event.id],
        confidence=0.9,
        importance=0.8,
        valence=0.5,
        arousal=0.3,
    )

    result = s["write_svc"].write_candidates([candidate])
    assert len(result.created) == 1
    assert s["memory_repo"].count() == 1

    mem = s["memory_repo"].get(result.created[0])
    assert mem is not None
    assert mem.content == "a fact"
    assert mem.memory_type == MemoryType.SEMANTIC


def test_write_candidates_dedup_by_content_hash(setup):
    s = setup
    event = make_user_event(s["ids"], s["clock"], "test")
    s["event_repo"].append(event)

    from ssa.domain.memories import MemoryCandidate

    candidate = MemoryCandidate(
        memory_type=MemoryType.SEMANTIC,
        content="same content",
        source_kind=SourceKind.USER_OBSERVED,
        evidence_event_ids=[event.id],
        confidence=0.9,
        importance=0.8,
        valence=0.5,
        arousal=0.3,
    )

    # Write twice.
    r1 = s["write_svc"].write_candidates([candidate])
    r2 = s["write_svc"].write_candidates([candidate])

    assert len(r1.created) == 1
    assert len(r2.merged) == 1
    assert s["memory_repo"].count() == 1  # only one memory


def test_write_candidates_adds_evidence(setup):
    s = setup
    event = make_user_event(s["ids"], s["clock"], "test")
    s["event_repo"].append(event)

    from ssa.domain.memories import MemoryCandidate

    candidate = MemoryCandidate(
        memory_type=MemoryType.SEMANTIC,
        content="fact with evidence",
        source_kind=SourceKind.USER_OBSERVED,
        evidence_event_ids=[event.id],
        confidence=0.9,
        importance=0.8,
        valence=0.5,
        arousal=0.3,
    )
    result = s["write_svc"].write_candidates([candidate])

    evidence = s["memory_repo"].get_evidence(result.created[0])
    assert len(evidence) == 1
    assert evidence[0][0] == event.id
    assert evidence[0][1] == "supports"


def test_write_candidates_agent_output_not_user_fact(setup):
    s = setup
    event = make_agent_event(s["ids"], s["clock"], "test")
    s["event_repo"].append(event)

    from ssa.domain.memories import MemoryCandidate

    # Agent output candidate.
    candidate = MemoryCandidate(
        memory_type=MemoryType.SEMANTIC,
        content="agent said something",
        source_kind=SourceKind.AGENT_OUTPUT,
        evidence_event_ids=[event.id],
        confidence=0.9,
        importance=0.5,
        valence=0.0,
        arousal=0.3,
    )
    result = s["write_svc"].write_candidates([candidate])
    mem = s["memory_repo"].get(result.created[0])
    assert mem is not None
    assert mem.source_kind == SourceKind.AGENT_OUTPUT  # not promoted to USER_OBSERVED


def test_write_candidates_rejects_unknown_evidence_without_partial_memory(setup):
    from ssa.domain.memories import MemoryCandidate

    candidate = MemoryCandidate(
        memory_type=MemoryType.REFLECTION,
        content="A reflection with fabricated provenance must not persist.",
        source_kind=SourceKind.MODEL_INFERENCE,
        evidence_event_ids=["missing-event"],
        confidence=0.5,
        importance=0.5,
        valence=0.0,
        arousal=0.2,
    )

    with pytest.raises(ValueError, match="unknown evidence"):
        setup["write_svc"].write_candidates([candidate])
    assert setup["memory_repo"].count() == 0


def test_direct_model_inference_candidate_is_capped_at_point_of_write(setup):
    event = make_user_event(setup["ids"], setup["clock"], "source observation")
    setup["event_repo"].append(event)
    from ssa.domain.memories import MemoryCandidate

    candidate = MemoryCandidate(
        memory_type=MemoryType.REFLECTION,
        content="A model-authored interpretation of the observation.",
        source_kind=SourceKind.MODEL_INFERENCE,
        evidence_event_ids=[event.id],
        confidence=0.95,
        importance=0.5,
        valence=0.0,
        arousal=0.2,
    )

    result = setup["write_svc"].write_candidates([candidate])
    memory = setup["memory_repo"].get(result.created[0])
    assert memory is not None
    assert memory.confidence == 0.6


# ---------------------------------------------------------------------------
# Memory retrieval tests
# ---------------------------------------------------------------------------


def test_retrieve_returns_empty_when_no_memories(setup):
    s = setup
    results = s["retrieval_svc"].retrieve("hello")
    assert results == []


def test_retrieve_returns_matching_memory(setup):
    s = setup
    event = make_user_event(s["ids"], s["clock"], "user likes pizza")
    s["event_repo"].append(event)

    from ssa.domain.memories import MemoryCandidate

    candidate = MemoryCandidate(
        memory_type=MemoryType.SEMANTIC,
        content="user likes pizza",
        source_kind=SourceKind.USER_OBSERVED,
        evidence_event_ids=[event.id],
        confidence=0.9,
        importance=0.8,
        valence=0.7,
        arousal=0.3,
    )
    s["write_svc"].write_candidates([candidate])

    # Retrieve with a similar query.
    results = s["retrieval_svc"].retrieve("what food does user like")
    assert len(results) >= 1
    assert "pizza" in results[0].content


def test_retrieve_includes_score_components(setup):
    s = setup
    event = make_user_event(s["ids"], s["clock"], "test memory")
    s["event_repo"].append(event)

    from ssa.domain.memories import MemoryCandidate

    candidate = MemoryCandidate(
        memory_type=MemoryType.SEMANTIC,
        content="test memory",
        source_kind=SourceKind.USER_OBSERVED,
        evidence_event_ids=[event.id],
        confidence=0.9,
        importance=0.8,
        valence=0.5,
        arousal=0.3,
    )
    s["write_svc"].write_candidates([candidate])

    results = s["retrieval_svc"].retrieve("test")
    assert len(results) >= 1
    assert "semantic" in results[0].score_components
    assert "recency" in results[0].score_components


def test_prefetch_does_not_update_access(setup):
    s = setup
    event = make_user_event(s["ids"], s["clock"], "prefetch test")
    s["event_repo"].append(event)

    from ssa.domain.memories import MemoryCandidate

    candidate = MemoryCandidate(
        memory_type=MemoryType.SEMANTIC,
        content="prefetch test",
        source_kind=SourceKind.USER_OBSERVED,
        evidence_event_ids=[event.id],
        confidence=0.9,
        importance=0.8,
        valence=0.5,
        arousal=0.3,
    )
    s["write_svc"].write_candidates([candidate])

    # Prefetch should not update access count.
    results = s["retrieval_svc"].prefetch_for_appraisal("prefetch")
    if results:
        mem = s["memory_repo"].get(results[0].memory_id)
        assert mem is not None
        assert mem.access_count == 0  # not updated by prefetch


def test_retrieve_updates_access(setup):
    s = setup
    event = make_user_event(s["ids"], s["clock"], "access test")
    s["event_repo"].append(event)

    from ssa.domain.memories import MemoryCandidate

    candidate = MemoryCandidate(
        memory_type=MemoryType.SEMANTIC,
        content="access test",
        source_kind=SourceKind.USER_OBSERVED,
        evidence_event_ids=[event.id],
        confidence=0.9,
        importance=0.8,
        valence=0.5,
        arousal=0.3,
    )
    s["write_svc"].write_candidates([candidate])

    s["retrieval_svc"].retrieve("access")
    mem_id = s["memory_repo"].all_active()[0].id
    mem = s["memory_repo"].get(mem_id)
    assert mem is not None
    assert mem.access_count >= 1


def test_retrieve_respects_max_results(setup):
    s = setup
    # Create multiple memories.
    for i in range(10):
        event = make_user_event(s["ids"], s["clock"], f"memory item {i}")
        s["event_repo"].append(event)
        from ssa.domain.memories import MemoryCandidate

        candidate = MemoryCandidate(
            memory_type=MemoryType.SEMANTIC,
            content=f"memory item {i}",
            source_kind=SourceKind.USER_OBSERVED,
            evidence_event_ids=[event.id],
            confidence=0.9,
            importance=0.5,
            valence=0.0,
            arousal=0.3,
        )
        s["write_svc"].write_candidates([candidate])
        s["clock"].advance_ms(1000)

    results = s["retrieval_svc"].retrieve("memory", max_results=3)
    assert len(results) <= 3
