"""Persistence tests for Phase 3 appraisal and state snapshots."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssa.adapters.llm import FakeLLMAdapter
from ssa.clock import FrozenClock
from ssa.config import DatabaseConfig
from ssa.domain.appraisal import AppraisalResult, StoredAppraisal
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event
from ssa.domain.state import OrganismState
from ssa.ids import SequentialIdGenerator
from ssa.services.appraisal_service import AppraisalService
from ssa.storage.appraisal_repository import SqliteAppraisalRepository
from ssa.storage.database import Database
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.state_repository import StateRepository, StateVersionConflict


@pytest.fixture
def persistence(tmp_path: Path):
    db = Database(DatabaseConfig(path=str(tmp_path / "phase3.db")))
    db.initialize()
    ids = SequentialIdGenerator(prefix="p3")
    clock = FrozenClock(1_700_000_000_000)
    event_repo = SqliteEventRepository(db.connection)
    appraisal_repo = SqliteAppraisalRepository(db.connection)
    event = Event(
        id=ids.new(),
        correlation_id="corr-phase3",
        conversation_id="conversation-1",
        actor=Actor.USER,
        event_type="user.message",
        source_kind=SourceKind.USER_OBSERVED,
        content="我今天升职了",
        content_hash="hash",
        created_at_ms=clock.now_ms(),
    )
    event_repo.append(event)
    yield db, ids, clock, appraisal_repo, event
    db.close()


def _positive_appraisal() -> AppraisalResult:
    return AppraisalResult(
        novelty=0.8,
        goal_congruence=0.7,
        controllability=0.7,
        certainty=0.9,
        self_agency=0.2,
        user_agency=0.8,
        external_agency=0.1,
        relationship_relevance=0.7,
        urgency=0.3,
        valence_signal=0.8,
        arousal_signal=0.6,
        supported_event_ids=["p3-000001"],
        explanation="positive career event",
    )


def test_appraisal_repository_round_trip(persistence):
    _, ids, clock, repo, event = persistence
    record = StoredAppraisal(
        id=ids.new(),
        correlation_id=event.correlation_id,
        cause_event_id=event.id,
        result=_positive_appraisal(),
        provider="fake",
        model="fake-model",
        prompt_version="appraisal_v1",
        created_at_ms=clock.now_ms(),
    )
    repo.insert(record)

    loaded = repo.get(record.id)
    assert loaded == record
    assert repo.find_by_correlation(event.correlation_id) == [record]


@pytest.mark.asyncio
async def test_appraisal_service_persists_success(persistence):
    _, ids, clock, repo, event = persistence
    llm = FakeLLMAdapter()
    llm.set_response("appraisal", _positive_appraisal().model_dump_json())
    service = AppraisalService(llm, repository=repo, ids=ids, clock=clock)

    result = await service.evaluate(event, "stable", [], [])

    records = repo.find_by_correlation(event.correlation_id)
    assert result.valence_signal == 0.8
    assert len(records) == 1
    assert records[0].result == result
    assert records[0].provider == "fake"


@pytest.mark.asyncio
async def test_appraisal_service_persists_neutral_fallback(persistence):
    _, ids, clock, repo, event = persistence
    llm = FakeLLMAdapter()
    llm.set_response("appraisal", "not-json")
    service = AppraisalService(llm, repository=repo, ids=ids, clock=clock)

    result = await service.evaluate(event, "stable", [], [])

    records = repo.find_by_correlation(event.correlation_id)
    assert result == AppraisalResult.neutral()
    assert len(records) == 1
    assert records[0].provider == "fallback"


@pytest.mark.asyncio
async def test_appraisal_service_marks_validation_fallback(persistence):
    _, ids, clock, repo, event = persistence
    llm = FakeLLMAdapter()
    llm.set_response(
        "appraisal",
        '{"novelty": 4, "goal_congruence": 0, "controllability": 0.5, '
        '"certainty": 0.9, "self_agency": 0.2, "user_agency": 0.8, '
        '"external_agency": 0.1, "relationship_relevance": 0.7, '
        '"urgency": 0.3, "valence_signal": 0.8, "arousal_signal": 0.6, '
        '"supported_event_ids": ["p3-000001"], "explanation": "invalid"}',
    )
    service = AppraisalService(llm, repository=repo, ids=ids, clock=clock)

    result = await service.evaluate(event, "stable", [], [])

    records = repo.find_by_correlation(event.correlation_id)
    assert result == AppraisalResult.neutral()
    assert records[0].provider == "fallback"


def test_appraisal_service_rejects_partial_persistence_dependencies(persistence):
    _, _, _, repo, _ = persistence
    with pytest.raises(ValueError, match="provided together"):
        AppraisalService(FakeLLMAdapter(), repository=repo)


def test_state_repository_links_versions_and_round_trips(persistence):
    db, ids, clock, _, event = persistence
    repo = StateRepository(db.connection, ids)
    first = OrganismState.initial(clock.now_ms())
    repo.append_if_version(0, first, event.id)

    clock.advance_ms(1000)
    second = first.model_copy(update={"version": 2, "updated_at_ms": clock.now_ms()})
    repo.append_if_version(1, second, event.id)

    assert repo.latest() == second
    assert repo.get_by_version(1) == first
    rows = db.connection.execute(
        "SELECT id, previous_id, version FROM state_snapshots ORDER BY version"
    ).fetchall()
    assert rows[0]["previous_id"] is None
    assert rows[1]["previous_id"] == rows[0]["id"]


def test_state_repository_rejects_stale_version(persistence):
    db, ids, clock, _, event = persistence
    repo = StateRepository(db.connection, ids)
    first = OrganismState.initial(clock.now_ms())
    repo.append_if_version(0, first, event.id)
    stale = first.model_copy(update={"version": 1})

    with pytest.raises(StateVersionConflict, match="current is 1"):
        repo.append_if_version(0, stale, event.id)


def test_state_repository_rejects_nonsequential_state_version(persistence):
    db, ids, clock, _, event = persistence
    repo = StateRepository(db.connection, ids)
    invalid = OrganismState.initial(clock.now_ms()).model_copy(update={"version": 3})

    with pytest.raises(ValueError, match="must be 1"):
        repo.append_if_version(0, invalid, event.id)


def test_state_repository_empty_latest(persistence):
    db, ids, _, _, _ = persistence
    repo = StateRepository(db.connection, ids)
    assert repo.latest() is None
    assert repo.get_by_version(1) is None
