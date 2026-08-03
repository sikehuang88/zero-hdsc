"""Focused tests for the M09 relationship evolution module."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from ssa.clock import FrozenClock
from ssa.config import DatabaseConfig, RelationshipConfig
from ssa.domain.appraisal import AppraisalResult
from ssa.domain.enums import ActionIntent, Actor, SourceKind
from ssa.domain.events import Event
from ssa.domain.relationship import (
    RelationshipAction,
    RelationshipEventKind,
    RelationshipSignal,
    RelationshipState,
)
from ssa.ids import SequentialIdGenerator
from ssa.services.relationship_service import RelationshipService
from ssa.storage.database import Database
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.relationship_repository import (
    RelationshipRepository,
    RelationshipVersionConflict,
)


def make_event(
    event_id: str,
    event_type: str,
    *,
    actor: Actor = Actor.USER,
    metadata: dict[str, object] | None = None,
) -> Event:
    return Event(
        id=event_id,
        correlation_id=f"corr-{event_id}",
        conversation_id="conversation-1",
        actor=actor,
        event_type=event_type,
        source_kind=(SourceKind.USER_OBSERVED if actor == Actor.USER else SourceKind.AGENT_OUTPUT),
        content=event_type,
        content_hash=f"hash-{event_id}",
        metadata=metadata or {},
        created_at_ms=1_700_000_000_000,
    )


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(1_700_000_001_000)


@pytest.fixture
def service(clock: FrozenClock) -> RelationshipService:
    return RelationshipService(RelationshipConfig(), clock)


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    db = Database(DatabaseConfig(path=str(tmp_path / "relationship.db")))
    db.initialize()
    yield db
    db.close()


def test_initial_state_is_bounded_and_validates_identifiers() -> None:
    state = RelationshipState.initial(1000)
    assert state.version == 1
    assert state.repair_debt == 0.0
    assert all(
        0.0 <= value <= 1.0
        for value in (
            state.trust,
            state.closeness,
            state.tension,
            state.reciprocity,
            state.repair_debt,
        )
    )

    with pytest.raises(ValueError, match="duplicates"):
        RelationshipState(
            **state.model_dump(exclude={"active_commitment_ids"}),
            active_commitment_ids=["promise-1", "promise-1"],
        )


def test_event_mapping_uses_structured_type_and_metadata(
    service: RelationshipService,
) -> None:
    event = make_event(
        "evt-1",
        "relationship.promise_kept",
        metadata={"promise_memory_id": "promise-1", "major": True},
    )

    signal = service.signal_from_event(event)

    assert signal.kind == RelationshipEventKind.USER_PROMISE_KEPT
    assert signal.commitment_ids == ["promise-1"]
    assert signal.major is True


def test_preview_is_transient_deterministic_and_evidence_bearing(
    service: RelationshipService,
    clock: FrozenClock,
) -> None:
    old = RelationshipState.initial(1000).model_copy(
        update={"active_commitment_ids": ["promise-1"]}
    )
    signal = RelationshipSignal(
        event_id="evt-kept",
        kind=RelationshipEventKind.USER_PROMISE_KEPT,
        commitment_ids=["promise-1"],
    )

    first = service.preview(old, signal)
    second = service.preview(old, signal)

    assert first.model_dump() == second.model_dump()
    assert first.version == old.version
    assert first.updated_at_ms == clock.now_ms()
    assert first.trust > old.trust
    assert first.active_commitment_ids == []
    assert old.active_commitment_ids == ["promise-1"]
    assert first.trust - old.trust <= 0.03


def test_neutral_fallback_does_not_raise_trust_or_closeness(
    service: RelationshipService,
) -> None:
    old = RelationshipState.initial(1000)
    event = make_event("evt-chat", "user.message")

    preview = service.preview(old, event, AppraisalResult.neutral())

    assert preview.trust == old.trust
    assert preview.closeness == old.closeness


@pytest.mark.parametrize(
    ("major", "cap"),
    [(False, 0.03), (True, 0.10)],
)
def test_complete_preview_finalize_transition_respects_delta_cap(
    service: RelationshipService,
    major: bool,
    cap: float,
) -> None:
    old = RelationshipState.initial(1000)
    signal = RelationshipSignal(
        event_id="evt-project",
        kind=RelationshipEventKind.SHARED_PROJECT_PROGRESS,
        major=major,
    )
    preview = service.preview(old, signal)
    final = service.finalize(preview, ActionIntent.PROJECT_WORK)

    assert final.version == old.version + 1
    for field in ("trust", "closeness", "tension", "reciprocity", "repair_debt"):
        assert abs(getattr(final, field) - getattr(old, field)) <= cap


def test_conflict_and_repair_track_unresolved_memory(
    service: RelationshipService,
) -> None:
    old = RelationshipState.initial(1000)
    conflict = RelationshipSignal(
        event_id="evt-conflict",
        kind=RelationshipEventKind.CONFLICT,
        unresolved_memory_ids=["unresolved-1"],
    )
    preview = service.preview(old, conflict)
    action = RelationshipAction(
        event_id="evt-agent-repair",
        intent=ActionIntent.REPAIR,
        resolved_memory_ids=["unresolved-1"],
    )

    final = service.finalize(preview, action)

    assert preview.tension > old.tension
    assert preview.repair_debt > old.repair_debt
    assert preview.unresolved_memory_ids == ["unresolved-1"]
    assert final.tension < preview.tension
    assert final.repair_debt < preview.repair_debt
    assert final.unresolved_memory_ids == []


def test_finalize_requires_evidence_for_a_bare_state(
    service: RelationshipService,
) -> None:
    with pytest.raises(ValueError, match="requires a preview"):
        service.finalize(RelationshipState.initial(1000), ActionIntent.ANSWER)


def test_long_term_trust_growth_requires_multiple_events(
    service: RelationshipService,
) -> None:
    state = RelationshipState.initial(1000)
    initial_trust = state.trust
    for number in range(4):
        preview = service.preview(
            state,
            RelationshipSignal(
                event_id=f"evt-kept-{number}",
                kind=RelationshipEventKind.USER_PROMISE_KEPT,
            ),
        )
        state = service.finalize(preview, ActionIntent.ACKNOWLEDGE)

    assert initial_trust < state.trust < initial_trust + 0.10


def test_repository_round_trip_links_versions_atomically(database: Database) -> None:
    event_repo = SqliteEventRepository(database.connection)
    initial_event = make_event("evt-initial", "system.start", actor=Actor.SYSTEM)
    update_event = make_event("evt-update", "relationship.project_progress")
    event_repo.append(initial_event)
    event_repo.append(update_event)

    ids = SequentialIdGenerator("relationship")
    repo = RelationshipRepository(database.connection, ids)
    first = RelationshipState.initial(1000)
    repo.append_if_version(0, first, initial_event.id)
    second = first.model_copy(update={"version": 2, "updated_at_ms": 2000})
    repo.append_if_version(1, second, update_event.id)

    assert repo.latest() == second
    assert repo.get_by_version(1) == first
    rows = database.connection.execute(
        "SELECT id, previous_id, version FROM relationship_snapshots ORDER BY version"
    ).fetchall()
    assert rows[0]["previous_id"] is None
    assert rows[1]["previous_id"] == rows[0]["id"]


def test_repository_rejects_stale_writer_without_partial_insert(
    database: Database,
) -> None:
    event_repo = SqliteEventRepository(database.connection)
    events = [make_event(f"evt-{number}", "user.message") for number in range(3)]
    for event in events:
        event_repo.append(event)

    repo = RelationshipRepository(database.connection, SequentialIdGenerator("relationship"))
    first = RelationshipState.initial(1000)
    repo.append_if_version(0, first, events[0].id)
    second = first.model_copy(update={"version": 2, "updated_at_ms": 2000})
    repo.append_if_version(1, second, events[1].id)

    with pytest.raises(RelationshipVersionConflict, match="current is 2"):
        repo.append_if_version(1, second, events[2].id)

    count = database.connection.execute(
        "SELECT COUNT(*) AS count FROM relationship_snapshots"
    ).fetchone()
    assert count["count"] == 2


def test_repository_rejects_nonsequential_version(database: Database) -> None:
    event = make_event("evt-1", "user.message")
    SqliteEventRepository(database.connection).append(event)
    repo = RelationshipRepository(database.connection, SequentialIdGenerator("relationship"))
    invalid = RelationshipState.initial(1000).model_copy(update={"version": 2})

    with pytest.raises(ValueError, match="must be 1"):
        repo.append_if_version(0, invalid, event.id)


def test_repository_returns_none_when_empty(database: Database) -> None:
    repo = RelationshipRepository(database.connection, SequentialIdGenerator("relationship"))
    assert repo.latest() is None
    assert repo.get_by_version(1) is None
