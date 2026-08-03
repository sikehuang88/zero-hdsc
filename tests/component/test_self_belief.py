"""Component coverage for Phase 4 M10 self-belief identity lifecycle."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import pytest

from ssa.adapters.llm import FakeLLMAdapter
from ssa.clock import FrozenClock
from ssa.config import DatabaseConfig, IdentityConfig, ReasoningEffort, ThinkingMode
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, compute_content_hash
from ssa.domain.self_belief import (
    SelfBelief,
    SelfBeliefCandidate,
    SelfBeliefEvidenceRelation,
    SelfBeliefStatus,
)
from ssa.ids import SequentialIdGenerator
from ssa.services.self_belief_service import (
    SelfBeliefEvidenceError,
    SelfBeliefLifecycleError,
    SelfBeliefService,
)
from ssa.storage.database import Database
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.self_belief_repository import (
    SelfBeliefVersionConflict,
    SqliteSelfBeliefRepository,
)

DAY_MS = 24 * 60 * 60 * 1000


@dataclass
class IdentitySetup:
    db: Database
    events: SqliteEventRepository
    beliefs: SqliteSelfBeliefRepository
    llm: FakeLLMAdapter
    ids: SequentialIdGenerator
    clock: FrozenClock
    service: SelfBeliefService


@pytest.fixture
def identity_setup(tmp_path: Path):
    db = Database(DatabaseConfig(path=str(tmp_path / "identity.db")))
    db.initialize()
    ids = SequentialIdGenerator(prefix="identity")
    clock = FrozenClock(1_800_000_000_000)
    event_repo = SqliteEventRepository(db.connection)
    belief_repo = SqliteSelfBeliefRepository(db.connection)
    llm = FakeLLMAdapter()
    service = SelfBeliefService(
        llm,
        belief_repo,
        event_repo.get,
        ids,
        clock,
        IdentityConfig(),
    )
    yield IdentitySetup(db, event_repo, belief_repo, llm, ids, clock, service)
    db.close()


def _append_event(
    setup: IdentitySetup,
    content: str,
    *,
    created_at_ms: int | None = None,
    actor: Actor = Actor.AGENT,
    event_type: str = "agent.action",
) -> Event:
    event = Event(
        id=setup.ids.new(),
        correlation_id="identity-review",
        conversation_id="identity",
        actor=actor,
        event_type=event_type,
        source_kind=(
            SourceKind.AGENT_OUTPUT if actor == Actor.AGENT else SourceKind.SYSTEM_DERIVED
        ),
        content=content,
        content_hash=compute_content_hash(content),
        created_at_ms=(setup.clock.now_ms() if created_at_ms is None else created_at_ms),
    )
    setup.events.append(event)
    return event


def _candidate(event_ids: list[str], confidence: float = 0.6) -> SelfBeliefCandidate:
    return SelfBeliefCandidate(
        claim="I tend to investigate uncertainty before answering",
        confidence=confidence,
        evidence_event_ids=event_ids,
        change_reason="repeated investigation choices",
        model="fake-model",
        prompt_version="identity_review_v1",
    )


def test_migration_adds_version_chain_and_evidence_schema(identity_setup: IdentitySetup):
    setup = identity_setup
    assert setup.db.schema_version == 17
    columns = {
        row["name"] for row in setup.db.connection.execute("PRAGMA table_info(self_beliefs)")
    }
    assert {
        "lineage_id",
        "previous_id",
        "cause_event_id",
        "change_reason",
        "model",
        "prompt_version",
        "candidate_since_ms",
        "activated_at_ms",
    } <= columns
    evidence_table = setup.db.connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'self_belief_evidence'"
    ).fetchone()
    assert evidence_table is not None


def test_migration_007_upgrades_legacy_belief_and_backfills_evidence(tmp_path: Path):
    db_path = tmp_path / "legacy-identity.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        );
        CREATE TABLE events (id TEXT PRIMARY KEY);
        CREATE TABLE self_beliefs (
            id TEXT PRIMARY KEY,
            claim TEXT NOT NULL,
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            status TEXT NOT NULL DEFAULT 'candidate',
            version INTEGER NOT NULL,
            evidence_json TEXT NOT NULL DEFAULT '[]',
            counterevidence_json TEXT NOT NULL DEFAULT '[]',
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
        );
        INSERT INTO events (id) VALUES ('support-event'), ('counter-event');
        INSERT INTO self_beliefs VALUES (
            'legacy-belief', 'I inspect evidence', 0.75, 'active', 3,
            '["support-event"]', '["counter-event"]', 1000, 2000
        );
        """
    )
    conn.executemany(
        "INSERT INTO schema_migrations VALUES (?, ?, ?)",
        [(version, f"legacy-{version}", "2026-01-01") for version in range(1, 7)],
    )
    conn.commit()
    conn.close()

    db = Database(DatabaseConfig(path=str(db_path)))
    db.initialize()
    repo = SqliteSelfBeliefRepository(db.connection)
    upgraded = repo.get("legacy-belief")

    assert upgraded is not None
    assert upgraded.lineage_id == upgraded.id
    assert upgraded.version == 1
    assert upgraded.activated_at_ms == upgraded.created_at_ms
    links = repo.evidence_for_version(upgraded.id)
    assert {(link.event_id, link.relation) for link in links} == {
        ("support-event", SelfBeliefEvidenceRelation.SUPPORTS),
        ("counter-event", SelfBeliefEvidenceRelation.CONTRADICTS),
    }
    db.close()


@pytest.mark.asyncio
async def test_extraction_requires_two_real_events_and_caps_candidate(
    identity_setup: IdentitySetup,
):
    setup = identity_setup
    first = _append_event(
        setup,
        "I checked two sources before answering",
        created_at_ms=setup.clock.now_ms() - DAY_MS,
    )
    second = _append_event(setup, "I asked for missing evidence")
    setup.llm.set_response(
        "identity_review",
        json.dumps(
            {
                "candidates": [
                    {
                        "claim": "I tend to investigate uncertainty before answering",
                        "confidence": 0.95,
                        "evidence_event_ids": [first.id, second.id],
                        "change_reason": "the behavior repeated",
                    },
                    {
                        "claim": "A single response defines me",
                        "confidence": 0.9,
                        "evidence_event_ids": [first.id],
                        "change_reason": "one sample",
                    },
                    {
                        "claim": "Forged evidence should be ignored",
                        "confidence": 0.6,
                        "evidence_event_ids": [first.id, second.id, "missing-event"],
                        "change_reason": "contains an unknown event",
                    },
                ]
            }
        ),
    )

    candidates = await setup.service.extract_candidates([first, second])
    assert len(candidates) == 1
    assert candidates[0].confidence == 0.6
    assert candidates[0].evidence_event_ids == [first.id, second.id]
    request = setup.llm.calls[-1]
    assert request.model == "deepseek/deepseek-v4-pro"
    assert request.thinking == ThinkingMode.ENABLED
    assert request.reasoning_effort == ReasoningEffort.HIGH
    assert request.temperature is None
    assert request.max_tokens == 2048
    assert "json" in "\n".join(message.content or "" for message in request.messages).casefold()
    assert [message.role for message in request.messages] == ["system", "user"]
    assert first.id in (request.messages[-1].content or "")
    assert second.id in (request.messages[-1].content or "")

    belief = setup.service.create_candidate(candidates[0])
    assert belief.status == SelfBeliefStatus.CANDIDATE
    assert belief.version == 1
    assert setup.beliefs.get(belief.id) == belief
    links = setup.beliefs.evidence_for_version(belief.id)
    assert {link.event_id for link in links} == {first.id, second.id}
    assert {link.relation for link in links} == {SelfBeliefEvidenceRelation.SUPPORTS}


def test_candidate_active_challenged_revised_archived_lifecycle(
    identity_setup: IdentitySetup,
):
    setup = identity_setup
    first = _append_event(
        setup,
        "I verified a disputed detail",
        created_at_ms=setup.clock.now_ms() - DAY_MS,
    )
    second = _append_event(setup, "I disclosed that my evidence was incomplete")
    candidate = setup.service.create_candidate(_candidate([first.id, second.id]))

    setup.clock.advance_ms(6 * DAY_MS)
    third = _append_event(setup, "I sought another source before committing")
    supported = setup.service.record_support(candidate.lineage_id, third.id)
    assert supported.confidence == pytest.approx(0.7)
    assert supported.status == SelfBeliefStatus.CANDIDATE

    setup.clock.advance_ms(DAY_MS)
    review = _append_event(
        setup,
        "weekly identity review",
        actor=Actor.SYSTEM,
        event_type="identity.review",
    )
    active = setup.service.review_candidate(candidate.lineage_id, review.id)
    assert active.status == SelfBeliefStatus.ACTIVE
    assert active.activated_at_ms == setup.clock.now_ms()

    counter = _append_event(setup, "I answered before checking an uncertain claim")
    challenged = setup.service.record_counterevidence(
        candidate.lineage_id,
        counter.id,
        major=True,
        change_reason="a material counterexample",
    )
    assert challenged.status == SelfBeliefStatus.CHALLENGED
    assert challenged.counterevidence_event_ids == [counter.id]
    assert challenged.confidence == pytest.approx(0.5)

    revised = setup.service.revise(
        candidate.lineage_id,
        "I usually investigate uncertainty, but sometimes answer too quickly",
        counter.id,
        change_reason="qualified the claim after a counterexample",
    )
    assert revised.status == SelfBeliefStatus.REVISED
    assert revised.previous_id == challenged.id
    assert revised.counterevidence_event_ids == [counter.id]
    assert setup.service.relevant_for_prompt([candidate.lineage_id]) == [revised]
    context = setup.service.render_prompt_context([revised])
    assert revised.claim in context
    assert "confidence=0.50" in context

    history = setup.beliefs.history(candidate.lineage_id)
    assert [item.version for item in history] == [1, 2, 3, 4, 5]
    assert all(current.previous_id == previous.id for previous, current in pairwise(history))
    revised_links = setup.beliefs.evidence_for_version(revised.id)
    assert any(
        link.event_id == counter.id and link.relation == SelfBeliefEvidenceRelation.CONTRADICTS
        for link in revised_links
    )

    archived = setup.service.archive(
        candidate.lineage_id,
        review.id,
        change_reason="superseded by a different identity dimension",
    )
    assert archived.status == SelfBeliefStatus.ARCHIVED
    with pytest.raises(SelfBeliefLifecycleError, match="terminal"):
        setup.service.record_support(candidate.lineage_id, third.id)


def test_integrate_candidates_appends_unseen_support(identity_setup: IdentitySetup):
    setup = identity_setup
    first = _append_event(setup, "I checked the source before answering")
    second = _append_event(setup, "I stated where the evidence was uncertain")
    initial = setup.service.create_candidate(_candidate([first.id, second.id]))
    third = _append_event(setup, "I compared two explanations before deciding")

    changed = setup.service.integrate_candidates([_candidate([first.id, third.id])])

    assert len(changed) == 1
    current = changed[0]
    assert current.lineage_id == initial.lineage_id
    assert current.version == 2
    assert current.confidence == pytest.approx(0.65)
    assert current.evidence_event_ids == [first.id, second.id, third.id]


@pytest.mark.asyncio
async def test_extract_candidates_accepts_deepseek_cited_event_alias(
    identity_setup: IdentitySetup,
):
    setup = identity_setup
    first = _append_event(setup, "I checked one uncertain detail")
    second = _append_event(setup, "I checked another uncertain detail")
    setup.llm.set_response(
        "identity_review",
        '{"candidates": [{"claim": "I verify uncertainty before answering", '
        '"confidence": 0.55, "cited_event_ids": ["' + first.id + '", "' + second.id + '"]}]}',
    )

    candidates = await setup.service.extract_candidates([first, second])

    assert len(candidates) == 1
    assert candidates[0].evidence_event_ids == [first.id, second.id]
    assert candidates[0].change_reason == "weekly identity review"


def test_candidate_does_not_activate_before_age_gate(identity_setup: IdentitySetup):
    setup = identity_setup
    first = _append_event(setup, "first action")
    second = _append_event(setup, "second action")
    candidate = setup.service.create_candidate(_candidate([first.id, second.id]))
    review = _append_event(
        setup,
        "early review",
        actor=Actor.SYSTEM,
        event_type="identity.review",
    )

    same = setup.service.review_candidate(candidate.lineage_id, review.id)
    assert same == candidate
    assert len(setup.beliefs.history(candidate.lineage_id)) == 1


def test_candidate_rejects_missing_or_old_evidence(identity_setup: IdentitySetup):
    setup = identity_setup
    recent = _append_event(setup, "recent action")
    old = _append_event(
        setup,
        "old action",
        created_at_ms=setup.clock.now_ms() - (8 * DAY_MS),
    )
    with pytest.raises(SelfBeliefEvidenceError, match="outside the recent week"):
        setup.service.create_candidate(_candidate([recent.id, old.id]))


def test_repository_rejects_stale_writer(identity_setup: IdentitySetup):
    setup = identity_setup
    first = _append_event(setup, "first action")
    second = _append_event(setup, "second action")
    initial = setup.service.create_candidate(_candidate([first.id, second.id]))
    now_ms = setup.clock.now_ms()

    def successor(version_id: str, reason: str) -> SelfBelief:
        return initial.model_copy(
            update={
                "id": version_id,
                "version": 2,
                "previous_id": initial.id,
                "cause_event_id": second.id,
                "change_reason": reason,
                "model": "deterministic",
                "prompt_version": "self_belief_rules_v1",
                "created_at_ms": now_ms,
                "updated_at_ms": now_ms,
            }
        )

    setup.beliefs.append_if_version(1, successor(setup.ids.new(), "writer one"))
    with pytest.raises(SelfBeliefVersionConflict, match="current is 2"):
        setup.beliefs.append_if_version(1, successor(setup.ids.new(), "stale writer"))
    assert len(setup.beliefs.history(initial.lineage_id)) == 2


def test_evidence_foreign_key_failure_rolls_back_version(identity_setup: IdentitySetup):
    setup = identity_setup
    existing = _append_event(setup, "existing action")
    belief_id = setup.ids.new()
    belief = SelfBelief(
        id=belief_id,
        lineage_id=belief_id,
        claim="I verify evidence",
        confidence=0.5,
        status=SelfBeliefStatus.CANDIDATE,
        version=1,
        evidence_event_ids=[existing.id, "missing-event"],
        cause_event_id=existing.id,
        change_reason="test invalid provenance",
        model="fake-model",
        prompt_version="identity_review_v1",
        candidate_since_ms=setup.clock.now_ms(),
        created_at_ms=setup.clock.now_ms(),
        updated_at_ms=setup.clock.now_ms(),
    )

    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        setup.beliefs.insert_initial(belief)
    assert setup.beliefs.get(belief_id) is None


def test_empty_self_beliefs_leave_prompt_context_empty(identity_setup: IdentitySetup):
    assert identity_setup.service.relevant_for_prompt() == []
    assert identity_setup.service.render_prompt_context([]) == ""
