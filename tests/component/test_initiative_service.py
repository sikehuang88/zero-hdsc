"""Component coverage for deterministic initiative gates and outbox creation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ssa.adapters.llm import FakeLLMAdapter
from ssa.clock import FrozenClock
from ssa.config import DatabaseConfig, InitiativeConfig, Settings
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import IncomingSignal, normalize_signal
from ssa.domain.lifecycle import EmotionEpisode, EmotionType, Initiative, InitiativeStatus
from ssa.domain.relationship import RelationshipState
from ssa.domain.state import OrganismState
from ssa.ids import SequentialIdGenerator
from ssa.services.initiative_service import InitiativeService
from ssa.storage.database import Database
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.lifecycle_repository import (
    BackgroundUsageRepository,
    CausalTraceRepository,
    EmotionEpisodeRepository,
    GoalRepository,
    InitiativeRepository,
    OutboxRepository,
)
from ssa.storage.trace_repository import SqliteTraceRepository

_NOW_MS = int(datetime(2030, 3, 5, 15, 0, tzinfo=UTC).timestamp() * 1000)


def _setup(
    tmp_path: Path,
    *,
    initiative_config: InitiativeConfig | None = None,
    llm: FakeLLMAdapter | None = None,
) -> tuple[
    Database,
    FrozenClock,
    SequentialIdGenerator,
    InitiativeService,
    SqliteEventRepository,
    InitiativeRepository,
    OutboxRepository,
    BackgroundUsageRepository,
]:
    settings = Settings(
        database=DatabaseConfig(path=str(tmp_path / "initiative.db")),
        initiative=initiative_config
        or InitiativeConfig(quiet_hours_start="23:00", quiet_hours_end="07:00"),
    )
    database = Database(settings.database)
    database.initialize()
    clock = FrozenClock(_NOW_MS)
    ids = SequentialIdGenerator("initiative")
    events = SqliteEventRepository(database.connection)
    initiatives = InitiativeRepository(database.connection)
    outbox = OutboxRepository(database.connection)
    usage = BackgroundUsageRepository(database.connection)
    service = InitiativeService(
        initiatives=initiatives,
        outbox=outbox,
        emotions=EmotionEpisodeRepository(database.connection),
        goals=GoalRepository(database.connection),
        traces=SqliteTraceRepository(database.connection),
        events=events,
        causal_traces=CausalTraceRepository(database.connection),
        usage=usage,
        clock=clock,
        ids=ids,
        initiative_config=settings.initiative,
        budget_config=settings.budget,
        goal_config=settings.goal,
        llm_config=settings.llm,
        timezone="UTC",
        llm=llm,
    )
    return database, clock, ids, service, events, initiatives, outbox, usage


def _seed_evidence(
    database: Database,
    clock: FrozenClock,
    ids: SequentialIdGenerator,
    events: SqliteEventRepository,
) -> str:
    event = events.append(
        normalize_signal(
            IncomingSignal(
                actor=Actor.USER,
                signal_type="user.message",
                content="Keep our shared continuity journal alive.",
                channel="fixture",
                channel_message_id=ids.new(),
                conversation_id="conversation-1",
            ),
            event_id=ids.new(),
            correlation_id=ids.new(),
            now_ms=clock.now_ms() - 8 * 3_600_000,
            source_kind=SourceKind.USER_OBSERVED,
        )
    )
    EmotionEpisodeRepository(database.connection).insert(
        EmotionEpisode(
            id=ids.new(),
            conversation_id="conversation-1",
            emotion_type=EmotionType.ATTACHMENT,
            target="our continuity journal",
            action_tendency="reach_out",
            intensity=0.95,
            inhibition=0.0,
            half_life_minutes=480,
            source_event_ids=[event.id],
            created_at_ms=clock.now_ms(),
            updated_at_ms=clock.now_ms(),
        )
    )
    return event.id


def _states(clock: FrozenClock) -> tuple[OrganismState, RelationshipState]:
    organism = OrganismState.initial(clock.now_ms()).model_copy(
        update={"energy": 0.95, "connection_need": 0.95, "safety": 0.95}
    )
    relationship = RelationshipState.initial(clock.now_ms()).model_copy(
        update={"trust": 0.9, "closeness": 0.8}
    )
    return organism, relationship


@pytest.mark.asyncio
async def test_approved_candidate_keeps_evidence_and_enqueues_once(tmp_path: Path) -> None:
    llm = FakeLLMAdapter()
    llm.set_response("initiative_draft", "I remembered our journal. Want to add one line together?")
    database, clock, ids, service, events, initiatives, outbox, usage = _setup(
        tmp_path,
        llm=llm,
    )
    try:
        evidence_id = _seed_evidence(database, clock, ids, events)
        organism, relationship = _states(clock)

        first = await service.evaluate("conversation-1", organism, relationship)
        second = await service.evaluate("conversation-1", organism, relationship)

        assert first.approved is True
        assert first.initiative is not None
        assert first.initiative.status == InitiativeStatus.QUEUED
        assert evidence_id in first.initiative.source_event_ids
        assert set(first.initiative.decision["components"]) == {
            "motive",
            "relevance",
            "goal_urgency",
            "connection_need",
            "opportunity",
            "intrusion",
            "repetition",
            "uncertainty",
            "fatigue",
        }
        assert second.gate == "duplicate"
        assert llm.call_count == 1
        assert usage.used("2030-03-05") == 1
        assert len(initiatives.list_active("conversation-1")) == 1
        assert outbox.find_by_dedup(f"outbox:{first.initiative.id}") is not None
        count = database.connection.execute("SELECT COUNT(*) AS c FROM outbox").fetchone()
        assert count["c"] == 1
    finally:
        database.close()


@pytest.mark.asyncio
async def test_quiet_hours_reject_before_llm_or_persistence(tmp_path: Path) -> None:
    llm = FakeLLMAdapter()
    config = InitiativeConfig(quiet_hours_start="00:00", quiet_hours_end="23:59")
    database, clock, ids, service, events, initiatives, _outbox, usage = _setup(
        tmp_path,
        initiative_config=config,
        llm=llm,
    )
    try:
        _seed_evidence(database, clock, ids, events)
        decision = await service.evaluate("conversation-1", *_states(clock))

        assert decision.gate == "quiet_hours"
        assert initiatives.list_active("conversation-1") == []
        assert llm.call_count == 0
        assert usage.used("2030-03-05") == 0
    finally:
        database.close()


@pytest.mark.asyncio
async def test_cooldown_and_daily_limit_are_hard_gates(tmp_path: Path) -> None:
    config = InitiativeConfig(daily_limit=1, cooldown_minutes=240)
    database, clock, ids, service, events, initiatives, _outbox, _usage = _setup(
        tmp_path,
        initiative_config=config,
    )
    try:
        _seed_evidence(database, clock, ids, events)
        initiatives.insert(
            Initiative(
                id=ids.new(),
                conversation_id="conversation-1",
                motive="fixture",
                intent="initiate",
                content_draft="fixture",
                urgency=1.0,
                decision_score=1.0,
                status=InitiativeStatus.SENT,
                earliest_send_at_ms=clock.now_ms(),
                created_at_ms=clock.now_ms(),
                updated_at_ms=clock.now_ms(),
            )
        )
        organism, relationship = _states(clock)
        daily = await service.evaluate("conversation-1", organism, relationship)
        assert daily.gate == "daily_limit"

        database.connection.execute("DELETE FROM initiatives")
        config.daily_limit = 3
        initiatives.insert(
            Initiative(
                id=ids.new(),
                conversation_id="conversation-1",
                motive="fixture",
                intent="initiate",
                content_draft="fixture",
                urgency=1.0,
                decision_score=1.0,
                status=InitiativeStatus.SENT,
                earliest_send_at_ms=clock.now_ms(),
                created_at_ms=clock.now_ms(),
                updated_at_ms=clock.now_ms(),
            )
        )
        cooldown = await service.evaluate("conversation-1", organism, relationship)
        assert cooldown.gate == "cooldown"
    finally:
        database.close()


@pytest.mark.asyncio
async def test_background_budget_uses_deterministic_fallback(tmp_path: Path) -> None:
    llm = FakeLLMAdapter()
    database, clock, ids, service, events, _initiatives, _outbox, usage = _setup(
        tmp_path,
        llm=llm,
    )
    try:
        _seed_evidence(database, clock, ids, events)
        assert usage.try_consume("2030-03-05", clock.now_ms(), limit=20, amount=20)
        decision = await service.evaluate("conversation-1", *_states(clock))

        assert decision.approved is True
        assert decision.initiative is not None
        assert decision.initiative.decision["draft_source"] == "budget_fallback"
        assert llm.call_count == 0
        assert usage.used("2030-03-05") == 20
    finally:
        database.close()
