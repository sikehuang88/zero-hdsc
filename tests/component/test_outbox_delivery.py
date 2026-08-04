"""Reliable outbox delivery tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssa.clock import FrozenClock
from ssa.config import DatabaseConfig, InitiativeConfig, Settings
from ssa.domain.lifecycle import Initiative, InitiativeStatus, OutboxMessage, OutboxStatus
from ssa.domain.relationship_preferences import ProactiveFrequency, RelationshipPreferences
from ssa.ids import SequentialIdGenerator
from ssa.services.outbox_service import DeliveryReceipt, OutboxDeliveryService
from ssa.services.proactive_contact_policy import ProactiveContactPolicy
from ssa.storage.database import Database
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.lifecycle_repository import InitiativeRepository, OutboxRepository
from ssa.storage.relationship_preferences_repository import RelationshipPreferencesRepository


class _FailOnceChannel:
    def __init__(self) -> None:
        self.calls = 0

    async def deliver(self, message: OutboxMessage) -> DeliveryReceipt:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary channel failure")
        return DeliveryReceipt(channel_message_id=message.id)


def _setup(
    tmp_path: Path,
) -> tuple[
    Database,
    FrozenClock,
    SequentialIdGenerator,
    InitiativeRepository,
    OutboxRepository,
    SqliteEventRepository,
]:
    settings = Settings(database=DatabaseConfig(path=str(tmp_path / "outbox.db")))
    database = Database(settings.database)
    database.initialize()
    return (
        database,
        FrozenClock(1_900_000_000_000),
        SequentialIdGenerator("delivery"),
        InitiativeRepository(database.connection),
        OutboxRepository(database.connection),
        SqliteEventRepository(database.connection),
    )


def _queued(
    clock: FrozenClock,
    ids: SequentialIdGenerator,
    initiatives: InitiativeRepository,
    outbox: OutboxRepository,
) -> tuple[Initiative, OutboxMessage]:
    initiative = initiatives.insert(
        Initiative(
            id=ids.new(),
            conversation_id="conversation-1",
            motive="connection_need",
            intent="initiate",
            content_draft="I thought of you.",
            urgency=0.9,
            decision_score=0.8,
            status=InitiativeStatus.QUEUED,
            earliest_send_at_ms=clock.now_ms(),
            correlation_id=ids.new(),
            created_at_ms=clock.now_ms(),
            updated_at_ms=clock.now_ms(),
        )
    )
    message = outbox.enqueue(
        OutboxMessage(
            id=ids.new(),
            correlation_id=initiative.correlation_id or ids.new(),
            channel="local",
            recipient="conversation-1",
            payload={"content": initiative.content_draft},
            next_attempt_at_ms=clock.now_ms(),
            created_at_ms=clock.now_ms(),
            initiative_id=initiative.id,
            dedup_key=f"outbox:{initiative.id}",
        )
    )
    return initiative, message


def _proactive_policy(
    database: Database,
    preferences: RelationshipPreferences,
) -> tuple[ProactiveContactPolicy, RelationshipPreferencesRepository]:
    repository = RelationshipPreferencesRepository(str(database.path))
    repository.save(preferences)
    return (
        ProactiveContactPolicy(
            preferences=repository,
            config=InitiativeConfig(),
            timezone="UTC",
        ),
        repository,
    )


@pytest.mark.asyncio
async def test_local_delivery_creates_one_proactive_event(tmp_path: Path) -> None:
    database, clock, ids, initiatives, outbox, events = _setup(tmp_path)
    try:
        initiative, message = _queued(clock, ids, initiatives, outbox)
        service = OutboxDeliveryService(
            outbox=outbox,
            initiatives=initiatives,
            events=events,
            clock=clock,
            ids=ids,
        )

        first = await service.deliver_due()
        second = await service.deliver_due()

        assert first.delivered == 1
        assert second.claimed == 0
        delivered = outbox.get(message.id)
        assert delivered is not None
        assert delivered.status == OutboxStatus.DELIVERED
        sent = initiatives.get(initiative.id)
        assert sent is not None
        assert sent.status == InitiativeStatus.SENT
        proactive = [
            event
            for event in events.recent_by_conversation("conversation-1")
            if event.event_type == "agent.initiative"
        ]
        assert len(proactive) == 1
        assert proactive[0].metadata["proactive"] is True
    finally:
        database.close()


@pytest.mark.asyncio
async def test_failed_delivery_retries_after_backoff(tmp_path: Path) -> None:
    database, clock, ids, initiatives, outbox, events = _setup(tmp_path)
    channel = _FailOnceChannel()
    try:
        _initiative, message = _queued(clock, ids, initiatives, outbox)
        service = OutboxDeliveryService(
            outbox=outbox,
            initiatives=initiatives,
            events=events,
            clock=clock,
            ids=ids,
            adapters={"local": channel},
        )

        failed = await service.deliver_due()
        assert failed.failed == 1
        assert outbox.get(message.id).status == OutboxStatus.PENDING  # type: ignore[union-attr]

        clock.advance_ms(30_000)
        recovered = await service.deliver_due()
        assert recovered.delivered == 1
        assert channel.calls == 2
        assert outbox.get(message.id).status == OutboxStatus.DELIVERED  # type: ignore[union-attr]
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blocked_preferences",
    [
        RelationshipPreferences(
            proactive_frequency=ProactiveFrequency.OFF,
            quiet_hours_enabled=False,
        ),
        RelationshipPreferences(
            proactive_frequency=ProactiveFrequency.BALANCED,
            quiet_hours_enabled=True,
            quiet_hours_start="00:00",
            quiet_hours_end="23:59",
        ),
    ],
    ids=["disabled", "quiet-hours"],
)
async def test_preference_gate_keeps_due_message_pending_until_resumed(
    tmp_path: Path,
    blocked_preferences: RelationshipPreferences,
) -> None:
    database, clock, ids, initiatives, outbox, events = _setup(tmp_path)
    try:
        _initiative, message = _queued(clock, ids, initiatives, outbox)
        policy, repository = _proactive_policy(database, blocked_preferences)
        service = OutboxDeliveryService(
            outbox=outbox,
            initiatives=initiatives,
            events=events,
            clock=clock,
            ids=ids,
            proactive_policy=policy,
        )

        blocked = await service.deliver_due()
        pending = outbox.get(message.id)
        assert blocked.claimed == 0
        assert pending is not None
        assert pending.status == OutboxStatus.PENDING
        assert pending.attempt_count == 0

        repository.save(
            RelationshipPreferences(
                proactive_frequency=ProactiveFrequency.BALANCED,
                quiet_hours_enabled=False,
            )
        )
        resumed = await service.deliver_due()
        assert resumed.delivered == 1
        assert outbox.get(message.id).status == OutboxStatus.DELIVERED  # type: ignore[union-attr]
    finally:
        database.close()


def test_expired_delivery_lease_is_recovered(tmp_path: Path) -> None:
    database, clock, ids, initiatives, outbox, _events = _setup(tmp_path)
    try:
        _initiative, message = _queued(clock, ids, initiatives, outbox)
        first = outbox.claim_due(clock.now_ms(), lease_ms=1_000)
        assert [item.id for item in first] == [message.id]
        assert outbox.claim_due(clock.now_ms(), lease_ms=1_000) == []

        clock.advance_ms(1_000)
        recovered = outbox.claim_due(clock.now_ms(), lease_ms=1_000)
        assert [item.id for item in recovered] == [message.id]
        assert recovered[0].attempt_count == 2
    finally:
        database.close()
