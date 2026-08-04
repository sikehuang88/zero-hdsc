"""Contact episode state-machine tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssa.clock import FrozenClock
from ssa.config import DatabaseConfig, InitiativeConfig, Settings
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import IncomingSignal, normalize_signal
from ssa.domain.lifecycle import ContactPhase, ContactStatus, Initiative, InitiativeStatus
from ssa.domain.relationship_preferences import RelationshipPreferences
from ssa.ids import SequentialIdGenerator
from ssa.services.contact_episode_service import ContactEpisodeService
from ssa.services.outbox_service import OutboxDeliveryService
from ssa.services.proactive_contact_policy import ProactiveContactPolicy
from ssa.storage.database import Database
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.lifecycle_repository import (
    ContactEpisodeRepository,
    InitiativeRepository,
    OutboxRepository,
)
from ssa.storage.relationship_preferences_repository import RelationshipPreferencesRepository


def _setup(
    tmp_path: Path,
) -> tuple[
    Database,
    FrozenClock,
    SequentialIdGenerator,
    ContactEpisodeService,
    OutboxDeliveryService,
    ContactEpisodeRepository,
    InitiativeRepository,
    SqliteEventRepository,
]:
    settings = Settings(database=DatabaseConfig(path=str(tmp_path / "contact.db")))
    database = Database(settings.database)
    database.initialize()
    clock = FrozenClock(1_900_000_000_000)
    ids = SequentialIdGenerator("contact")
    contacts = ContactEpisodeRepository(database.connection)
    initiatives = InitiativeRepository(database.connection)
    outbox = OutboxRepository(database.connection)
    events = SqliteEventRepository(database.connection)
    preferences = RelationshipPreferencesRepository(str(database.path))
    preferences.save(RelationshipPreferences(quiet_hours_enabled=False))
    initiative_config = InitiativeConfig(
        daily_limit=8,
        cooldown_minutes=1,
        quiet_hours_start="00:00",
        quiet_hours_end="00:00",
        min_gap_hours=0,
    )
    proactive_policy = ProactiveContactPolicy(
        preferences=preferences,
        config=initiative_config,
        timezone="UTC",
    )
    contact_service = ContactEpisodeService(
        contacts=contacts,
        initiatives=initiatives,
        outbox=outbox,
        events=events,
        clock=clock,
        ids=ids,
        timezone="UTC",
        proactive_policy=proactive_policy,
        wait_hours=1,
        max_messages=3,
    )
    delivery = OutboxDeliveryService(
        outbox=outbox,
        initiatives=initiatives,
        events=events,
        clock=clock,
        ids=ids,
        proactive_policy=proactive_policy,
        on_delivered=contact_service.on_delivered,
    )
    return database, clock, ids, contact_service, delivery, contacts, initiatives, events


def _opening(
    clock: FrozenClock,
    ids: SequentialIdGenerator,
    initiatives: InitiativeRepository,
) -> Initiative:
    return initiatives.insert(
        Initiative(
            id=ids.new(),
            conversation_id="conversation-1",
            motive="attachment:reach_out",
            intent="initiate",
            content_draft="I thought of you.",
            urgency=0.9,
            decision_score=0.8,
            status=InitiativeStatus.SENT,
            earliest_send_at_ms=clock.now_ms(),
            correlation_id=ids.new(),
            source_event_ids=[],
            created_at_ms=clock.now_ms(),
            updated_at_ms=clock.now_ms(),
        )
    )


def _delivery_event(
    clock: FrozenClock,
    ids: SequentialIdGenerator,
    events: SqliteEventRepository,
    initiative: Initiative,
):
    return events.append(
        normalize_signal(
            IncomingSignal(
                actor=Actor.AGENT,
                signal_type="agent.initiative",
                content=initiative.content_draft,
                channel="fixture",
                channel_message_id=ids.new(),
                conversation_id=initiative.conversation_id,
            ),
            event_id=initiative.sent_event_id or ids.new(),
            correlation_id=initiative.correlation_id or ids.new(),
            now_ms=clock.now_ms(),
            source_kind=SourceKind.AGENT_OUTPUT,
        )
    )


@pytest.mark.asyncio
async def test_silence_followups_are_bounded_and_withdraw(tmp_path: Path) -> None:
    database, clock, ids, service, delivery, contacts, initiatives, events = _setup(tmp_path)
    try:
        initiative = _opening(clock, ids, initiatives)
        episode = service.on_delivered(initiative, _delivery_event(clock, ids, events, initiative))
        assert episode.phase == ContactPhase.WAITING
        assert episode.message_count == 1

        for expected_count in (2, 3):
            clock.advance_ms(expected_count * 3_600_000)
            result = service.evaluate_due("conversation-1")
            assert result.follow_ups_queued == 1
            delivered = await delivery.deliver_due()
            assert delivered.delivered == 1
            current = contacts.get(episode.id)
            assert current is not None
            assert current.message_count == expected_count
            assert current.phase == ContactPhase.WAITING
            assert current.hypotheses["avoidance"] < 0.5

        clock.advance_ms(4 * 3_600_000)
        withdrawn = service.evaluate_due("conversation-1")
        assert withdrawn.withdrawn == 1
        current = contacts.get(episode.id)
        assert current is not None
        assert current.phase == ContactPhase.WITHDRAWAL
        assert current.status == ContactStatus.CANCELLED
        assert current.message_count == 3
    finally:
        database.close()


def test_disabled_preference_blocks_silence_follow_up(tmp_path: Path) -> None:
    database, clock, ids, service, _delivery, contacts, initiatives, events = _setup(tmp_path)
    try:
        initiative = _opening(clock, ids, initiatives)
        episode = service.on_delivered(initiative, _delivery_event(clock, ids, events, initiative))
        RelationshipPreferencesRepository(str(database.path)).save(
            RelationshipPreferences(
                proactive_frequency="off",
                quiet_hours_enabled=False,
            )
        )
        clock.advance_ms(2 * 3_600_000)

        result = service.evaluate_due("conversation-1")

        assert result.follow_ups_queued == 0
        current = contacts.get(episode.id)
        assert current is not None
        assert current.message_count == 1
        assert current.phase == ContactPhase.WAITING
    finally:
        database.close()


def test_user_event_resolves_active_episode(tmp_path: Path) -> None:
    database, clock, ids, service, _delivery, contacts, initiatives, events = _setup(tmp_path)
    try:
        initiative = _opening(clock, ids, initiatives)
        episode = service.on_delivered(initiative, _delivery_event(clock, ids, events, initiative))
        clock.advance_ms(10_000)
        user_event = events.append(
            normalize_signal(
                IncomingSignal(
                    actor=Actor.USER,
                    signal_type="user.message",
                    content="I am back.",
                    channel="fixture",
                    channel_message_id=ids.new(),
                    conversation_id="conversation-1",
                ),
                event_id=ids.new(),
                correlation_id=ids.new(),
                now_ms=clock.now_ms(),
                source_kind=SourceKind.USER_OBSERVED,
            )
        )

        resolved = service.observe_user_event(user_event)

        assert [item.id for item in resolved] == [episode.id]
        current = contacts.get(episode.id)
        assert current is not None
        assert current.phase == ContactPhase.RESOLUTION
        assert current.status == ContactStatus.RESOLVED
        assert current.resolved_by_event_id == user_event.id
        assert service.evaluate_due("conversation-1").observed_silence == 0
    finally:
        database.close()
