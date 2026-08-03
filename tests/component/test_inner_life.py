"""Contextual waiting, heartbeat recurrence, and inner-loop persistence tests."""

from __future__ import annotations

from pathlib import Path

from ssa.clock import FrozenClock
from ssa.config import DatabaseConfig, InnerLifeConfig
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, IncomingSignal, normalize_signal
from ssa.domain.lifecycle import InnerLifeMode
from ssa.domain.relationship import RelationshipState
from ssa.domain.state import OrganismState
from ssa.ids import SequentialIdGenerator
from ssa.services.inner_life_service import InnerLifeService
from ssa.storage.database import Database
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.lifecycle_repository import InnerLoopRepository


def _setup(
    tmp_path: Path,
) -> tuple[
    Database,
    FrozenClock,
    SequentialIdGenerator,
    SqliteEventRepository,
    InnerLoopRepository,
    InnerLifeService,
]:
    database = Database(DatabaseConfig(path=str(tmp_path / "inner-life.db")))
    database.initialize()
    clock = FrozenClock(1_900_000_000_000)
    ids = SequentialIdGenerator("inner")
    events = SqliteEventRepository(database.connection)
    states = InnerLoopRepository(database.connection)
    service = InnerLifeService(
        states=states,
        events=events,
        clock=clock,
        ids=ids,
        config=InnerLifeConfig(
            heartbeat_seconds=10,
            min_wait_minutes=1,
            base_wait_minutes=5,
            max_wait_minutes=120,
            offline_entry_threshold=0.58,
            offline_exit_threshold=0.42,
        ),
    )
    return database, clock, ids, events, states, service


def _event(
    events: SqliteEventRepository,
    clock: FrozenClock,
    ids: SequentialIdGenerator,
    *,
    conversation_id: str,
    actor: Actor,
    content: str,
    parent_event_id: str | None = None,
) -> Event:
    return events.append(
        normalize_signal(
            IncomingSignal(
                actor=actor,
                signal_type=f"{actor.value}.message",
                content=content,
                channel="fixture",
                channel_message_id=ids.new(),
                conversation_id=conversation_id,
                parent_event_id=parent_event_id,
            ),
            event_id=ids.new(),
            correlation_id=ids.new(),
            now_ms=clock.now_ms(),
            source_kind=(
                SourceKind.USER_OBSERVED if actor == Actor.USER else SourceKind.AGENT_OUTPUT
            ),
        )
    )


def test_context_changes_reply_expectation_and_wait_horizon(tmp_path: Path) -> None:
    database, clock, ids, events, states, service = _setup(tmp_path)
    organism = OrganismState.initial(clock.now_ms())
    relationship = RelationshipState.initial(clock.now_ms())
    try:
        waiting_user = _event(
            events,
            clock,
            ids,
            conversation_id="waiting",
            actor=Actor.USER,
            content="等我一下, 我忙完晚点回来继续。",
        )
        service.observe_user_event(waiting_user)
        waiting_agent = _event(
            events,
            clock,
            ids,
            conversation_id="waiting",
            actor=Actor.AGENT,
            content="好, 我在这里。回来以后还想从刚才的问题继续吗?",
            parent_event_id=waiting_user.id,
        )
        waiting = service.observe_agent_event(
            waiting_agent,
            organism,
            relationship,
            user_event=waiting_user,
        )

        closing_user = _event(
            events,
            clock,
            ids,
            conversation_id="closing",
            actor=Actor.USER,
            content="谢谢, 今天先这样, 晚安。",
        )
        service.observe_user_event(closing_user)
        closing_agent = _event(
            events,
            clock,
            ids,
            conversation_id="closing",
            actor=Actor.AGENT,
            content="好, 晚安。",
            parent_event_id=closing_user.id,
        )
        closing = service.observe_agent_event(
            closing_agent,
            organism,
            relationship,
            user_event=closing_user,
        )

        assert waiting.mode == closing.mode == InnerLifeMode.WAITING
        assert waiting.reply_expectation > closing.reply_expectation + 0.5
        assert waiting.wait_deadline_at_ms is not None
        assert closing.wait_deadline_at_ms is not None
        assert waiting.wait_deadline_at_ms > closing.wait_deadline_at_ms
        assert states.get("waiting") == waiting
    finally:
        database.close()


def test_heartbeat_enters_offline_mode_and_user_return_resets_it(tmp_path: Path) -> None:
    database, clock, ids, events, _states, service = _setup(tmp_path)
    organism = OrganismState.initial(clock.now_ms())
    relationship = RelationshipState.initial(clock.now_ms())
    try:
        user = _event(
            events,
            clock,
            ids,
            conversation_id="conversation-1",
            actor=Actor.USER,
            content="谢谢, 先这样, 晚安。",
        )
        service.observe_user_event(user)
        agent = _event(
            events,
            clock,
            ids,
            conversation_id="conversation-1",
            actor=Actor.AGENT,
            content="晚安。",
            parent_event_id=user.id,
        )
        waiting = service.observe_agent_event(
            agent,
            organism,
            relationship,
            user_event=user,
        )
        assert waiting.wait_deadline_at_ms is not None
        clock.set_ms(waiting.wait_deadline_at_ms + 1)

        heartbeat = service.heartbeat("conversation-1", organism, relationship)

        assert heartbeat.state.mode == InnerLifeMode.OFFLINE
        assert heartbeat.state.offline_readiness >= 0.58
        assert heartbeat.state.transition_reason.startswith("runtime_resumed:")
        assert heartbeat.transition_event is not None
        assert heartbeat.transition_event.event_type == "lifecycle.inner_transition"
        assert service.allows_offline_action("conversation-1") is True
        assert service.allows_proactive_contact("conversation-1") is True

        returned = _event(
            events,
            clock,
            ids,
            conversation_id="conversation-1",
            actor=Actor.USER,
            content="我回来了。",
        )
        engaged = service.observe_user_event(returned)

        assert engaged.mode == InnerLifeMode.ENGAGED
        assert engaged.reply_expectation == 0.0
        assert service.allows_offline_action("conversation-1") is False
        assert "mode=engaged" in service.context_summary("conversation-1")
    finally:
        database.close()
