"""Web gateway lifecycle ownership and delivery buffering."""

from __future__ import annotations

import asyncio
from contextlib import suppress

import pytest

from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, IncomingSignal, normalize_signal
from ssa.interfaces.web_gateway import _LifecycleEventBuffer, _run_lifecycle_pump


def _event(event_id: str) -> Event:
    return normalize_signal(
        IncomingSignal(
            actor=Actor.SYSTEM,
            signal_type="lifecycle.test",
            content=event_id,
            channel="test",
            channel_message_id=event_id,
            conversation_id="test-conversation",
        ),
        event_id=event_id,
        correlation_id=f"correlation-{event_id}",
        now_ms=1_900_000_000_000,
        source_kind=SourceKind.SYSTEM_DERIVED,
    )


def test_lifecycle_event_buffer_is_bounded_and_drains_once() -> None:
    event_buffer = _LifecycleEventBuffer(max_events=2)
    event_buffer.extend((_event("event-1"), _event("event-2"), _event("event-3")))

    assert [event.id for event in event_buffer.drain()] == ["event-2", "event-3"]
    assert event_buffer.drain() == ()


class _FakeLifecycleSession:
    def __init__(self) -> None:
        self.inference_ready = True
        self.polled = asyncio.Event()
        self.calls = 0

    async def poll_lifecycle(self) -> tuple[Event, ...]:
        self.calls += 1
        self.polled.set()
        return (_event("background-event"),)


@pytest.mark.asyncio
async def test_lifecycle_pump_ticks_without_a_client_poll() -> None:
    session = _FakeLifecycleSession()
    event_buffer = _LifecycleEventBuffer()
    task = asyncio.create_task(
        _run_lifecycle_pump(session, event_buffer, poll_seconds=60.0)  # type: ignore[arg-type]
    )
    try:
        await asyncio.wait_for(session.polled.wait(), timeout=1.0)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    assert session.calls == 1
    assert [event.id for event in event_buffer.drain()] == ["background-event"]
