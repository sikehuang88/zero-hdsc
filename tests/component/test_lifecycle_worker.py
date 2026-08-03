"""Component tests for the persistent lifecycle worker."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssa.clock import FrozenClock
from ssa.config import DatabaseConfig, Settings
from ssa.domain.lifecycle import JobStatus, ScheduledJob
from ssa.ids import SequentialIdGenerator
from ssa.runtime.lifecycle import PersistentLifecycleWorker
from ssa.storage.database import Database
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.lifecycle_repository import ScheduledJobRepository
from ssa.storage.state_repository import StateRepository


def _worker(
    tmp_path: Path,
) -> tuple[
    Database,
    FrozenClock,
    PersistentLifecycleWorker,
    ScheduledJobRepository,
    SqliteEventRepository,
    StateRepository,
]:
    settings = Settings(database=DatabaseConfig(path=str(tmp_path / "worker.db")))
    database = Database(settings.database)
    database.initialize()
    clock = FrozenClock(1_900_000_000_000)
    ids = SequentialIdGenerator("lifecycle")
    jobs = ScheduledJobRepository(database.connection)
    events = SqliteEventRepository(database.connection)
    states = StateRepository(database.connection, ids)
    worker = PersistentLifecycleWorker(
        jobs=jobs,
        events=events,
        states=states,
        clock=clock,
        ids=ids,
        conversation_id="conversation-1",
        state_config=settings.state,
        lease_ms=1_000,
    )
    return database, clock, worker, jobs, events, states


@pytest.mark.asyncio
async def test_worker_bootstrap_is_idempotent_and_initializes_state(tmp_path: Path) -> None:
    database, _clock, worker, jobs, events, states = _worker(tmp_path)

    try:
        first = worker.bootstrap()
        second = worker.bootstrap()
        assert [job.id for job in first] == [job.id for job in second]
        assert len(first) == 2

        result = await worker.run_due()
        assert result.leased == 2
        assert result.rescheduled == 2
        assert result.failed == 0
        assert states.latest() is not None
        assert events.recent_by_conversation("conversation-1")[-1].event_type == "world.time_tick"
        assert jobs.find_by_dedup("lifecycle:conversation-1:state.refresh") is not None
    finally:
        database.close()


@pytest.mark.asyncio
async def test_elapsed_time_advances_state_without_user_signal(tmp_path: Path) -> None:
    database, clock, worker, _jobs, events, states = _worker(tmp_path)

    try:
        worker.bootstrap()
        await worker.run_due()
        initial = states.latest()
        assert initial is not None

        clock.advance_ms(60 * 60 * 1_000)
        result = await worker.run_due()
        advanced = states.latest()

        assert result.failed == 0
        assert advanced is not None
        assert advanced.version == initial.version + 1
        assert advanced.connection_need > initial.connection_need
        assert advanced.updated_at_ms == clock.now_ms()
        ticks = [
            event
            for event in events.recent_by_conversation("conversation-1")
            if event.event_type == "world.time_tick"
        ]
        assert len(ticks) == 2
    finally:
        database.close()


@pytest.mark.asyncio
async def test_successful_recurring_job_does_not_exhaust_retry_budget(tmp_path: Path) -> None:
    database, clock, worker, jobs, _events, _states = _worker(tmp_path)
    calls = 0

    async def recurring(_job: ScheduledJob) -> None:
        nonlocal calls
        calls += 1

    try:
        worker.register("fixture.recurring", recurring, interval_ms=1_000)
        worker.bootstrap()
        for _index in range(12):
            await worker.run_due()
            clock.advance_ms(1_000)

        job = jobs.find_by_dedup("lifecycle:conversation-1:fixture.recurring")
        assert calls == 12
        assert job is not None
        assert job.attempt_count == 0
        assert job.last_error is None
    finally:
        database.close()


@pytest.mark.asyncio
async def test_worker_retries_handler_failure_and_recovers(tmp_path: Path) -> None:
    database, clock, worker, jobs, _events, _states = _worker(tmp_path)
    attempts = 0

    async def flaky(_job: ScheduledJob) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient failure")

    try:
        worker.register("fixture.flaky", flaky, interval_ms=60_000)
        worker.bootstrap()
        first = await worker.run_due()
        assert first.failed == 1
        failed_job = jobs.find_by_dedup("lifecycle:conversation-1:fixture.flaky")
        assert failed_job is not None
        assert failed_job.last_error is not None

        clock.advance_ms(30_000)
        second = await worker.run_due()
        assert second.failed == 0
        assert attempts == 2
    finally:
        database.close()


@pytest.mark.asyncio
async def test_bootstrap_revives_job_that_dead_lettered_before_handler_registration(
    tmp_path: Path,
) -> None:
    database, clock, worker, jobs, _events, _states = _worker(tmp_path)
    calls = 0
    job_type = "fixture.recovered-handler"
    dedup_key = f"lifecycle:conversation-1:{job_type}"

    async def recovered(_job: ScheduledJob) -> None:
        nonlocal calls
        calls += 1

    try:
        jobs.enqueue(
            ScheduledJob(
                id="missing-handler-job",
                conversation_id="conversation-1",
                job_type=job_type,
                dedup_key=dedup_key,
                payload={"recurring": True},
                due_at_ms=clock.now_ms(),
                max_attempts=1,
                created_at_ms=clock.now_ms(),
                updated_at_ms=clock.now_ms(),
            )
        )
        failed = await worker.run_due()
        dead_letter = jobs.find_by_dedup(dedup_key)
        assert failed.dead_lettered == 1
        assert dead_letter is not None
        assert dead_letter.status == JobStatus.DEAD_LETTER

        worker.register(job_type, recovered, interval_ms=60_000)
        worker.bootstrap()
        revived = jobs.find_by_dedup(dedup_key)
        assert revived is not None
        assert revived.status == JobStatus.PENDING
        assert revived.attempt_count == 0
        assert revived.last_error is None

        completed = await worker.run_due()
        assert completed.failed == 0
        assert calls == 1
    finally:
        database.close()
