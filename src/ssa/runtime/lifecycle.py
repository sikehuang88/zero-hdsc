"""Persistent lifecycle worker that advances the system without user input."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from ssa.clock import Clock
from ssa.config import StateConfig
from ssa.domain.appraisal import AppraisalResult
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, IncomingSignal, normalize_signal
from ssa.domain.lifecycle import JobStatus, ScheduledJob
from ssa.domain.state import DeterministicStateEngine, OrganismState
from ssa.ids import IdGenerator
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.lifecycle_repository import ScheduledJobRepository
from ssa.storage.state_repository import StateRepository

logger = logging.getLogger(__name__)

LifecycleJobHandler = Callable[[ScheduledJob], Awaitable[None]]


@dataclass(frozen=True)
class LifecycleTickResult:
    leased: int
    completed: int
    rescheduled: int
    failed: int
    dead_lettered: int
    job_ids: tuple[str, ...]


class PersistentLifecycleWorker:
    """Lease, execute, and reschedule durable lifecycle jobs."""

    def __init__(
        self,
        *,
        jobs: ScheduledJobRepository,
        events: SqliteEventRepository,
        states: StateRepository,
        clock: Clock,
        ids: IdGenerator,
        conversation_id: str,
        state_config: StateConfig,
        lease_ms: int = 60_000,
    ) -> None:
        if not conversation_id.strip():
            raise ValueError("conversation_id must not be empty")
        if lease_ms < 1:
            raise ValueError("lease_ms must be positive")
        self._jobs = jobs
        self._events = events
        self._states = states
        self._clock = clock
        self._ids = ids
        self._conversation_id = conversation_id
        self._state_engine = DeterministicStateEngine(state_config)
        self._state_threshold = state_config.refresh_delta_threshold
        self._lease_ms = lease_ms
        self._handlers: dict[str, LifecycleJobHandler] = {}
        self._intervals_ms: dict[str, int] = {
            "state.refresh": 60 * 60 * 1_000,
            "health.check": 60 * 1_000,
        }
        self.register("state.refresh", self._refresh_state)
        self.register("health.check", self._health_check)

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    def register(
        self,
        job_type: str,
        handler: LifecycleJobHandler,
        *,
        interval_ms: int | None = None,
    ) -> None:
        if not job_type.strip():
            raise ValueError("job_type must not be empty")
        if interval_ms is not None:
            if interval_ms < 1:
                raise ValueError("interval_ms must be positive")
            self._intervals_ms[job_type] = interval_ms
        self._handlers[job_type] = handler

    def bootstrap(self, *, due_at_ms: int | None = None) -> list[ScheduledJob]:
        """Create one durable row per recurring job, idempotently."""
        now_ms = self._clock.now_ms()
        first_due = now_ms if due_at_ms is None else due_at_ms
        scheduled: list[ScheduledJob] = []
        for job_type in sorted(self._intervals_ms):
            job = ScheduledJob(
                id=self._ids.new(),
                conversation_id=self._conversation_id,
                job_type=job_type,
                dedup_key=f"lifecycle:{self._conversation_id}:{job_type}",
                payload={"recurring": True},
                due_at_ms=first_due,
                created_at_ms=now_ms,
                updated_at_ms=now_ms,
            )
            stored = self._jobs.enqueue(job)
            if stored.status == JobStatus.DEAD_LETTER:
                stored = (
                    self._jobs.revive_missing_handler_dead_letter(
                        stored.id,
                        first_due,
                        now_ms,
                    )
                    or stored
                )
            scheduled.append(stored)
        return scheduled

    async def run_due(self, *, limit: int = 16) -> LifecycleTickResult:
        now_ms = self._clock.now_ms()
        leased = self._jobs.lease_due(now_ms, lease_ms=self._lease_ms, limit=limit)
        completed = 0
        rescheduled = 0
        failed = 0
        dead_lettered = 0
        processed_ids: list[str] = []

        for job in leased:
            processed_ids.append(job.id)
            handler = self._handlers.get(job.job_type)
            try:
                if handler is None:
                    raise LookupError(f"no lifecycle handler registered for {job.job_type!r}")
                await handler(job)
                interval_ms = self._intervals_ms.get(job.job_type)
                if interval_ms is not None and bool(job.payload.get("recurring", True)):
                    self._jobs.reschedule(job.id, now_ms + interval_ms, self._clock.now_ms())
                    rescheduled += 1
                else:
                    self._jobs.complete(job.id, self._clock.now_ms())
                    completed += 1
            except Exception as exc:
                failed += 1
                retry_delay = min(60 * 60 * 1_000, 30_000 * (2 ** max(0, job.attempt_count - 1)))
                failed_job = self._jobs.fail(
                    job.id,
                    f"{type(exc).__name__}: {exc}",
                    self._clock.now_ms(),
                    self._clock.now_ms() + retry_delay,
                )
                if failed_job.status == JobStatus.DEAD_LETTER:
                    dead_lettered += 1
                logger.warning("lifecycle job %s failed: %s", job.job_type, exc)

        return LifecycleTickResult(
            leased=len(leased),
            completed=completed,
            rescheduled=rescheduled,
            failed=failed,
            dead_lettered=dead_lettered,
            job_ids=tuple(processed_ids),
        )

    async def _refresh_state(self, job: ScheduledJob) -> None:
        now_ms = self._clock.now_ms()
        current = self._states.latest()
        if current is None:
            event = self._append_lifecycle_event(
                job,
                event_type="world.time_tick",
                content="The lifecycle clock initialized the organism state.",
            )
            self._states.append_if_version(0, OrganismState.initial(now_ms), event.id)
            return

        preview = self._state_engine.preview(current, _elapsed_time_appraisal(), now_ms)
        delta = max(
            abs(preview.energy - current.energy),
            abs(preview.connection_need - current.connection_need),
            abs(preview.autonomy_need - current.autonomy_need),
            abs(preview.curiosity - current.curiosity),
            abs(preview.safety - current.safety),
            abs(preview.valence - current.valence),
            abs(preview.arousal - current.arousal),
        )
        if delta < self._state_threshold:
            return
        event = self._append_lifecycle_event(
            job,
            event_type="world.time_tick",
            content="Elapsed digital time changed the organism state.",
            metadata={"elapsed_ms": max(0, now_ms - current.updated_at_ms), "delta": delta},
        )
        refreshed = preview.model_copy(update={"version": current.version + 1})
        self._states.append_if_version(current.version, refreshed, event.id)

    async def _health_check(self, _job: ScheduledJob) -> None:
        return

    def _append_lifecycle_event(
        self,
        job: ScheduledJob,
        *,
        event_type: str,
        content: str,
        metadata: Mapping[str, object] | None = None,
    ) -> Event:
        channel_message_id = f"{job.id}:{job.due_at_ms}"
        existing = self._events.find_by_channel_message("lifecycle", channel_message_id)
        if existing is not None:
            return existing
        event_id = self._ids.new()
        correlation_id = self._ids.new()
        return self._events.append(
            normalize_signal(
                IncomingSignal(
                    actor=Actor.SYSTEM,
                    signal_type=event_type,
                    content=content,
                    channel="lifecycle",
                    channel_message_id=channel_message_id,
                    conversation_id=self._conversation_id,
                    metadata={"job_id": job.id, "job_type": job.job_type, **(metadata or {})},
                ),
                event_id=event_id,
                correlation_id=correlation_id,
                now_ms=self._clock.now_ms(),
                source_kind=SourceKind.SYSTEM_DERIVED,
            )
        )


def _elapsed_time_appraisal() -> AppraisalResult:
    """Zero-signal appraisal used to expose only deterministic time dynamics."""
    return AppraisalResult(
        novelty=0.5,
        goal_congruence=0.0,
        controllability=0.5,
        certainty=0.5,
        self_agency=0.0,
        user_agency=0.0,
        external_agency=0.0,
        relationship_relevance=0.0,
        urgency=0.0,
        valence_signal=0.0,
        arousal_signal=0.0,
        explanation="Elapsed time without an external event.",
    )


__all__ = ["LifecycleJobHandler", "LifecycleTickResult", "PersistentLifecycleWorker"]
