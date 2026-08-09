"""Unified persistent runtime for autonomous digital-life background activity."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ssa.adapters.embedding import EmbeddingService, SentenceTransformerEmbeddingService
from ssa.adapters.llm import LLMAdapter
from ssa.clock import Clock, SystemClock
from ssa.config import Settings
from ssa.domain.appraisal import AppraisalResult
from ssa.domain.enums import ActionIntent, Actor
from ssa.domain.events import Event
from ssa.domain.lifecycle import Initiative, ScheduledJob
from ssa.domain.perception import SituationPerception
from ssa.domain.relationship import RelationshipState
from ssa.domain.state import DeterministicStateEngine, OrganismState
from ssa.ids import IdGenerator, UuidIdGenerator
from ssa.runtime.lifecycle import LifecycleTickResult, PersistentLifecycleWorker
from ssa.services.contact_episode_service import ContactEpisodeService
from ssa.services.goal_service import GoalService
from ssa.services.initiative_service import InitiativeService
from ssa.services.inner_life_service import InnerLifeService
from ssa.services.memory_write_service import MemoryWriteService
from ssa.services.offline_agency_service import OfflineAgencyService, OfflineCycleResult
from ssa.services.outbox_service import (
    ChannelAdapter,
    LocalEventChannel,
    OutboxDeliveryService,
    WindowsDesktopChannel,
)
from ssa.services.prediction_service import GroundedPredictionService
from ssa.services.proactive_contact_policy import ProactiveContactPolicy
from ssa.services.reflective_learning_service import ReflectiveLearningService
from ssa.services.resting_state_service import RestingStateService
from ssa.services.world_observation_service import WorldObservationService
from ssa.storage.database import Database
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.learning_repository import SqliteLearningRepository
from ssa.storage.lifecycle_repository import (
    BackgroundUsageRepository,
    CausalTraceRepository,
    ContactEpisodeRepository,
    EmotionEpisodeRepository,
    GoalRepository,
    InitiativeRepository,
    InnerLoopRepository,
    OfflineAgencyRepository,
    OutboxRepository,
    ScheduledJobRepository,
    WorldObservationRepository,
)
from ssa.storage.memory_repository import SqliteMemoryRepository
from ssa.storage.prediction_repository import PredictionRepository
from ssa.storage.relationship_preferences_repository import RelationshipPreferencesRepository
from ssa.storage.relationship_repository import RelationshipRepository
from ssa.storage.state_repository import StateRepository
from ssa.storage.trace_repository import SqliteTraceRepository
from ssa.tools.external_truth import ExternalTruthExecutor


@dataclass(frozen=True)
class AutonomousTickResult:
    lifecycle: LifecycleTickResult
    proactive_events: tuple[Event, ...]
    dashboard_events: tuple[Event, ...] = ()


class AutonomousRuntime:
    """Own and connect every durable P1-P6 lifecycle service."""

    def __init__(
        self,
        *,
        database: Database,
        llm: LLMAdapter,
        settings: Settings,
        conversation_id: str,
        clock: Clock | None = None,
        ids: IdGenerator | None = None,
        external_truth: ExternalTruthExecutor | None = None,
        embedding: EmbeddingService | None = None,
        prediction_service: GroundedPredictionService | None = None,
    ) -> None:
        self.conversation_id = conversation_id
        self._settings = settings
        self._clock = clock or SystemClock()
        self._ids = ids or UuidIdGenerator()
        self._enabled = settings.ablation.enable_lifecycle
        self._lock = asyncio.Lock()
        self._proactive_events: list[Event] = []
        self._dashboard_events: list[Event] = []
        self._state_engine = DeterministicStateEngine(settings.state)

        connection = database.connection
        self.events = SqliteEventRepository(connection)
        self.predictions = (
            prediction_service.repository
            if prediction_service is not None
            else PredictionRepository(connection)
        )
        self.prediction_service = prediction_service or GroundedPredictionService(
            predictions=self.predictions,
            events=self.events,
            clock=self._clock,
            ids=self._ids,
        )
        self.states = StateRepository(connection, self._ids)
        self.relationships = RelationshipRepository(connection, self._ids)
        self.traces = SqliteTraceRepository(connection)
        self.jobs = ScheduledJobRepository(connection)
        self.goals = GoalRepository(connection)
        self.emotions = EmotionEpisodeRepository(connection)
        self.initiatives = InitiativeRepository(connection)
        self.inner_states = InnerLoopRepository(connection)
        self.outbox = OutboxRepository(connection)
        self.offline_episodes = OfflineAgencyRepository(connection)
        self.contacts = ContactEpisodeRepository(connection)
        self.world_observations = WorldObservationRepository(connection)
        self.learning = SqliteLearningRepository(connection)
        self.memories = SqliteMemoryRepository(connection)
        self.relationship_preferences = RelationshipPreferencesRepository(
            str(database.path),
            busy_timeout_ms=settings.database.busy_timeout_ms,
        )
        self.proactive_policy = ProactiveContactPolicy(
            preferences=self.relationship_preferences,
            config=settings.initiative,
            timezone=settings.app.timezone,
        )
        embedding_service = embedding or SentenceTransformerEmbeddingService(settings.embedding)

        self.goal_service = GoalService(
            goals=self.goals,
            traces=self.traces,
            events=self.events,
            clock=self._clock,
            ids=self._ids,
            config=settings.goal,
        )
        self.resting_service = RestingStateService(
            traces=self.traces,
            emotions=self.emotions,
            events=self.events,
            clock=self._clock,
            ids=self._ids,
            capacity=settings.hdsc.h2_active_capacity,
        )
        self.initiative_service = InitiativeService(
            initiatives=self.initiatives,
            outbox=self.outbox,
            emotions=self.emotions,
            goals=self.goals,
            traces=self.traces,
            events=self.events,
            causal_traces=CausalTraceRepository(connection),
            usage=BackgroundUsageRepository(connection),
            clock=self._clock,
            ids=self._ids,
            initiative_config=settings.initiative,
            budget_config=settings.budget,
            goal_config=settings.goal,
            llm_config=settings.llm,
            timezone=settings.app.timezone,
            proactive_policy=self.proactive_policy,
            llm=llm,
        )
        self.contact_service = ContactEpisodeService(
            contacts=self.contacts,
            initiatives=self.initiatives,
            outbox=self.outbox,
            events=self.events,
            clock=self._clock,
            ids=self._ids,
            timezone=settings.app.timezone,
            proactive_policy=self.proactive_policy,
        )
        self.inner_service = InnerLifeService(
            states=self.inner_states,
            events=self.events,
            clock=self._clock,
            ids=self._ids,
            config=settings.inner_life,
        )
        adapters: dict[str, ChannelAdapter] = {
            "local": LocalEventChannel(),
            "desktop": WindowsDesktopChannel(settings.tools.powershell_executable),
        }
        self.delivery_service = OutboxDeliveryService(
            outbox=self.outbox,
            initiatives=self.initiatives,
            events=self.events,
            clock=self._clock,
            ids=self._ids,
            proactive_policy=self.proactive_policy,
            adapters=adapters,
            on_delivered=self._on_delivered,
        )
        self.world_service = WorldObservationService(
            observations=self.world_observations,
            events=self.events,
            clock=self._clock,
            ids=self._ids,
            config=settings.world,
            timezone=settings.app.timezone,
        )
        truth_executor = external_truth
        if (
            truth_executor is None
            and settings.external_truth.enabled
            and "news" in settings.external_truth.enabled_sources
        ):
            truth_executor = ExternalTruthExecutor(settings.external_truth)
        self.offline_service = OfflineAgencyService(
            episodes=self.offline_episodes,
            goals=self.goals,
            traces=self.traces,
            events=self.events,
            usage=BackgroundUsageRepository(connection),
            external_truth=truth_executor,
            llm=llm,
            clock=self._clock,
            ids=self._ids,
            config=settings.offline_agency,
            budget_config=settings.budget,
            llm_config=settings.llm,
            timezone=settings.app.timezone,
        )
        memory_write = MemoryWriteService(
            llm,
            embedding_service,
            self.memories,
            self.events.get,
            self._ids,
            self._clock,
            settings.retrieval,
            embedding_model_name=settings.embedding.model,
            embedding_dim=settings.embedding.dim,
            default_model=settings.llm.model,
        )
        self.reflective_service = ReflectiveLearningService(
            learning=self.learning,
            offline=self.offline_episodes,
            events=self.events,
            traces=self.traces,
            memory_write=memory_write,
            clock=self._clock,
            ids=self._ids,
            config=settings.reflective_learning,
        )
        self.worker = PersistentLifecycleWorker(
            jobs=self.jobs,
            events=self.events,
            states=self.states,
            clock=self._clock,
            ids=self._ids,
            conversation_id=conversation_id,
            state_config=settings.state,
        )
        self._register_jobs()

    def bootstrap(self) -> list[ScheduledJob]:
        if not self._enabled:
            return []
        return self.worker.bootstrap()

    async def tick(self, *, limit: int = 32) -> AutonomousTickResult:
        async with self._lock:
            self._proactive_events = []
            self._dashboard_events = []
            if not self._enabled:
                return AutonomousTickResult(_empty_tick(), (), ())
            lifecycle = await self.worker.run_due(limit=limit)
            return AutonomousTickResult(
                lifecycle,
                tuple(self._proactive_events),
                tuple(self._dashboard_events),
            )

    def observe_user_event(self, event: Event) -> None:
        if self._enabled:
            if self._settings.inner_life.enabled:
                self.inner_service.observe_user_event(event)
            self.contact_service.observe_user_event(event)
            self.emotions.resolve_all(event.conversation_id, event.created_at_ms)
            now_ms = self._clock.now_ms()
            self.jobs.expedite(
                f"lifecycle:{self.conversation_id}:learning.evaluate_outcomes",
                now_ms,
                now_ms,
            )
            self.jobs.expedite(
                f"lifecycle:{self.conversation_id}:learning.resolve_predictions",
                now_ms,
                now_ms,
            )

    def observe_agent_event(
        self,
        event: Event,
        perception: SituationPerception | None = None,
        *,
        user_event: Event | None = None,
    ) -> None:
        if self._enabled and self._settings.inner_life.enabled:
            self.inner_service.observe_agent_event(
                event,
                self._organism(),
                self._relationship(),
                user_event=user_event,
                perception=perception,
            )

    def context_summary(self) -> str:
        return "\n\n".join(
            (
                self.inner_service.context_summary(self.conversation_id),
                self.offline_service.context_summary(self.conversation_id),
                self.reflective_service.context_summary(self.conversation_id),
            )
        )

    def _on_delivered(self, initiative: Initiative, event: Event) -> None:
        self.contact_service.on_delivered(initiative, event)
        self.observe_agent_event(
            event,
            user_event=self.events.latest_by_actor(self.conversation_id, Actor.USER),
        )

    def _register_jobs(self) -> None:
        self.worker.register(
            "agency.offline_cycle",
            self._offline_cycle,
            interval_ms=self._settings.offline_agency.interval_minutes * 60_000,
        )
        self.worker.register(
            "inner.heartbeat",
            self._inner_heartbeat,
            interval_ms=self._settings.inner_life.heartbeat_seconds * 1_000,
        )
        self.worker.register(
            "reflection.schedule",
            self._reflection_schedule,
            interval_ms=self._settings.reflective_learning.schedule_interval_minutes * 60_000,
        )
        self.worker.register(
            "learning.consolidate",
            self._learning_consolidate,
            interval_ms=(
                self._settings.reflective_learning.consolidation_interval_minutes * 60_000
            ),
        )
        self.worker.register(
            "learning.evaluate_outcomes",
            self._learning_evaluate_outcomes,
            interval_ms=self._settings.reflective_learning.outcome_interval_minutes * 60_000,
        )
        self.worker.register(
            "learning.resolve_predictions",
            self._learning_resolve_predictions,
            interval_ms=self._settings.reflective_learning.outcome_interval_minutes * 60_000,
        )
        self.worker.register(
            "trace.resting_step",
            self._resting_step,
            interval_ms=30 * 60_000,
        )
        self.worker.register("goal.advance", self._goal_advance, interval_ms=60 * 60_000)
        self.worker.register(
            "initiative.evaluate",
            self._initiative_evaluate,
            interval_ms=30 * 60_000,
        )
        self.worker.register(
            "initiative.expire",
            self._initiative_expire,
            interval_ms=30 * 60_000,
        )
        self.worker.register(
            "contact.evaluate",
            self._contact_evaluate,
            interval_ms=60 * 60_000,
        )
        self.worker.register("outbox.deliver", self._outbox_deliver, interval_ms=30_000)
        self.worker.register(
            "world.observe",
            self._world_observe,
            interval_ms=self._settings.world.observe_interval_seconds * 1_000,
        )

    async def _resting_step(self, _job: ScheduledJob) -> None:
        self.resting_service.step(
            conversation_id=self.conversation_id,
            organism=self._organism(),
            relationship=self._relationship(),
            goals=self.goals.list_active(self.conversation_id),
        )

    async def _offline_cycle(self, _job: ScheduledJob) -> None:
        if self.states.latest() is None:
            return
        if self._settings.inner_life.enabled and not self.inner_service.allows_offline_action(
            self.conversation_id
        ):
            return
        result = await self.offline_service.run_cycle(
            self.conversation_id,
            self._organism(),
            self._relationship(),
        )
        self._apply_offline_cost(result)
        if result.event is not None:
            self._dashboard_events.append(result.event)
            now_ms = self._clock.now_ms()
            self.jobs.expedite(
                f"lifecycle:{self.conversation_id}:reflection.schedule",
                now_ms,
                now_ms,
            )

    async def _goal_advance(self, _job: ScheduledJob) -> None:
        if not self._settings.ablation.enable_goals:
            return
        existing = self.goals.list_open(self.conversation_id)
        self.goal_service.ensure_self_project(self.conversation_id, self._organism())
        if existing:
            self.goal_service.advance_due(self.conversation_id)

    async def _initiative_evaluate(self, _job: ScheduledJob) -> None:
        if self._settings.inner_life.enabled and not self.inner_service.allows_proactive_contact(
            self.conversation_id
        ):
            return
        await self.initiative_service.evaluate(
            self.conversation_id,
            self._organism(),
            self._relationship(),
        )

    async def _initiative_expire(self, _job: ScheduledJob) -> None:
        self.initiative_service.expire_due(self.conversation_id)

    async def _contact_evaluate(self, _job: ScheduledJob) -> None:
        if self._settings.inner_life.enabled and not self.inner_service.allows_proactive_contact(
            self.conversation_id
        ):
            return
        self.contact_service.evaluate_due(self.conversation_id)

    async def _inner_heartbeat(self, _job: ScheduledJob) -> None:
        if not self._settings.inner_life.enabled:
            return
        result = self.inner_service.heartbeat(
            self.conversation_id,
            self._organism(),
            self._relationship(),
        )
        if result.transition_event is not None:
            self._dashboard_events.append(result.transition_event)
            if result.transition_event.metadata.get("to_mode") == "offline":
                now_ms = self._clock.now_ms()
                self.jobs.expedite(
                    f"lifecycle:{self.conversation_id}:agency.offline_cycle",
                    now_ms,
                    now_ms,
                )

    async def _reflection_schedule(self, _job: ScheduledJob) -> None:
        if not self._settings.reflective_learning.enabled:
            return
        if self._settings.inner_life.enabled and not self.inner_service.allows_offline_action(
            self.conversation_id
        ):
            return
        result = self.reflective_service.schedule_reflections(
            self.conversation_id,
            self._organism(),
            self._relationship(),
            self.goals.list_open(self.conversation_id),
        )
        self._dashboard_events.extend(result.events)
        if any(run.status.value == "completed" for run in result.runs):
            now_ms = self._clock.now_ms()
            self.jobs.expedite(
                f"lifecycle:{self.conversation_id}:learning.consolidate",
                now_ms,
                now_ms,
            )

    async def _learning_consolidate(self, _job: ScheduledJob) -> None:
        if not self._settings.reflective_learning.enabled:
            return
        if self._settings.inner_life.enabled and not self.inner_service.allows_offline_action(
            self.conversation_id
        ):
            return
        result = self.reflective_service.consolidate(self.conversation_id)
        self._dashboard_events.extend(result.events)

    async def _learning_evaluate_outcomes(self, _job: ScheduledJob) -> None:
        result = self.reflective_service.evaluate_outcomes(self.conversation_id)
        self._dashboard_events.extend(result.events)

    async def _learning_resolve_predictions(self, _job: ScheduledJob) -> None:
        result = self.prediction_service.resolve_due(self.conversation_id)
        self._dashboard_events.extend(result.events)

    async def _outbox_deliver(self, _job: ScheduledJob) -> None:
        result = await self.delivery_service.deliver_due(limit=16)
        self._proactive_events.extend(result.events)
        self._dashboard_events.extend(result.events)

    async def _world_observe(self, _job: ScheduledJob) -> None:
        result = self.world_service.observe(self.conversation_id)
        if result.material_count:
            now_ms = self._clock.now_ms()
            prefix = f"lifecycle:{self.conversation_id}:"
            self.jobs.expedite(f"{prefix}trace.resting_step", now_ms, now_ms)
            self.jobs.expedite(f"{prefix}initiative.evaluate", now_ms, now_ms)
            self.jobs.expedite(f"{prefix}learning.evaluate_outcomes", now_ms, now_ms)
            self.jobs.expedite(f"{prefix}learning.resolve_predictions", now_ms, now_ms)

    def _organism(self) -> OrganismState:
        return self.states.latest() or OrganismState.initial(self._clock.now_ms())

    def _relationship(self) -> RelationshipState:
        return self.relationships.latest() or RelationshipState.initial(self._clock.now_ms())

    def _apply_offline_cost(self, result: OfflineCycleResult) -> None:
        if result.event is None:
            return
        current = self.states.latest()
        if current is None:
            return
        preview = self._state_engine.preview(
            current,
            AppraisalResult.neutral(),
            self._clock.now_ms(),
        )
        updated = self._state_engine.finalize(preview, ActionIntent.PROJECT_WORK)
        self.states.append_if_version(current.version, updated, result.event.id)


def _empty_tick() -> LifecycleTickResult:
    return LifecycleTickResult(
        leased=0,
        completed=0,
        rescheduled=0,
        failed=0,
        dead_lettered=0,
        job_ids=(),
    )


__all__ = ["AutonomousRuntime", "AutonomousTickResult"]
