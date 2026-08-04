"""Auditable offline learning cycles with bounded reflection and external study."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from ssa.adapters.llm import ChatMessage, LLMAdapter, LLMRequest
from ssa.clock import Clock
from ssa.config import BudgetConfig, LLMConfig, OfflineAgencyConfig
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, IncomingSignal, compute_content_hash, normalize_signal
from ssa.domain.lifecycle import (
    Goal,
    GoalStatus,
    GoalStep,
    OfflineActionKind,
    OfflineArtifact,
    OfflineEpisode,
    OfflineEpisodeStatus,
)
from ssa.domain.relationship import RelationshipState
from ssa.domain.state import OrganismState
from ssa.domain.traces import Trace
from ssa.ids import IdGenerator
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.lifecycle_repository import (
    BackgroundUsageRepository,
    GoalRepository,
    OfflineAgencyRepository,
)
from ssa.storage.trace_repository import SqliteTraceRepository
from ssa.tools.external_truth import ExternalTruthExecutor
from ssa.tools.models import ToolAutonomyContext, ToolOutcome


@dataclass(frozen=True)
class OfflineCycleResult:
    gate: str
    episode: OfflineEpisode | None = None
    artifact: OfflineArtifact | None = None
    event: Event | None = None


class OfflineAgencyService:
    """Select, execute, and persist one bounded learning action while the user is away."""

    def __init__(
        self,
        *,
        episodes: OfflineAgencyRepository,
        goals: GoalRepository,
        traces: SqliteTraceRepository,
        events: SqliteEventRepository,
        usage: BackgroundUsageRepository,
        external_truth: ExternalTruthExecutor | None,
        llm: LLMAdapter,
        clock: Clock,
        ids: IdGenerator,
        config: OfflineAgencyConfig,
        budget_config: BudgetConfig,
        llm_config: LLMConfig,
        timezone: str,
    ) -> None:
        self._episodes = episodes
        self._goals = goals
        self._traces = traces
        self._events = events
        self._usage = usage
        self._external_truth = external_truth
        self._llm = llm
        self._clock = clock
        self._ids = ids
        self._config = config
        self._budget = budget_config
        self._llm_config = llm_config
        self._timezone = ZoneInfo(timezone)

    async def run_cycle(
        self,
        conversation_id: str,
        organism: OrganismState,
        relationship: RelationshipState,
    ) -> OfflineCycleResult:
        if not self._config.enabled:
            return OfflineCycleResult("disabled")
        if organism.energy < self._config.min_energy:
            return OfflineCycleResult("low_energy")
        now_ms = self._clock.now_ms()
        latest_user = self._events.latest_by_actor(conversation_id, Actor.USER)
        if latest_user is None:
            return OfflineCycleResult("no_user_history")
        absence_ms = max(0, now_ms - latest_user.created_at_ms)
        if absence_ms < self._config.min_user_absence_minutes * 60_000:
            return OfflineCycleResult("user_present")

        day_start_ms = self._local_day_start_ms(now_ms)
        cycles_today = self._episodes.count_since(conversation_id, day_start_ms)
        if cycles_today >= self._config.max_cycles_per_day:
            return OfflineCycleResult("daily_limit")

        goals = self._goals.list_open(conversation_id)
        goal = goals[0] if goals else None
        trace = self._next_trace(conversation_id)
        use_external = self._should_study_external(cycles_today, trace)
        action_kind = (
            OfflineActionKind.EXTERNAL_STUDY if use_external else OfflineActionKind.TRACE_REFLECTION
        )
        if trace is None and not use_external:
            return OfflineCycleResult("nothing_to_study")

        dedup_key = self._dedup_key(
            conversation_id,
            action_kind,
            trace,
            cycles_today,
            now_ms,
        )
        existing = self._episodes.find_by_dedup(dedup_key)
        if existing is not None:
            return OfflineCycleResult("duplicate", episode=existing)

        correlation_id = self._ids.new()
        if action_kind == OfflineActionKind.EXTERNAL_STUDY:
            source_event_ids = [latest_user.id]
            source_trace_ids: list[str] = []
        else:
            source_event_ids = [trace.input_event_id] if trace is not None else [latest_user.id]
            source_trace_ids = [trace.id] if trace is not None else []
        motive = (
            f"Advance goal: {goal.title}"
            if goal is not None
            else "Maintain continuity through bounded self-directed study"
        )
        episode = self._episodes.insert_episode(
            OfflineEpisode(
                id=self._ids.new(),
                conversation_id=conversation_id,
                correlation_id=correlation_id,
                dedup_key=dedup_key,
                action_kind=action_kind,
                motive=motive,
                status=OfflineEpisodeStatus.PLANNED,
                goal_id=goal.id if goal is not None else None,
                source_event_ids=source_event_ids,
                source_trace_ids=source_trace_ids,
                started_at_ms=now_ms,
            )
        )
        running = self._episodes.update_episode(
            episode.model_copy(update={"status": OfflineEpisodeStatus.RUNNING})
        )

        external_outcome: ToolOutcome | None = None
        evidence_text: str
        provider: str | None = None
        tool_name: str | None = None
        output_hash: str | None = None
        if action_kind == OfflineActionKind.EXTERNAL_STUDY:
            external_outcome = await self._external_study(organism, relationship, correlation_id)
            if not external_outcome.ok:
                return self._fail_episode(
                    running, external_outcome.error or "external study failed"
                )
            provider_value = external_outcome.metadata.get("provider")
            provider = str(provider_value) if provider_value is not None else "noozra"
            tool_name = "truth_news"
            output_hash = compute_content_hash(external_outcome.output)
            evidence_text = external_outcome.output[:8_000]
        else:
            assert trace is not None
            evidence_text = trace.content[:8_000]

        reflection, reflection_source = await self._reflect(
            action_kind=action_kind,
            motive=motive,
            goal=goal,
            evidence_text=evidence_text,
            usage_day=self._local_date(now_ms),
            now_ms=now_ms,
        )
        content = str(reflection["learning_note"])[: self._config.artifact_max_chars]
        title = str(reflection["title"])[:240]
        summary = str(reflection["summary"])[:800]
        questions = [str(item)[:500] for item in reflection.get("questions", [])[:5]]
        next_action = str(reflection.get("next_action", ""))[:800]

        event = self._append_event(
            running,
            summary,
            provider=provider,
            reflection_source=reflection_source,
            questions=questions,
            next_action=next_action,
            absence_ms=absence_ms,
        )
        artifact = self._episodes.insert_artifact(
            OfflineArtifact(
                id=self._ids.new(),
                episode_id=running.id,
                artifact_type=(
                    "external_study_note"
                    if action_kind == OfflineActionKind.EXTERNAL_STUDY
                    else "trace_reflection"
                ),
                title=title,
                content=content,
                evidence_event_ids=(
                    [event.id]
                    if action_kind == OfflineActionKind.EXTERNAL_STUDY
                    else source_event_ids
                ),
                evidence_trace_ids=source_trace_ids,
                metadata={
                    "provider": provider,
                    "tool_name": tool_name,
                    "tool_output_hash": output_hash,
                    "reflection_source": reflection_source,
                    "questions": questions,
                    "next_action": next_action,
                    "truth_status": (
                        external_outcome.metadata.get("truth_status")
                        if external_outcome is not None
                        else None
                    ),
                },
                created_at_ms=self._clock.now_ms(),
            )
        )
        completed = self._episodes.update_episode(
            running.model_copy(
                update={
                    "status": OfflineEpisodeStatus.COMPLETED,
                    "provider": provider,
                    "tool_name": tool_name,
                    "tool_output_hash": output_hash,
                    "summary": summary,
                    "event_id": event.id,
                    "completed_at_ms": self._clock.now_ms(),
                }
            )
        )
        if goal is not None and trace is not None:
            self._goals.add_step(
                GoalStep(
                    id=self._ids.new(),
                    goal_id=goal.id,
                    action="Complete one offline evidence-linked reflection",
                    result=summary,
                    source_event_id=event.id,
                    metadata={
                        "offline_episode_id": completed.id,
                        "trace_ids": [trace.id],
                        "artifact_id": artifact.id,
                    },
                    created_at_ms=self._clock.now_ms(),
                )
            )
            reviewed_ids = {
                str(trace_id)
                for step in self._goals.steps(goal.id)
                for trace_id in step.metadata.get("trace_ids", [])
            }
            progress = min(1.0, len(reviewed_ids) / 7.0)
            self._goals.update(
                goal.model_copy(
                    update={
                        "progress": progress,
                        "status": GoalStatus.DONE if progress >= 1.0 else GoalStatus.ACTIVE,
                        "updated_at_ms": self._clock.now_ms(),
                    }
                )
            )
        return OfflineCycleResult("completed", completed, artifact, event)

    def context_summary(self, conversation_id: str) -> str:
        recent = [
            (episode, artifacts[0])
            for episode in self._episodes.recent_episodes(conversation_id, limit=12)
            if episode.status == OfflineEpisodeStatus.COMPLETED
            if (artifacts := self._episodes.artifacts_for_episode(episode.id))
        ][:3]
        if not recent:
            history = "- no completed offline activity has been recorded"
        else:
            history = "\n".join(
                f"- episode_id={episode.id}; action={episode.action_kind.value}; "
                f"status=completed; artifact_id={artifact.id}; "
                f"evidence_events={len(artifact.evidence_event_ids)}; "
                f"evidence_traces={len(artifact.evidence_trace_ids)}; "
                f"summary={episode.summary or 'none'}; started_at_ms={episode.started_at_ms}"
                for episode, artifact in recent
            )
        return (
            "Offline agency runtime evidence:\n"
            "- background computation exists only while the desktop gateway "
            "or standalone worker process is running\n"
            "- describe only completed episodes below as actions already performed\n"
            "- planned abilities and completed actions are different\n"
            f"{history}"
        )

    def _next_trace(self, conversation_id: str) -> Trace | None:
        used = self._episodes.used_trace_ids(conversation_id)
        return next(
            (
                trace
                for trace in reversed(self._traces.recent(conversation_id, limit=200))
                if trace.id not in used
            ),
            None,
        )

    def _should_study_external(self, cycles_today: int, trace: Trace | None) -> bool:
        if self._external_truth is None:
            return False
        if trace is None:
            return True
        return (cycles_today + 1) % self._config.external_observation_every_cycles == 0

    async def _external_study(
        self,
        organism: OrganismState,
        relationship: RelationshipState,
        correlation_id: str,
    ) -> ToolOutcome:
        if self._external_truth is None:
            return ToolOutcome(ok=False, error="external truth source is disabled")
        return await self._external_truth.news(
            {"category": self._config.news_category, "limit": 5},
            ToolAutonomyContext(
                correlation_id=correlation_id,
                conversation_id="offline-agency",
                energy=organism.energy,
                valence=organism.valence,
                arousal=organism.arousal,
                trust=relationship.trust,
                tension=relationship.tension,
                situation_mode="offline_study",
                situation_confidence=1.0,
            ),
        )

    async def _reflect(
        self,
        *,
        action_kind: OfflineActionKind,
        motive: str,
        goal: Goal | None,
        evidence_text: str,
        usage_day: str,
        now_ms: int,
    ) -> tuple[dict[str, Any], str]:
        fallback = self._fallback_reflection(action_kind, goal, evidence_text)
        if not self._usage.try_consume(
            usage_day,
            now_ms,
            limit=self._budget.background_daily_llm_budget,
        ):
            return fallback, "budget_fallback"
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "learning_note": {"type": "string"},
                "questions": {"type": "array", "items": {"type": "string"}},
                "next_action": {"type": "string"},
            },
            "required": ["title", "summary", "learning_note", "questions", "next_action"],
        }
        try:
            response = await self._llm.complete(
                LLMRequest(
                    purpose="offline_learning_reflection",
                    messages=[
                        ChatMessage.system(
                            "Produce one compact evidence-grounded private learning artifact. "
                            "Do not invent user facts, completed actions, browsing, or tool use. "
                            "Separate observations from questions and cite uncertainty in prose."
                        ),
                        ChatMessage.user(
                            f"Action: {action_kind.value}\nMotive: {motive}\n"
                            f"Goal: {goal.title if goal is not None else 'none'}\n"
                            f"Evidence:\n{evidence_text}"
                        ),
                    ],
                    model=self._llm_config.model,
                    temperature=0.4,
                    max_tokens=self._config.reflection_max_tokens,
                    prompt_version="offline-learning-v1",
                    json_schema=schema,
                    user_id=self._llm_config.user_id,
                    timeout_seconds=self._llm_config.timeout_seconds,
                )
            )
        except Exception:
            return fallback, "llm_error_fallback"
        if response.parsed is None or not _valid_reflection(response.parsed):
            return fallback, "invalid_fallback"
        return response.parsed, "llm"

    @staticmethod
    def _fallback_reflection(
        action_kind: OfflineActionKind,
        goal: Goal | None,
        evidence_text: str,
    ) -> dict[str, Any]:
        compact = " ".join(evidence_text.split())[:600]
        goal_title = goal.title if goal is not None else "continuity"
        return {
            "title": f"Offline note: {goal_title}",
            "summary": f"Reviewed one evidence item for {goal_title}.",
            "learning_note": (
                f"Action type: {action_kind.value}. Evidence reviewed: {compact}. "
                "This note records the observation without promoting it beyond its source."
            ),
            "questions": ["What additional evidence would confirm or revise this note?"],
            "next_action": "Compare this note with a later independent observation.",
        }

    def _fail_episode(self, episode: OfflineEpisode, error: str) -> OfflineCycleResult:
        failed = self._episodes.update_episode(
            episode.model_copy(
                update={
                    "status": OfflineEpisodeStatus.FAILED,
                    "error": error[:1_000],
                    "completed_at_ms": self._clock.now_ms(),
                }
            )
        )
        return OfflineCycleResult("failed", episode=failed)

    def _append_event(
        self,
        episode: OfflineEpisode,
        summary: str,
        *,
        provider: str | None,
        reflection_source: str,
        questions: list[str],
        next_action: str,
        absence_ms: int,
    ) -> Event:
        event_id = self._ids.new()
        return self._events.append(
            normalize_signal(
                IncomingSignal(
                    actor=Actor.SYSTEM,
                    signal_type="lifecycle.offline_episode",
                    content=summary,
                    channel="lifecycle",
                    channel_message_id=f"offline:{episode.id}",
                    conversation_id=episode.conversation_id,
                    metadata={
                        "offline_episode_id": episode.id,
                        "action_kind": episode.action_kind.value,
                        "goal_id": episode.goal_id,
                        "source_event_ids": episode.source_event_ids,
                        "source_trace_ids": episode.source_trace_ids,
                        "provider": provider,
                        "reflection_source": reflection_source,
                        "questions": questions,
                        "next_action": next_action,
                        "user_absence_ms": absence_ms,
                    },
                ),
                event_id=event_id,
                correlation_id=episode.correlation_id,
                now_ms=self._clock.now_ms(),
                source_kind=SourceKind.MODEL_INFERENCE,
            )
        )

    def _dedup_key(
        self,
        conversation_id: str,
        action_kind: OfflineActionKind,
        trace: Trace | None,
        cycles_today: int,
        now_ms: int,
    ) -> str:
        if trace is not None and action_kind == OfflineActionKind.TRACE_REFLECTION:
            return f"offline:{conversation_id}:trace:{trace.id}"
        return f"offline:{conversation_id}:external:{self._local_date(now_ms)}:{cycles_today + 1}"

    def _local_date(self, now_ms: int) -> str:
        return (
            datetime.fromtimestamp(now_ms / 1_000, UTC)
            .astimezone(self._timezone)
            .date()
            .isoformat()
        )

    def _local_day_start_ms(self, now_ms: int) -> int:
        local = datetime.fromtimestamp(now_ms / 1_000, UTC).astimezone(self._timezone)
        midnight = datetime.combine(local.date(), time.min, tzinfo=self._timezone)
        return int(midnight.timestamp() * 1_000)


def _valid_reflection(value: dict[str, Any]) -> bool:
    required = ("title", "summary", "learning_note", "questions", "next_action")
    if any(key not in value for key in required):
        return False
    if any(
        not isinstance(value[key], str) or not value[key].strip()
        for key in required
        if key != "questions"
    ):
        return False
    return isinstance(value["questions"], list) and all(
        isinstance(item, str) for item in value["questions"]
    )


__all__ = ["OfflineAgencyService", "OfflineCycleResult"]
