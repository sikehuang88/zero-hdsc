"""Deterministic initiative approval with optional LLM message drafting."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

from ssa.adapters.llm import ChatMessage, LLMAdapter, LLMRequest
from ssa.clock import Clock
from ssa.config import BudgetConfig, GoalConfig, InitiativeConfig, LLMConfig
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, IncomingSignal, normalize_signal
from ssa.domain.lifecycle import (
    CausalTraceRecord,
    EmotionEpisode,
    Goal,
    Initiative,
    InitiativeStatus,
    OutboxMessage,
)
from ssa.domain.relationship import RelationshipState
from ssa.domain.state import OrganismState
from ssa.ids import IdGenerator
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

_DAY_MS = 86_400_000
_HOUR_MS = 3_600_000


@dataclass(frozen=True)
class InitiativeDecision:
    approved: bool
    gate: str
    initiative: Initiative | None = None
    outbox: OutboxMessage | None = None


class InitiativeService:
    """Generate grounded initiative candidates and apply deterministic gates."""

    def __init__(
        self,
        *,
        initiatives: InitiativeRepository,
        outbox: OutboxRepository,
        emotions: EmotionEpisodeRepository,
        goals: GoalRepository,
        traces: SqliteTraceRepository,
        events: SqliteEventRepository,
        causal_traces: CausalTraceRepository,
        usage: BackgroundUsageRepository,
        clock: Clock,
        ids: IdGenerator,
        initiative_config: InitiativeConfig,
        budget_config: BudgetConfig,
        goal_config: GoalConfig,
        llm_config: LLMConfig,
        timezone: str,
        llm: LLMAdapter | None = None,
    ) -> None:
        self._initiatives = initiatives
        self._outbox = outbox
        self._emotions = emotions
        self._goals = goals
        self._traces = traces
        self._events = events
        self._causal_traces = causal_traces
        self._usage = usage
        self._clock = clock
        self._ids = ids
        self._config = initiative_config
        self._budget = budget_config
        self._goal_config = goal_config
        self._llm_config = llm_config
        self._timezone = ZoneInfo(timezone)
        self._llm = llm

    async def evaluate(
        self,
        conversation_id: str,
        organism: OrganismState,
        relationship: RelationshipState,
    ) -> InitiativeDecision:
        now_ms = self._clock.now_ms()
        recent_events = self._events.recent_by_conversation(conversation_id, limit=80)
        self._emotions.expire_decayed(conversation_id, now_ms)
        active_emotions = self._emotions.active(conversation_id)
        active_goals = self._goals.list_active(conversation_id)
        activations = self._traces.latest_activations(conversation_id)
        source_trace_ids = [item.trace.id for item in activations[:6]]
        source_event_ids = self._source_event_ids(
            recent_events,
            active_emotions,
            source_trace_ids,
        )

        if not source_event_ids:
            return InitiativeDecision(False, "missing_evidence")
        if self._is_quiet(now_ms):
            return InitiativeDecision(False, "quiet_hours")

        day_start_ms = self._local_day_start_ms(now_ms)
        if (
            self._initiatives.count_sent_since(conversation_id, day_start_ms)
            >= self._config.daily_limit
        ):
            return InitiativeDecision(False, "daily_limit")

        latest_sent = self._initiatives.latest_sent(conversation_id)
        cooldown_ms = self._config.cooldown_minutes * 60_000
        if latest_sent is not None and now_ms - latest_sent.updated_at_ms < cooldown_ms:
            return InitiativeDecision(False, "cooldown")

        latest_user = self._events.latest_by_actor(conversation_id, Actor.USER)
        if latest_user is None:
            return InitiativeDecision(False, "no_user_history")
        user_gap_ms = max(0, now_ms - latest_user.created_at_ms)
        if user_gap_ms < self._config.min_gap_hours * _HOUR_MS:
            return InitiativeDecision(False, "minimum_user_gap")

        goal, goal_urgency = self._leading_goal(active_goals, organism, now_ms)
        emotion, emotion_signal = self._leading_emotion(active_emotions)
        motive, intent, topic = self._motive(goal, goal_urgency, emotion, emotion_signal, organism)
        if motive is None:
            return InitiativeDecision(False, "missing_motive")

        dedup_key = self._dedup_key(
            conversation_id,
            motive,
            topic,
            source_event_ids,
            now_ms,
        )
        duplicate = self._initiatives.find_by_dedup(dedup_key)
        if duplicate is not None:
            return InitiativeDecision(False, "duplicate", initiative=duplicate)

        components = self._score_components(
            organism=organism,
            relationship=relationship,
            emotion_signal=emotion_signal,
            goal_urgency=goal_urgency,
            user_gap_ms=user_gap_ms,
            active_count=len(self._initiatives.list_active(conversation_id)),
        )
        decision_score = self._decision_score(components)
        urgency = max(emotion_signal, goal_urgency, organism.connection_need)
        if urgency < self._config.min_urgency or decision_score < 0.25:
            return InitiativeDecision(False, "score_below_threshold")

        draft, draft_source = await self._draft(
            motive=motive,
            intent=intent,
            topic=topic,
            evidence=[event.content for event in recent_events if event.id in source_event_ids],
            usage_day=self._local_date(now_ms),
            now_ms=now_ms,
        )
        correlation_id = self._ids.new()
        decision = {
            "gate": "approved",
            "components": components,
            "thresholds": {
                "min_urgency": self._config.min_urgency,
                "min_decision_score": 0.25,
            },
            "draft_source": draft_source,
            "source_event_ids": source_event_ids,
            "source_trace_ids": source_trace_ids,
            "topic": topic,
        }
        candidate = self._initiatives.insert(
            Initiative(
                id=self._ids.new(),
                conversation_id=conversation_id,
                motive=motive,
                intent=intent,
                content_draft=draft,
                urgency=urgency,
                decision_score=decision_score,
                earliest_send_at_ms=now_ms,
                expires_at_ms=now_ms + 12 * _HOUR_MS,
                correlation_id=correlation_id,
                goal_id=goal.id if goal is not None else None,
                source_event_ids=source_event_ids,
                source_trace_ids=source_trace_ids,
                decision=decision,
                channel="local",
                dedup_key=dedup_key,
                created_at_ms=now_ms,
                updated_at_ms=now_ms,
            )
        )
        approved = self._initiatives.update_status(
            candidate.id,
            InitiativeStatus.APPROVED,
            now_ms,
        )
        queued_message = self._outbox.enqueue(
            OutboxMessage(
                id=self._ids.new(),
                correlation_id=correlation_id,
                channel=approved.channel,
                recipient=conversation_id,
                payload={
                    "content": approved.content_draft,
                    "intent": approved.intent,
                    "motive": approved.motive,
                },
                next_attempt_at_ms=approved.earliest_send_at_ms,
                created_at_ms=now_ms,
                initiative_id=approved.id,
                dedup_key=f"outbox:{approved.id}",
            )
        )
        queued = self._initiatives.update_status(
            approved.id,
            InitiativeStatus.QUEUED,
            now_ms,
        )
        audit_event = self._append_decision_event(queued, components)
        self._causal_traces.insert(
            CausalTraceRecord(
                id=self._ids.new(),
                correlation_id=correlation_id,
                input_event_id=source_event_ids[0],
                goal_ids=[goal.id] if goal is not None else [],
                decision=decision,
                output_event_id=audit_event.id,
                created_at_ms=now_ms,
            )
        )
        return InitiativeDecision(True, "approved", queued, queued_message)

    def expire_due(self, conversation_id: str) -> int:
        return self._initiatives.expire_due(conversation_id, self._clock.now_ms())

    def _source_event_ids(
        self,
        recent_events: list[Event],
        emotions: list[EmotionEpisode],
        source_trace_ids: list[str],
    ) -> list[str]:
        valid_event_ids = {event.id for event in recent_events}
        ordered: list[str] = []
        for emotion in emotions:
            ordered.extend(emotion.source_event_ids)
        for trace_id in source_trace_ids:
            trace = self._traces.get(trace_id)
            if trace is not None:
                ordered.append(trace.input_event_id)
        ordered.extend(
            event.id
            for event in reversed(recent_events)
            if event.actor in {Actor.USER, Actor.WORLD}
        )
        return list(dict.fromkeys(event_id for event_id in ordered if event_id in valid_event_ids))[
            :8
        ]

    def _leading_goal(
        self,
        goals: list[Goal],
        organism: OrganismState,
        now_ms: int,
    ) -> tuple[Goal | None, float]:
        ranked = [(goal, self._goal_urgency(goal, organism, now_ms)) for goal in goals]
        return max(ranked, key=lambda item: item[1], default=(None, 0.0))

    def _goal_urgency(self, goal: Goal, organism: OrganismState, now_ms: int) -> float:
        due = 0.0
        if goal.due_at_ms is not None:
            remaining_days = max(0.0, (goal.due_at_ms - now_ms) / _DAY_MS)
            due = max(0.0, 1.0 - remaining_days / 7.0)
        need = max(organism.connection_need, organism.autonomy_need, organism.curiosity)
        continuity = 1.0 - goal.progress
        config = self._goal_config
        return min(
            1.0,
            config.urgency_priority_weight * goal.priority
            + config.urgency_due_weight * due
            + config.urgency_need_weight * need
            + config.urgency_opportunity_weight * organism.energy
            + config.urgency_continuity_weight * continuity,
        )

    @staticmethod
    def _leading_emotion(emotions: list[EmotionEpisode]) -> tuple[EmotionEpisode | None, float]:
        ranked = [(emotion, emotion.intensity * (1.0 - emotion.inhibition)) for emotion in emotions]
        return max(ranked, key=lambda item: item[1], default=(None, 0.0))

    @staticmethod
    def _motive(
        goal: Goal | None,
        goal_urgency: float,
        emotion: EmotionEpisode | None,
        emotion_signal: float,
        organism: OrganismState,
    ) -> tuple[str | None, str, str]:
        if goal is not None and goal_urgency >= max(emotion_signal, organism.connection_need):
            return goal.motive, "project_work", goal.title
        if emotion is not None and emotion_signal >= organism.connection_need:
            return (
                f"{emotion.emotion_type.value}:{emotion.action_tendency}",
                "initiate",
                emotion.target,
            )
        if organism.connection_need >= 0.5:
            return "connection_need", "initiate", "shared continuity"
        return None, "defer", ""

    def _score_components(
        self,
        *,
        organism: OrganismState,
        relationship: RelationshipState,
        emotion_signal: float,
        goal_urgency: float,
        user_gap_ms: int,
        active_count: int,
    ) -> dict[str, float]:
        gap_floor = max(1, self._config.min_gap_hours * _HOUR_MS)
        return {
            "motive": max(emotion_signal, goal_urgency, organism.connection_need),
            "relevance": max(relationship.closeness, emotion_signal),
            "goal_urgency": goal_urgency,
            "connection_need": organism.connection_need,
            "opportunity": organism.energy * organism.safety,
            "intrusion": max(0.0, 1.0 - user_gap_ms / (gap_floor * 2.0)),
            "repetition": min(1.0, active_count / 3.0),
            "uncertainty": min(1.0, (1.0 - relationship.trust) * 0.7 + relationship.tension * 0.3),
            "fatigue": 1.0 - organism.energy,
        }

    @staticmethod
    def _decision_score(components: dict[str, float]) -> float:
        positive = (
            0.32 * components["motive"]
            + 0.12 * components["relevance"]
            + 0.15 * components["goal_urgency"]
            + 0.22 * components["connection_need"]
            + 0.19 * components["opportunity"]
        )
        negative = (
            0.10 * components["intrusion"]
            + 0.10 * components["repetition"]
            + 0.08 * components["uncertainty"]
            + 0.10 * components["fatigue"]
        )
        return max(-1.0, min(1.0, positive - negative))

    async def _draft(
        self,
        *,
        motive: str,
        intent: str,
        topic: str,
        evidence: list[str],
        usage_day: str,
        now_ms: int,
    ) -> tuple[str, str]:
        fallback = self._fallback_draft(intent, topic)
        if self._llm is None:
            return fallback, "deterministic"
        if not self._usage.try_consume(
            usage_day,
            now_ms,
            limit=self._budget.background_daily_llm_budget,
        ):
            return fallback, "budget_fallback"
        try:
            response = await self._llm.complete(
                LLMRequest(
                    purpose="initiative_draft",
                    messages=[
                        ChatMessage.system(
                            "Draft one warm, concise proactive message. Stay grounded in the "
                            "provided evidence. Do not mention internal scores, goals, or memory systems."
                        ),
                        ChatMessage.user(
                            f"Motive: {motive}\nIntent: {intent}\nTopic: {topic}\n"
                            f"Evidence: {evidence[:4]}"
                        ),
                    ],
                    model=self._llm_config.model,
                    temperature=self._llm_config.temperature,
                    top_p=self._llm_config.top_p,
                    max_tokens=min(160, self._llm_config.max_tokens),
                    prompt_version="initiative-draft-v1",
                    user_id=self._llm_config.user_id,
                    timeout_seconds=self._llm_config.timeout_seconds,
                )
            )
        except Exception:
            return fallback, "llm_error_fallback"
        text = response.text.strip()
        return (text[:800], "llm") if text else (fallback, "empty_fallback")

    @staticmethod
    def _fallback_draft(intent: str, topic: str) -> str:
        if intent == "project_work":
            return f"I was thinking about our {topic} project and made a little progress on it."
        return f"You crossed my mind just now, especially our thread about {topic}. How are you?"

    def _append_decision_event(
        self,
        initiative: Initiative,
        components: dict[str, float],
    ) -> Event:
        return self._events.append(
            normalize_signal(
                IncomingSignal(
                    actor=Actor.SYSTEM,
                    signal_type="lifecycle.initiative_decision",
                    content="A grounded proactive initiative passed deterministic delivery gates.",
                    channel="lifecycle",
                    channel_message_id=f"initiative:{initiative.id}",
                    conversation_id=initiative.conversation_id,
                    metadata={
                        "initiative_id": initiative.id,
                        "decision_score": initiative.decision_score,
                        "components": components,
                    },
                ),
                event_id=self._ids.new(),
                correlation_id=initiative.correlation_id or self._ids.new(),
                now_ms=self._clock.now_ms(),
                source_kind=SourceKind.SYSTEM_DERIVED,
            )
        )

    def _is_quiet(self, now_ms: int) -> bool:
        local_time = datetime.fromtimestamp(now_ms / 1000, UTC).astimezone(self._timezone).time()
        start = time.fromisoformat(self._config.quiet_hours_start)
        end = time.fromisoformat(self._config.quiet_hours_end)
        if start == end:
            return False
        if start < end:
            return start <= local_time < end
        return local_time >= start or local_time < end

    def _local_date(self, now_ms: int) -> str:
        return (
            datetime.fromtimestamp(now_ms / 1000, UTC).astimezone(self._timezone).date().isoformat()
        )

    def _local_day_start_ms(self, now_ms: int) -> int:
        local = datetime.fromtimestamp(now_ms / 1000, UTC).astimezone(self._timezone)
        midnight = datetime.combine(local.date(), time.min, tzinfo=self._timezone)
        return int(midnight.timestamp() * 1000)

    def _dedup_key(
        self,
        conversation_id: str,
        motive: str,
        topic: str,
        source_event_ids: list[str],
        now_ms: int,
    ) -> str:
        raw = "|".join(
            [conversation_id, self._local_date(now_ms), motive, topic, *source_event_ids[:3]]
        )
        return f"initiative:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


__all__ = ["InitiativeDecision", "InitiativeService"]
