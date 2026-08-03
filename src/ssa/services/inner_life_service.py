"""Contextual waiting and bounded persistent inner-state recurrence."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ssa.clock import Clock
from ssa.config import InnerLifeConfig
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, IncomingSignal, normalize_signal
from ssa.domain.lifecycle import InnerLifeMode, InnerLoopState
from ssa.domain.perception import SituationMode, SituationPerception
from ssa.domain.relationship import RelationshipState
from ssa.domain.state import OrganismState
from ssa.ids import IdGenerator
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.lifecycle_repository import InnerLoopRepository

_MINUTE_MS = 60_000

_PAUSE_CUES = (
    "等一下",
    "等会",
    "稍等",
    "待会",
    "晚点",
    "一会回来",
    "忙完",
    "later",
    "be back",
    "hold on",
)
_CLOSURE_CUES = (
    "先这样",
    "不聊了",
    "结束吧",
    "晚安",
    "再见",
    "去睡",
    "bye",
    "good night",
    "that's all",
    "talk later",
)


@dataclass(frozen=True)
class InnerHeartbeatResult:
    state: InnerLoopState
    transition_event: Event | None = None


class InnerLifeService:
    """Maintain a compact latent state without generating private monologue text."""

    def __init__(
        self,
        *,
        states: InnerLoopRepository,
        events: SqliteEventRepository,
        clock: Clock,
        ids: IdGenerator,
        config: InnerLifeConfig,
    ) -> None:
        self._states = states
        self._events = events
        self._clock = clock
        self._ids = ids
        self._config = config

    def observe_user_event(self, event: Event) -> InnerLoopState:
        prior = self._states.get(event.conversation_id)
        now_ms = self._clock.now_ms()
        state = InnerLoopState(
            conversation_id=event.conversation_id,
            version=(prior.version + 1 if prior is not None else 1),
            mode=InnerLifeMode.ENGAGED,
            reply_expectation=0.0,
            concern=_clip((prior.concern if prior is not None else 0.0) * 0.35),
            curiosity=_clip((prior.curiosity if prior is not None else 0.5) * 0.70),
            connection_pressure=_clip(
                (prior.connection_pressure if prior is not None else 0.5) * 0.50
            ),
            uncertainty=0.20,
            offline_readiness=0.0,
            last_user_event_id=event.id,
            last_agent_event_id=prior.last_agent_event_id if prior is not None else None,
            last_heartbeat_at_ms=now_ms,
            transition_reason="user_returned" if prior is not None else "user_input",
            updated_at_ms=now_ms,
        )
        return self._states.upsert(state)

    def observe_agent_event(
        self,
        agent_event: Event,
        organism: OrganismState,
        relationship: RelationshipState,
        *,
        user_event: Event | None = None,
        perception: SituationPerception | None = None,
    ) -> InnerLoopState:
        prior = self._states.get(agent_event.conversation_id)
        now_ms = self._clock.now_ms()
        user = user_event or self._events.latest_by_actor(agent_event.conversation_id, Actor.USER)
        user_text = user.content if user is not None else ""
        agent_text = agent_event.content
        question = 1.0 if agent_text.rstrip().endswith(("?", "\uff1f")) else 0.0
        pause = _has_cue(user_text, _PAUSE_CUES)
        closure = max(
            _has_cue(user_text, _CLOSURE_CUES),
            _has_cue(agent_text, _CLOSURE_CUES),
        )
        affect = perception.affect.intensity if perception is not None else 0.20
        negative = perception.affect.negative if perception is not None else 0.0
        uncertainty = perception.affect.uncertainty if perception is not None else 0.40
        relation_signal = perception.semantic.relationship if perception is not None else 0.0
        support = 0.0
        information_gap = 0.0
        if perception is not None:
            support = max(
                perception.mode_distribution.get(SituationMode.SUPPORT, 0.0),
                perception.mode_distribution.get(SituationMode.REPAIR, 0.0),
            )
            information_gap = perception.semantic.information_gap

        reply_expectation = _sigmoid(
            -0.95
            + 1.65 * question
            + 0.90 * support
            + 0.65 * affect
            + 0.55 * relation_signal
            + 0.50 * uncertainty
            + 0.85 * pause
            - 1.60 * closure
        )
        wait_multiplier = math.exp(
            1.10 * reply_expectation
            + 0.70 * affect
            + 0.45 * relationship.tension
            + 1.00 * pause
            - 1.20 * closure
        )
        wait_minutes = round(
            _clip_range(
                self._config.base_wait_minutes * wait_multiplier,
                self._config.min_wait_minutes,
                self._config.max_wait_minutes,
            )
        )
        concern = _clip(
            0.35 * negative
            + 0.25 * uncertainty
            + 0.20 * relationship.tension
            + 0.20 * reply_expectation
        )
        curiosity = _clip(
            0.45 * organism.curiosity + 0.35 * information_gap + 0.20 * reply_expectation
        )
        connection_pressure = _clip(
            0.45 * organism.connection_need + 0.30 * relationship.closeness + 0.25 * relation_signal
        )
        state = InnerLoopState(
            conversation_id=agent_event.conversation_id,
            version=(prior.version + 1 if prior is not None else 1),
            mode=InnerLifeMode.WAITING,
            reply_expectation=reply_expectation,
            concern=concern,
            curiosity=curiosity,
            connection_pressure=connection_pressure,
            uncertainty=uncertainty,
            offline_readiness=0.0,
            wait_started_at_ms=agent_event.created_at_ms,
            wait_deadline_at_ms=agent_event.created_at_ms + wait_minutes * _MINUTE_MS,
            last_user_event_id=user.id if user is not None else None,
            last_agent_event_id=agent_event.id,
            last_heartbeat_at_ms=now_ms,
            transition_reason=(
                f"context_wait:{wait_minutes}m:q={reply_expectation:.3f}:"
                f"pause={pause:.0f}:closure={closure:.0f}"
            ),
            updated_at_ms=now_ms,
        )
        return self._states.upsert(state)

    def heartbeat(
        self,
        conversation_id: str,
        organism: OrganismState,
        relationship: RelationshipState,
    ) -> InnerHeartbeatResult:
        prior = self._states.get(conversation_id)
        latest_user = self._events.latest_by_actor(conversation_id, Actor.USER)
        latest_agent = self._events.latest_by_actor(conversation_id, Actor.AGENT)
        if latest_user is None:
            state = self._quiet_initial(conversation_id, prior)
            return InnerHeartbeatResult(state)
        if prior is None or latest_user.id != prior.last_user_event_id:
            if latest_agent is not None and self._agent_follows(latest_agent, latest_user):
                state = self.observe_agent_event(
                    latest_agent,
                    organism,
                    relationship,
                    user_event=latest_user,
                )
            else:
                state = self.observe_user_event(latest_user)
            return InnerHeartbeatResult(state)
        if latest_agent is not None and latest_agent.id != prior.last_agent_event_id:
            state = self.observe_agent_event(
                latest_agent,
                organism,
                relationship,
                user_event=latest_user,
            )
            return InnerHeartbeatResult(state)
        if prior.mode == InnerLifeMode.ENGAGED or prior.wait_deadline_at_ms is None:
            state = self._refresh_engaged(prior)
            return InnerHeartbeatResult(state)

        state = self._advance_waiting(prior, organism, relationship)
        transition = None
        if state.mode != prior.mode:
            transition = self._append_transition(prior, state)
        return InnerHeartbeatResult(self._states.upsert(state), transition)

    def allows_offline_action(self, conversation_id: str) -> bool:
        state = self._states.get(conversation_id)
        return state is not None and state.mode == InnerLifeMode.OFFLINE

    def allows_proactive_contact(self, conversation_id: str) -> bool:
        state = self._states.get(conversation_id)
        return state is not None and state.mode == InnerLifeMode.OFFLINE

    def context_summary(self, conversation_id: str) -> str:
        state = self._states.get(conversation_id)
        if state is None:
            return "Inner-loop state: not initialized; do not infer waiting or offline activity."
        deadline = state.wait_deadline_at_ms if state.wait_deadline_at_ms is not None else "none"
        return (
            "Inner-loop control state (private computed posture, not a user fact or free-form "
            "chain of thought):\n"
            f"- mode={state.mode.value}; reply_expectation={state.reply_expectation:.3f}; "
            f"concern={state.concern:.3f}; curiosity={state.curiosity:.3f}; "
            f"connection_pressure={state.connection_pressure:.3f}; "
            f"uncertainty={state.uncertainty:.3f}; "
            f"offline_readiness={state.offline_readiness:.3f}\n"
            f"- wait_deadline_at_ms={deadline}; last_heartbeat_at_ms="
            f"{state.last_heartbeat_at_ms}; reason={state.transition_reason}\n"
            "- Use this only as a soft behavioral control. Silence has multiple plausible causes."
        )

    def _advance_waiting(
        self,
        prior: InnerLoopState,
        organism: OrganismState,
        relationship: RelationshipState,
    ) -> InnerLoopState:
        now_ms = self._clock.now_ms()
        deadline = prior.wait_deadline_at_ms or now_ms
        started = prior.wait_started_at_ms or prior.updated_at_ms
        observed_gap_ms = max(1, now_ms - prior.last_heartbeat_at_ms)
        expected_heartbeat_ms = self._config.heartbeat_seconds * 1_000
        runtime_resumed = observed_gap_ms > expected_heartbeat_ms * 3
        dt_ms = min(observed_gap_ms, expected_heartbeat_ms * 2)
        wait_span = max(_MINUTE_MS, deadline - started)
        overdue_ms = max(0, now_ms - deadline)
        incremental_overdue_ms = max(0, now_ms - max(deadline, prior.last_heartbeat_at_ms))
        reply_decay = 2.0 ** (
            -(incremental_overdue_ms / _MINUTE_MS) / self._config.reply_half_life_minutes
        )
        reply = _clip(prior.reply_expectation * reply_decay)
        progress = _clip((now_ms - started) / wait_span)
        overdue_pressure = 1.0 - math.exp(-overdue_ms / wait_span)
        silence_surprise = _clip(reply * (1.0 - math.exp(-2.0 * progress)))
        affect_decay = 2.0 ** (-(dt_ms / _MINUTE_MS) / self._config.affect_half_life_minutes)
        target_concern = _clip(
            0.40 * prior.uncertainty * silence_surprise
            + 0.25 * relationship.tension
            + 0.20 * prior.connection_pressure
        )
        concern = _clip(affect_decay * prior.concern + (1.0 - affect_decay) * target_concern)
        curiosity = _clip(
            affect_decay * prior.curiosity
            + (1.0 - affect_decay) * (0.55 * organism.curiosity + 0.25 * silence_surprise)
        )
        elapsed_hours = dt_ms / 3_600_000
        connection = _clip(
            prior.connection_pressure
            + elapsed_hours * 0.025 * (0.40 + 0.60 * relationship.closeness)
        )
        uncertainty = _clip(
            affect_decay * prior.uncertainty
            + (1.0 - affect_decay) * (0.30 + 0.40 * silence_surprise)
        )
        readiness = _sigmoid(
            -0.55
            + 1.90 * overdue_pressure
            + 1.20 * (1.0 - reply)
            + 0.55 * organism.energy
            + 0.35 * organism.autonomy_need
            - 1.25 * concern
        )
        if organism.energy < self._config.min_offline_energy:
            mode = InnerLifeMode.QUIET_REST
            reason = "low_energy_rest"
        elif (
            prior.mode == InnerLifeMode.OFFLINE and readiness >= self._config.offline_exit_threshold
        ):
            mode = InnerLifeMode.OFFLINE
            reason = "offline_hysteresis_hold"
        elif now_ms >= deadline and readiness >= self._config.offline_entry_threshold:
            mode = InnerLifeMode.OFFLINE
            reason = "context_wait_elapsed"
        else:
            mode = InnerLifeMode.WAITING
            reason = "reply_window_open" if now_ms < deadline else "high_reply_expectation"
        if runtime_resumed:
            reason = f"runtime_resumed:{observed_gap_ms}ms:{reason}"
        return prior.model_copy(
            update={
                "version": prior.version + 1,
                "mode": mode,
                "reply_expectation": reply,
                "concern": concern,
                "curiosity": curiosity,
                "connection_pressure": connection,
                "uncertainty": uncertainty,
                "offline_readiness": readiness,
                "last_heartbeat_at_ms": now_ms,
                "transition_reason": reason,
                "updated_at_ms": now_ms,
            }
        )

    def _quiet_initial(
        self,
        conversation_id: str,
        prior: InnerLoopState | None,
    ) -> InnerLoopState:
        now_ms = self._clock.now_ms()
        return self._states.upsert(
            InnerLoopState(
                conversation_id=conversation_id,
                version=(prior.version + 1 if prior is not None else 1),
                mode=InnerLifeMode.QUIET_REST,
                reply_expectation=0.0,
                concern=0.0,
                curiosity=prior.curiosity if prior is not None else 0.5,
                connection_pressure=prior.connection_pressure if prior is not None else 0.5,
                uncertainty=0.0,
                offline_readiness=0.0,
                last_heartbeat_at_ms=now_ms,
                transition_reason="no_user_history",
                updated_at_ms=now_ms,
            )
        )

    def _refresh_engaged(self, prior: InnerLoopState) -> InnerLoopState:
        now_ms = self._clock.now_ms()
        return self._states.upsert(
            prior.model_copy(
                update={
                    "version": prior.version + 1,
                    "last_heartbeat_at_ms": now_ms,
                    "transition_reason": "conversation_open",
                    "updated_at_ms": now_ms,
                }
            )
        )

    def _append_transition(self, prior: InnerLoopState, state: InnerLoopState) -> Event:
        event_id = self._ids.new()
        return self._events.append(
            normalize_signal(
                IncomingSignal(
                    actor=Actor.SYSTEM,
                    signal_type="lifecycle.inner_transition",
                    content=f"Inner lifecycle moved from {prior.mode.value} to {state.mode.value}.",
                    channel="lifecycle",
                    channel_message_id=(
                        f"inner:{state.conversation_id}:{state.version}:{state.mode.value}"
                    ),
                    conversation_id=state.conversation_id,
                    metadata={
                        "from_mode": prior.mode.value,
                        "to_mode": state.mode.value,
                        "reply_expectation": state.reply_expectation,
                        "concern": state.concern,
                        "offline_readiness": state.offline_readiness,
                        "wait_deadline_at_ms": state.wait_deadline_at_ms,
                        "reason": state.transition_reason,
                    },
                ),
                event_id=event_id,
                correlation_id=self._ids.new(),
                now_ms=self._clock.now_ms(),
                source_kind=SourceKind.SYSTEM_DERIVED,
            )
        )

    @staticmethod
    def _agent_follows(agent: Event, user: Event) -> bool:
        return agent.parent_event_id == user.id or agent.created_at_ms > user.created_at_ms


def _has_cue(content: str, cues: tuple[str, ...]) -> float:
    normalized = content.casefold()
    return 1.0 if any(cue in normalized for cue in cues) else 0.0


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clip_range(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


__all__ = ["InnerHeartbeatResult", "InnerLifeService"]
