"""Bounded query-free trace activation and target-specific emotion episodes."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ssa.clock import Clock
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, IncomingSignal, normalize_signal
from ssa.domain.lifecycle import EmotionEpisode, EmotionType, Goal
from ssa.domain.relationship import RelationshipState
from ssa.domain.state import OrganismState
from ssa.domain.traces import ActivatedTrace
from ssa.ids import IdGenerator
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.lifecycle_repository import EmotionEpisodeRepository
from ssa.storage.trace_repository import SqliteTraceRepository


@dataclass(frozen=True)
class RestingStateResult:
    event: Event | None
    activations: tuple[ActivatedTrace, ...]
    emotions: tuple[EmotionEpisode, ...]
    active_mass: float
    null_mass: float


class RestingStateService:
    """Run one bounded spontaneous activation over the directed trace topology."""

    def __init__(
        self,
        *,
        traces: SqliteTraceRepository,
        emotions: EmotionEpisodeRepository,
        events: SqliteEventRepository,
        clock: Clock,
        ids: IdGenerator,
        capacity: int = 8,
        propagation_gain: float = 0.35,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if not 0.0 <= propagation_gain < 1.0:
            raise ValueError("propagation_gain must be in [0, 1)")
        self._traces = traces
        self._emotions = emotions
        self._events = events
        self._clock = clock
        self._ids = ids
        self._capacity = capacity
        self._gain = propagation_gain

    def step(
        self,
        *,
        conversation_id: str,
        organism: OrganismState,
        relationship: RelationshipState,
        goals: list[Goal],
    ) -> RestingStateResult:
        self._emotions.expire_decayed(conversation_id, self._clock.now_ms())
        traces = self._traces.recent(conversation_id, limit=120)
        if not traces:
            return RestingStateResult(None, (), (), 0.0, 1.0)

        now_ms = self._clock.now_ms()
        goal_terms = _goal_terms(goals)
        unresolved_pressure = min(1.0, len(relationship.unresolved_memory_ids) / 3.0)
        scores: dict[str, float] = {}
        traces_by_id = {trace.id: trace for trace in traces}
        for trace in traces:
            freshness = _freshness(trace.created_at_ms, now_ms)
            goal_alignment = _term_overlap(trace.content, goal_terms)
            scores[trace.id] = (
                trace.importance
                * (
                    0.40
                    + 0.20 * trace.arousal
                    + 0.15 * abs(trace.valence)
                    + 0.15 * freshness
                    + 0.10 * goal_alignment
                )
                + 0.08 * unresolved_pressure
            )

        seeds = sorted(scores, key=scores.__getitem__, reverse=True)[: min(3, len(scores))]
        seed_ids = set(seeds)
        propagated = dict(scores)
        for source_id in seeds:
            source_score = scores[source_id]
            for link in self._traces.links_from(source_id):
                if link.target_trace_id not in traces_by_id:
                    continue
                propagated[link.target_trace_id] = propagated.get(link.target_trace_id, 0.0) + (
                    source_score * link.weight * self._gain
                )

        ordered = sorted(propagated, key=propagated.__getitem__, reverse=True)[: self._capacity]
        raw_mass = sum(max(0.0, propagated[trace_id]) for trace_id in ordered)
        scale = 1.0 / raw_mass if raw_mass > 1.0 else 1.0
        activations: list[ActivatedTrace] = []
        for rank, trace_id in enumerate(ordered, 1):
            trace = traces_by_id[trace_id]
            freshness = _freshness(trace.created_at_ms, now_ms)
            score = max(0.0, propagated[trace_id] * scale)
            activations.append(
                ActivatedTrace(
                    trace=trace,
                    rank=rank,
                    score=score,
                    semantic_similarity=_term_overlap(trace.content, goal_terms),
                    freshness=freshness,
                    importance_factor=trace.importance,
                    activation_kind="main" if trace_id in seed_ids else "radiation",
                )
            )

        active_mass = min(1.0, sum(item.score for item in activations))
        correlation_id = self._ids.new()
        event = self._append_event(
            conversation_id,
            correlation_id,
            activations,
            active_mass,
        )
        self._traces.record_activations(
            activation_ids=[self._ids.new() for _ in activations],
            conversation_id=conversation_id,
            correlation_id=correlation_id,
            query_event_id=event.id,
            activations=activations,
            activated_at_ms=now_ms,
        )
        emotion_episodes = self._derive_emotions(
            conversation_id=conversation_id,
            correlation_id=correlation_id,
            event=event,
            activations=activations,
            organism=organism,
            relationship=relationship,
        )
        return RestingStateResult(
            event=event,
            activations=tuple(activations),
            emotions=tuple(emotion_episodes),
            active_mass=active_mass,
            null_mass=1.0 - active_mass,
        )

    def _derive_emotions(
        self,
        *,
        conversation_id: str,
        correlation_id: str,
        event: Event,
        activations: list[ActivatedTrace],
        organism: OrganismState,
        relationship: RelationshipState,
    ) -> list[EmotionEpisode]:
        existing_types = {
            episode.emotion_type for episode in self._emotions.active(conversation_id)
        }
        trace_ids = [item.trace.id for item in activations[:4]]
        created: list[EmotionEpisode] = []
        candidates = [
            (
                EmotionType.ATTACHMENT,
                _clip(0.65 * organism.connection_need + 0.25 * relationship.closeness),
                _clip(0.45 * relationship.tension + 0.25 * (1.0 - organism.safety)),
                "seek contact",
                "owner",
            ),
            (
                EmotionType.CURIOSITY,
                _clip(0.70 * organism.curiosity + 0.15 * activations[0].trace.importance),
                _clip(0.35 * (1.0 - organism.energy)),
                "explore a recalled topic",
                activations[0].trace.id,
            ),
        ]
        if relationship.unresolved_memory_ids or relationship.repair_debt > 0.1:
            candidates.append(
                (
                    EmotionType.CONCERN,
                    _clip(0.55 + 0.35 * relationship.repair_debt),
                    _clip(0.25 + 0.35 * (1.0 - organism.safety)),
                    "check unresolved relationship context",
                    "relationship",
                )
            )

        now_ms = self._clock.now_ms()
        for emotion_type, intensity, inhibition, tendency, target in candidates:
            if emotion_type in existing_types or intensity < 0.35:
                continue
            episode = EmotionEpisode(
                id=self._ids.new(),
                conversation_id=conversation_id,
                correlation_id=correlation_id,
                emotion_type=emotion_type,
                target=target,
                action_tendency=tendency,
                intensity=intensity,
                inhibition=inhibition,
                half_life_minutes=240.0 if emotion_type == EmotionType.ATTACHMENT else 120.0,
                source_event_ids=[event.id],
                source_trace_ids=trace_ids,
                resolution_condition="new user interaction or natural half-life expiry",
                created_at_ms=now_ms,
                updated_at_ms=now_ms,
            )
            created.append(self._emotions.insert(episode))
        return created

    def _append_event(
        self,
        conversation_id: str,
        correlation_id: str,
        activations: list[ActivatedTrace],
        active_mass: float,
    ) -> Event:
        top = activations[0]
        event_id = self._ids.new()
        return self._events.append(
            normalize_signal(
                IncomingSignal(
                    actor=Actor.SYSTEM,
                    signal_type="lifecycle.resting_activation",
                    content=f"A resting-state trace became salient: {top.trace.content[:160]}",
                    channel="lifecycle",
                    channel_message_id=event_id,
                    conversation_id=conversation_id,
                    metadata={
                        "trace_ids": [item.trace.id for item in activations],
                        "active_mass": active_mass,
                        "null_mass": 1.0 - active_mass,
                        "directed": True,
                    },
                ),
                event_id=event_id,
                correlation_id=correlation_id,
                now_ms=self._clock.now_ms(),
                source_kind=SourceKind.MODEL_INFERENCE,
            )
        )


def _freshness(created_at_ms: int, now_ms: int) -> float:
    age_days = max(0.0, (now_ms - created_at_ms) / 86_400_000.0)
    return math.exp(-math.log(2.0) * age_days / 7.0)


def _goal_terms(goals: list[Goal]) -> set[str]:
    terms: set[str] = set()
    for goal in goals:
        terms.update(_tokens(f"{goal.title} {goal.motive}"))
    return terms


def _term_overlap(content: str, terms: set[str]) -> float:
    if not terms:
        return 0.0
    content_terms = _tokens(content)
    return len(content_terms & terms) / max(1, len(terms))


def _tokens(value: str) -> set[str]:
    return {token.strip(".,!?;:()[]{}\"'").casefold() for token in value.split() if token.strip()}


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


__all__ = ["RestingStateResult", "RestingStateService"]
