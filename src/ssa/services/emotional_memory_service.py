"""Capture and retrieve evidence-linked long-term emotional memories."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from math import hypot, sqrt
from statistics import median

from ssa.clock import Clock
from ssa.config import EmotionLibraryConfig
from ssa.domain.appraisal import AppraisalResult
from ssa.domain.emotional_memory import EmotionalMemory
from ssa.domain.events import Event
from ssa.domain.lifecycle import EmotionType
from ssa.domain.traces import ActivatedTrace, Trace
from ssa.hdsc.resonance import (
    RecallState,
    ResonanceConfig,
    resonance_metrics,
    state_metric_tensor,
)
from ssa.ids import IdGenerator
from ssa.storage.emotional_memory_repository import EmotionalMemoryRepository

_DAY_MS = 86_400_000
_ASCII_WORD_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
_CJK_RUN_RE = re.compile(r"[\u3400-\u9fff]+")


@dataclass(frozen=True)
class EmotionalRecall:
    memory: EmotionalMemory
    score: float
    lexical_score: float
    trace_score: float
    recency_score: float
    resonance_amplitude: float = 0.0
    resonance_ratio: float = 0.0
    detuning: float = 0.0
    damping: float = 1.0


class EmotionalMemoryService:
    """Keep subjective feeling history separate from factual memory."""

    def __init__(
        self,
        repository: EmotionalMemoryRepository,
        clock: Clock,
        ids: IdGenerator,
        config: EmotionLibraryConfig,
    ) -> None:
        self._repository = repository
        self._clock = clock
        self._ids = ids
        self._config = config

    def capture(
        self,
        user_event: Event,
        agent_event: Event,
        appraisal: AppraisalResult,
        trace: Trace,
    ) -> EmotionalMemory | None:
        if not self._config.enabled:
            return None
        intensity = _appraisal_intensity(appraisal)
        if intensity < self._config.min_intensity:
            return None
        emotion_type = _emotion_from_appraisal(appraisal)
        target = (
            "owner relationship"
            if appraisal.relationship_relevance >= 0.45
            else _topic_target(user_event.content)
        )
        memory = EmotionalMemory(
            id=self._ids.new(),
            conversation_id=user_event.conversation_id,
            correlation_id=user_event.correlation_id,
            origin_trace_id=trace.id,
            emotion_type=emotion_type,
            target=target,
            trigger_summary=user_event.content[:500],
            felt_summary=_felt_summary(emotion_type, target, appraisal.valence_signal, intensity),
            valence=appraisal.valence_signal,
            arousal=appraisal.arousal_signal,
            intensity=intensity,
            action_tendency=_action_tendency(emotion_type),
            source_event_ids=[user_event.id, agent_event.id],
            source_trace_ids=[trace.id],
            created_at_ms=trace.created_at_ms,
            updated_at_ms=trace.created_at_ms,
        )
        return self._repository.insert(memory)

    def capture_from_trace(self, trace: Trace) -> EmotionalMemory | None:
        """Capture a salient trace when a low-latency turn skipped full appraisal."""
        if not self._config.enabled:
            return None
        intensity = _trace_intensity(trace)
        cue_intensity = _emotional_cue_intensity(trace.content)
        intensity = max(intensity, cue_intensity)
        if intensity < self._config.min_intensity:
            return None
        emotion_type = _emotion_from_trace(trace, content=trace.content)
        target = "owner relationship" if not trace.is_internal else "inner reflection"
        source_events = [trace.input_event_id]
        if trace.output_event_id is not None:
            source_events.append(trace.output_event_id)
        return self._repository.insert(
            EmotionalMemory(
                id=self._ids.new(),
                conversation_id=trace.conversation_id,
                correlation_id=trace.correlation_id,
                origin_trace_id=trace.id,
                emotion_type=emotion_type,
                target=target,
                trigger_summary=trace.content[:500],
                felt_summary=_felt_summary(
                    emotion_type,
                    target,
                    trace.valence,
                    intensity,
                ),
                valence=trace.valence,
                arousal=trace.arousal,
                intensity=intensity,
                action_tendency=_action_tendency(emotion_type),
                source_event_ids=source_events,
                source_trace_ids=[trace.id],
                created_at_ms=trace.created_at_ms,
                updated_at_ms=trace.created_at_ms,
            )
        )

    def backfill(self, conversation_id: str, traces: list[Trace]) -> int:
        if not self._config.enabled or self._config.backfill_limit == 0:
            return 0
        created = 0
        for trace in traces[-self._config.backfill_limit :]:
            if self._repository.find_by_origin_trace(trace.id) is not None:
                continue
            intensity = max(_trace_intensity(trace), _emotional_cue_intensity(trace.content))
            if intensity < self._config.min_intensity:
                continue
            emotion_type = _emotion_from_trace(trace, content=trace.content)
            target = "owner relationship" if not trace.is_internal else "inner reflection"
            source_events = [trace.input_event_id]
            if trace.output_event_id is not None:
                source_events.append(trace.output_event_id)
            self._repository.insert(
                EmotionalMemory(
                    id=self._ids.new(),
                    conversation_id=conversation_id,
                    correlation_id=trace.correlation_id,
                    origin_trace_id=trace.id,
                    emotion_type=emotion_type,
                    target=target,
                    trigger_summary=trace.content[:500],
                    felt_summary=_felt_summary(
                        emotion_type,
                        target,
                        trace.valence,
                        intensity,
                    ),
                    valence=trace.valence,
                    arousal=trace.arousal,
                    intensity=intensity,
                    action_tendency=_action_tendency(emotion_type),
                    source_event_ids=source_events,
                    source_trace_ids=[trace.id],
                    created_at_ms=trace.created_at_ms,
                    updated_at_ms=trace.created_at_ms,
                )
            )
            created += 1
        return created

    def recall(
        self,
        conversation_id: str,
        query: str,
        activated_traces: list[ActivatedTrace],
        *,
        state: RecallState | None = None,
        situation_mode: str | None = None,
    ) -> list[EmotionalRecall]:
        if not self._config.enabled:
            return []
        query_terms = _terms(query)
        now_ms = self._clock.now_ms()
        ranked: list[EmotionalRecall] = []
        if state is None:
            return self._recall_legacy(
                conversation_id,
                activated_traces,
                query_terms=query_terms,
                now_ms=now_ms,
            )

        if situation_mode is not None and situation_mode != state.situation_mode:
            state = RecallState(
                valence=state.valence,
                arousal=state.arousal,
                energy=state.energy,
                connection_need=state.connection_need,
                situation_mode=situation_mode,
                origin_intent=state.origin_intent,
            )
        active_by_id = {item.trace.id: item for item in activated_traces}
        resonance_config = ResonanceConfig()
        metric_tensor = state_metric_tensor(state, resonance_config)
        for memory in self._repository.candidates(
            conversation_id,
            limit=self._config.candidate_limit,
        ):
            memory_terms = _terms(f"{memory.target} {memory.trigger_summary} {memory.felt_summary}")
            lexical = _overlap(query_terms, memory_terms)
            matching = [
                active_by_id[trace_id]
                for trace_id in memory.source_trace_ids
                if trace_id in active_by_id
            ]
            trace = max(
                (min(1.0, item.coupling_mass or item.score) for item in matching),
                default=0.0,
            )
            age_days = max(0.0, (now_ms - memory.updated_at_ms) / _DAY_MS)
            recency = 0.5 ** (age_days / self._config.recency_half_life_days)
            if lexical <= 0.0 and trace <= 0.0:
                continue
            content_distance = 1.0 - lexical
            affect_distance = hypot(
                (memory.valence - state.valence) / 2.0,
                memory.arousal - state.arousal,
            ) / sqrt(2.0)
            relation_distance = (
                0.0 if memory.target == "owner relationship" else 1.0 - state.connection_need
            )
            situation_distance = 0.0 if state.origin_intent and matching else 1.0 - recency
            arc_distance = min(
                (item.detuning / (1.0 + item.detuning) for item in matching),
                default=0.65,
            )
            rehearsal = max(
                0,
                memory.recall_count + len(set(memory.source_event_ids)) - 1,
            )
            alpha = min(1.0, len(set(memory.source_event_ids)) / 3.0)
            damping = 1.0 / (1.0 + rehearsal * alpha)
            metrics = resonance_metrics(
                content_distance=content_distance,
                affect_distance=affect_distance,
                relation_distance=relation_distance,
                situation_distance=situation_distance,
                arc_distance=arc_distance,
                coupling_mass=trace,
                damping=damping,
                config=resonance_config,
                metric_tensor=metric_tensor,
            )
            ranked.append(
                EmotionalRecall(
                    memory=memory,
                    score=metrics.amplitude,
                    lexical_score=lexical,
                    trace_score=trace,
                    recency_score=recency,
                    resonance_amplitude=metrics.amplitude,
                    detuning=metrics.detuning,
                    damping=damping,
                )
            )
        if not ranked:
            return []
        background = max(
            resonance_config.background_floor,
            float(median(item.resonance_amplitude for item in ranked))
            if len(ranked) > 1
            else resonance_config.background_floor,
        )
        emerged = [
            replace(item, resonance_ratio=item.resonance_amplitude / background)
            for item in ranked
            if item.resonance_amplitude / background >= resonance_config.emergence_ratio
        ]
        recalled = sorted(emerged, key=lambda item: item.score, reverse=True)[
            : self._config.recall_limit
        ]
        self._repository.mark_recalled(
            [item.memory.id for item in recalled],
            now_ms,
        )
        return recalled

    def _recall_legacy(
        self,
        conversation_id: str,
        activated_traces: list[ActivatedTrace],
        *,
        query_terms: set[str],
        now_ms: int,
    ) -> list[EmotionalRecall]:
        """Preserve the pre-resonance score for callers without an organism state."""
        active_ids = {item.trace.id for item in activated_traces}
        ranked: list[EmotionalRecall] = []
        for memory in self._repository.candidates(
            conversation_id,
            limit=self._config.candidate_limit,
        ):
            memory_terms = _terms(f"{memory.target} {memory.trigger_summary} {memory.felt_summary}")
            lexical = _overlap(query_terms, memory_terms)
            trace = 1.0 if active_ids.intersection(memory.source_trace_ids) else 0.0
            age_days = max(0.0, (now_ms - memory.updated_at_ms) / _DAY_MS)
            recency = 0.5 ** (age_days / self._config.recency_half_life_days)
            score = 0.38 * lexical + 0.27 * trace + 0.22 * memory.intensity + 0.13 * recency
            if lexical <= 0.0 and trace <= 0.0 and score < 0.27:
                continue
            ranked.append(
                EmotionalRecall(
                    memory=memory,
                    score=score,
                    lexical_score=lexical,
                    trace_score=trace,
                    recency_score=recency,
                )
            )
        recalled = sorted(ranked, key=lambda item: item.score, reverse=True)[
            : self._config.recall_limit
        ]
        self._repository.mark_recalled(
            [item.memory.id for item in recalled],
            now_ms,
        )
        return recalled


def _appraisal_intensity(appraisal: AppraisalResult) -> float:
    return _clip(
        0.32 * appraisal.arousal_signal
        + 0.28 * abs(appraisal.valence_signal)
        + 0.16 * appraisal.relationship_relevance
        + 0.14 * appraisal.novelty
        + 0.10 * appraisal.urgency
    )


def _trace_intensity(trace: Trace) -> float:
    return _clip(0.42 * trace.arousal + 0.33 * abs(trace.valence) + 0.25 * trace.importance)


def _emotion_from_appraisal(appraisal: AppraisalResult) -> EmotionType:
    if appraisal.valence_signal <= -0.30:
        return (
            EmotionType.CONCERN
            if appraisal.relationship_relevance >= 0.45
            else EmotionType.FRUSTRATION
        )
    if appraisal.relationship_relevance >= 0.55 and appraisal.valence_signal >= 0.12:
        return EmotionType.ATTACHMENT
    if appraisal.novelty >= 0.65 or appraisal.certainty <= 0.35:
        return EmotionType.CURIOSITY
    if appraisal.arousal_signal >= 0.58 and appraisal.valence_signal >= 0.0:
        return EmotionType.ANTICIPATION
    if appraisal.valence_signal >= 0.25:
        return EmotionType.CONTENTMENT
    return EmotionType.AMBIVALENCE


def _emotion_from_trace(trace: Trace, *, content: str) -> EmotionType:
    lowered = content.casefold()
    if any(cue in lowered for cue in _CONCERN_CUES):
        return EmotionType.CONCERN
    if any(cue in lowered for cue in _ATTACHMENT_CUES):
        return EmotionType.ATTACHMENT
    if any(cue in lowered for cue in _FRUSTRATION_CUES):
        return EmotionType.FRUSTRATION
    if trace.valence <= -0.30:
        return EmotionType.CONCERN if trace.arousal >= 0.45 else EmotionType.FRUSTRATION
    if trace.valence >= 0.30:
        return EmotionType.ATTACHMENT if trace.arousal >= 0.45 else EmotionType.CONTENTMENT
    if trace.arousal >= 0.55:
        return EmotionType.CURIOSITY
    return EmotionType.AMBIVALENCE


_ATTACHMENT_CUES = ("想你", "喜欢你", "爱你", "陪我", "miss you", "love you")
_CONCERN_CUES = ("难过", "害怕", "担心", "焦虑", "sad", "afraid", "worried", "anxious")
_FRUSTRATION_CUES = ("生气", "烦", "讨厌", "失败", "angry", "annoyed", "hate")


def _emotional_cue_intensity(content: str) -> float:
    lowered = content.casefold()
    if any(cue in lowered for cue in (*_ATTACHMENT_CUES, *_CONCERN_CUES)):
        return 0.62
    if any(cue in lowered for cue in _FRUSTRATION_CUES):
        return 0.55
    return 0.0


def _action_tendency(emotion_type: EmotionType) -> str:
    return {
        EmotionType.ATTACHMENT: "preserve closeness",
        EmotionType.CURIOSITY: "explore and understand",
        EmotionType.CONCERN: "check safety and support",
        EmotionType.ANTICIPATION: "prepare for what comes next",
        EmotionType.FRUSTRATION: "slow down and repair obstruction",
        EmotionType.CONTENTMENT: "remain present",
        EmotionType.AMBIVALENCE: "hold multiple feelings without forcing closure",
    }[emotion_type]


def _felt_summary(
    emotion_type: EmotionType,
    target: str,
    valence: float,
    intensity: float,
) -> str:
    tone = "warm" if valence > 0.15 else "heavy" if valence < -0.15 else "mixed"
    return (
        f"A {tone} {emotion_type.value} response toward {target}, "
        f"held with {intensity:.2f} subjective intensity."
    )


def _topic_target(content: str) -> str:
    compact = " ".join(content.split())
    return compact[:80] or "current conversation"


def _terms(value: str) -> set[str]:
    lowered = value.casefold()
    terms = set(_ASCII_WORD_RE.findall(lowered))
    for run in _CJK_RUN_RE.findall(lowered):
        terms.update(run)
        terms.update(run[index : index + 2] for index in range(max(0, len(run) - 1)))
    return terms


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, min(len(left), len(right)))


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


__all__ = ["EmotionalMemoryService", "EmotionalRecall"]
