"""Deterministic time and situation perception for the interactive runtime."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Final
from zoneinfo import ZoneInfo

from ssa.config import PerceptionConfig
from ssa.domain.appraisal import AppraisalResult
from ssa.domain.enums import Actor
from ssa.domain.events import Event
from ssa.domain.perception import (
    AffectState,
    ContextState,
    ConversationGap,
    DayPhase,
    ResponsePosture,
    SemanticState,
    SituationMode,
    SituationPerception,
    TimePerception,
)
from ssa.domain.relationship import RelationshipState
from ssa.domain.state import OrganismState
from ssa.domain.traces import ActivatedTrace

MODEL_ID: Final = "hdsc-p1-environment-perception"
OPERATOR_VERSION: Final = "semantic-affect-context-time-v1"

_WEEKDAYS: Final = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
_QUESTION_CUES: Final = (
    "?",
    "\uff1f",
    "吗",
    "呢",
    "为什么",
    "怎么",
    "如何",
    "什么",
    "which",
    "what",
    "why",
    "how",
    "can you",
    "could you",
)
_DIRECTIVE_CUES: Final = (
    "请",
    "帮我",
    "需要你",
    "给我",
    "替我",
    "写出",
    "实现",
    "修复",
    "开始",
    "please",
    "build",
    "implement",
    "fix",
    "show me",
)
_SELF_CUES: Final = ("我", "我的", "自己", "i ", "i'm", "my ", "me ")
_DISCLOSURE_CUES: Final = (
    "难过",
    "开心",
    "害怕",
    "焦虑",
    "孤独",
    "生气",
    "担心",
    "喜欢",
    "讨厌",
    "感觉",
    "觉得",
    "feel",
    "afraid",
    "anxious",
    "happy",
    "sad",
    "lonely",
)
_AUTOBIOGRAPHICAL_CUES: Final = (
    "以前",
    "曾经",
    "前女友",
    "前男友",
    "小时候",
    "记得",
    "我们那次",
    "过去",
    "former",
    "used to",
    "remember",
    "last time",
)
_RELATIONSHIP_CUES: Final = (
    "我们",
    "前女友",
    "前男友",
    "朋友",
    "家人",
    "关系",
    "一起",
    "between us",
    "relationship",
    "together",
)
_TEMPORAL_CUES: Final = (
    "今天",
    "明天",
    "昨天",
    "现在",
    "刚才",
    "今晚",
    "早上",
    "下午",
    "晚上",
    "以前",
    "曾经",
    "today",
    "tomorrow",
    "yesterday",
    "now",
    "tonight",
)


class EnvironmentPerceptionService:
    """Combine clock, semantics, appraisal, and context into one P1 snapshot."""

    def __init__(
        self,
        *,
        config: PerceptionConfig,
        timezone_name: str,
        quiet_hours_start: str,
        quiet_hours_end: str,
    ) -> None:
        self._config = config
        self._timezone = ZoneInfo(timezone_name)
        self._timezone_name = timezone_name
        self._quiet_start = _clock_minutes(quiet_hours_start)
        self._quiet_end = _clock_minutes(quiet_hours_end)

    @property
    def context_window_events(self) -> int:
        return self._config.context_window_events

    def perceive(
        self,
        event: Event,
        appraisal: AppraisalResult,
        organism: OrganismState,
        relationship: RelationshipState,
        activated_traces: Sequence[ActivatedTrace],
        recent_events: Sequence[Event],
    ) -> SituationPerception:
        """Run the versioned semantic-affect-context-time cross operator."""
        prior_events = [
            item
            for item in recent_events
            if item.id != event.id and item.created_at_ms <= event.created_at_ms
        ]
        last_event = prior_events[-1] if prior_events else None
        last_user = next(
            (item for item in reversed(prior_events) if item.actor == Actor.USER),
            None,
        )
        time_state = self.perceive_time(
            event.created_at_ms,
            last_event_ms=last_event.created_at_ms if last_event is not None else None,
            last_user_ms=last_user.created_at_ms if last_user is not None else None,
        )
        affect = _affect_state(appraisal)
        semantic = _semantic_state(event.content, appraisal, affect, activated_traces)
        context = _context_state(
            relationship,
            activated_traces,
            prior_events,
            time_state,
        )
        cross_terms = _cross_terms(semantic, affect, context, time_state, appraisal)
        mode_distribution = _mode_distribution(
            semantic,
            affect,
            context,
            cross_terms,
            appraisal,
            organism,
            temperature=self._config.mode_temperature,
        )
        primary_mode = max(mode_distribution, key=mode_distribution.__getitem__)
        posture = _response_posture(
            mode_distribution,
            affect,
            context,
            time_state,
        )
        explanation = (
            f"mode={primary_mode.value}; confidence={mode_distribution[primary_mode]:.3f}; "
            f"semantic_affect={cross_terms['semantic_affect']:.3f}; "
            f"semantic_context={cross_terms['semantic_context']:.3f}; "
            f"affect_context={cross_terms['affect_context']:.3f}; "
            f"action_readiness={cross_terms['action_readiness']:.3f}"
        )
        return SituationPerception(
            model_id=MODEL_ID,
            operator_version=OPERATOR_VERSION,
            time=time_state,
            semantic=semantic,
            affect=affect,
            context=context,
            cross_terms=cross_terms,
            mode_distribution=mode_distribution,
            primary_mode=primary_mode,
            confidence=mode_distribution[primary_mode],
            posture=posture,
            explanation=explanation,
        )

    def perceive_time(
        self,
        now_ms: int,
        *,
        last_event_ms: int | None = None,
        last_user_ms: int | None = None,
    ) -> TimePerception:
        """Project an epoch timestamp into UTC, local clock, and rhythm signals."""
        if now_ms < 0:
            raise ValueError("now_ms must be non-negative")
        utc_time = datetime.fromtimestamp(now_ms / 1000.0, tz=UTC)
        local_time = utc_time.astimezone(self._timezone)
        local_hour = local_time.hour + local_time.minute / 60.0 + local_time.second / 3600.0
        phase = _day_phase(local_hour)
        local_minutes = local_time.hour * 60 + local_time.minute
        quiet = _inside_clock_interval(local_minutes, self._quiet_start, self._quiet_end)
        radians = 2.0 * math.pi * local_hour / 24.0
        elapsed_event = _elapsed(now_ms, last_event_ms)
        elapsed_user = _elapsed(now_ms, last_user_ms)
        gap = _conversation_gap(elapsed_user, self._config)
        weekday = local_time.weekday()
        human_clock = (
            f"{_WEEKDAYS[weekday]} {local_time:%Y-%m-%d %H:%M} "
            f"{self._timezone_name} ({phase.value})"
        )
        return TimePerception(
            observed_at_ms=now_ms,
            utc_iso=utc_time.isoformat(timespec="seconds"),
            local_iso=local_time.isoformat(timespec="seconds"),
            timezone=self._timezone_name,
            human_clock=human_clock,
            weekday=weekday,
            is_weekend=weekday >= 5,
            local_hour=local_hour,
            day_phase=phase,
            is_quiet_hours=quiet,
            circadian_sin=_clip_signed(math.sin(radians)),
            circadian_cos=_clip_signed(math.cos(radians)),
            elapsed_since_last_event_ms=elapsed_event,
            elapsed_since_last_user_ms=elapsed_user,
            conversation_gap=gap,
        )


def _semantic_state(
    content: str,
    appraisal: AppraisalResult,
    affect: AffectState,
    activated_traces: Sequence[ActivatedTrace],
) -> SemanticState:
    text = f" {content.casefold()} "
    question = max(
        1.0 if content.rstrip().endswith(("?", "\uff1f")) else 0.0,
        _cue_strength(text, _QUESTION_CUES),
    )
    directive = _cue_strength(text, _DIRECTIVE_CUES)
    self_reference = _cue_strength(text, _SELF_CUES)
    disclosure_cue = _cue_strength(text, _DISCLOSURE_CUES)
    self_disclosure = _clip(
        max(
            disclosure_cue,
            self_reference
            * (0.20 + 0.50 * affect.intensity + 0.30 * appraisal.relationship_relevance),
        )
    )
    trace_continuity = max(
        (item.semantic_similarity for item in activated_traces),
        default=0.0,
    )
    autobiographical_cue = _cue_strength(text, _AUTOBIOGRAPHICAL_CUES)
    autobiographical = _clip(autobiographical_cue * (0.65 + 0.35 * trace_continuity))
    relationship = _clip(
        max(appraisal.relationship_relevance, _cue_strength(text, _RELATIONSHIP_CUES))
    )
    temporal_reference = _cue_strength(text, _TEMPORAL_CUES)
    information_gap = _clip(
        question * (0.45 + 0.35 * (1.0 - appraisal.certainty) + 0.20 * appraisal.novelty)
    )
    return SemanticState(
        question=question,
        directive=directive,
        self_disclosure=self_disclosure,
        autobiographical=autobiographical,
        relationship=relationship,
        temporal_reference=temporal_reference,
        information_gap=information_gap,
    )


def _affect_state(appraisal: AppraisalResult) -> AffectState:
    positive = max(0.0, appraisal.valence_signal)
    negative = max(0.0, -appraisal.valence_signal)
    intensity = _clip(
        0.45 * abs(appraisal.valence_signal)
        + 0.35 * appraisal.arousal_signal
        + 0.20 * appraisal.urgency
    )
    return AffectState(
        valence=appraisal.valence_signal,
        positive=positive,
        negative=negative,
        arousal=appraisal.arousal_signal,
        urgency=appraisal.urgency,
        uncertainty=1.0 - appraisal.certainty,
        intensity=intensity,
    )


def _context_state(
    relationship: RelationshipState,
    activated_traces: Sequence[ActivatedTrace],
    prior_events: Sequence[Event],
    time_state: TimePerception,
) -> ContextState:
    trace_support = _clip(sum(min(1.0, item.score) for item in activated_traces) / 2.0)
    semantic_continuity = max(
        (item.semantic_similarity for item in activated_traces),
        default=0.0,
    )
    memory_density = _clip(len(activated_traces) / 8.0)
    freshness = {
        ConversationGap.FIRST_CONTACT: 0.25,
        ConversationGap.CONTINUOUS: 1.0,
        ConversationGap.RESUMED: 0.60,
        ConversationGap.LONG_GAP: 0.20,
    }[time_state.conversation_gap]
    recent_turn_count = sum(item.actor == Actor.USER for item in prior_events)
    topic_shift = _clip(1.0 - semantic_continuity) if recent_turn_count > 0 else 0.50
    coherence = _clip(0.50 * semantic_continuity + 0.25 * trace_support + 0.25 * freshness)
    unresolved_pressure = _clip(
        len(relationship.unresolved_memory_ids) / 4.0 + 0.50 * relationship.repair_debt
    )
    return ContextState(
        trace_support=trace_support,
        semantic_continuity=semantic_continuity,
        memory_density=memory_density,
        conversation_freshness=freshness,
        topic_shift=topic_shift,
        coherence=coherence,
        relationship_closeness=relationship.closeness,
        relationship_tension=relationship.tension,
        unresolved_pressure=unresolved_pressure,
        recent_turn_count=recent_turn_count,
    )


def _cross_terms(
    semantic: SemanticState,
    affect: AffectState,
    context: ContextState,
    time_state: TimePerception,
    appraisal: AppraisalResult,
) -> dict[str, float]:
    gap_pressure = {
        ConversationGap.FIRST_CONTACT: 0.25,
        ConversationGap.CONTINUOUS: 0.0,
        ConversationGap.RESUMED: 0.55,
        ConversationGap.LONG_GAP: 1.0,
    }[time_state.conversation_gap]
    return {
        "semantic_affect": _clip(semantic.self_disclosure * affect.intensity),
        "semantic_context": _clip(semantic.autobiographical * context.trace_support),
        "affect_context": _clip(
            affect.negative * max(context.relationship_tension, context.unresolved_pressure)
        ),
        "temporal_context": _clip(gap_pressure * (1.0 - context.conversation_freshness)),
        "action_readiness": _clip(
            semantic.directive * appraisal.urgency * appraisal.controllability
        ),
    }


def _mode_distribution(
    semantic: SemanticState,
    affect: AffectState,
    context: ContextState,
    cross: dict[str, float],
    appraisal: AppraisalResult,
    organism: OrganismState,
    *,
    temperature: float,
) -> dict[SituationMode, float]:
    raw = {
        SituationMode.ANSWER: (
            0.08
            + semantic.question * (0.35 + 0.30 * appraisal.certainty + 0.25 * context.coherence)
        ),
        SituationMode.ACT: (0.08 + 0.45 * semantic.directive + 0.45 * cross["action_readiness"]),
        SituationMode.CLARIFY: (
            0.05
            + semantic.information_gap
            * (0.40 + 0.30 * affect.uncertainty + 0.20 * context.topic_shift)
        ),
        SituationMode.SUPPORT: (
            0.05
            + affect.negative
            * (
                0.25
                + 0.25 * affect.arousal
                + 0.25 * semantic.self_disclosure
                + 0.15 * semantic.relationship
            )
            + 0.20 * cross["semantic_affect"]
        ),
        SituationMode.REPAIR: (
            0.03 + 0.55 * cross["affect_context"] + 0.25 * context.unresolved_pressure
        ),
        SituationMode.REMINISCE: (
            0.04
            + semantic.autobiographical
            * (0.35 + 0.30 * context.trace_support + 0.25 * context.relationship_closeness)
            + 0.20 * cross["semantic_context"]
        ),
        SituationMode.EXPLORE: (
            0.06
            + appraisal.novelty
            * (
                0.25
                + 0.20 * semantic.question
                + 0.25 * organism.curiosity
                + 0.15 * context.topic_shift
            )
        ),
        SituationMode.ACKNOWLEDGE: (
            0.05
            + semantic.self_disclosure
            * (0.25 + 0.25 * semantic.relationship + 0.20 * affect.intensity)
        ),
    }
    maximum = max(raw.values())
    exponentials = {mode: math.exp((score - maximum) / temperature) for mode, score in raw.items()}
    total = sum(exponentials.values())
    return {mode: value / total for mode, value in exponentials.items()}


def _response_posture(
    distribution: dict[SituationMode, float],
    affect: AffectState,
    context: ContextState,
    time_state: TimePerception,
) -> ResponsePosture:
    warmth = _clip(
        0.20
        + 0.55 * distribution[SituationMode.SUPPORT]
        + 0.35 * distribution[SituationMode.REMINISCE]
        + 0.20 * distribution[SituationMode.ACKNOWLEDGE]
        + 0.15 * context.relationship_closeness
    )
    directness = _clip(
        0.20 + 0.55 * distribution[SituationMode.ACT] + 0.45 * distribution[SituationMode.ANSWER]
    )
    exploration = _clip(
        0.15
        + 0.50 * distribution[SituationMode.CLARIFY]
        + 0.45 * distribution[SituationMode.EXPLORE]
    )
    caution = _clip(
        0.15
        + 0.45 * affect.uncertainty
        + 0.20 * context.topic_shift
        + 0.25 * distribution[SituationMode.REPAIR]
    )
    temporal_sensitivity = _clip(
        0.15
        + (0.45 if time_state.is_quiet_hours else 0.0)
        + 0.35 * (1.0 - context.conversation_freshness)
    )
    return ResponsePosture(
        warmth=warmth,
        directness=directness,
        exploration=exploration,
        caution=caution,
        temporal_sensitivity=temporal_sensitivity,
    )


def _clock_minutes(value: str) -> int:
    parts = value.split(":")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError("clock values must use HH:MM")
    hour, minute = (int(part) for part in parts)
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("clock values must use a valid 24-hour time")
    return hour * 60 + minute


def _inside_clock_interval(value: int, start: int, end: int) -> bool:
    if start == end:
        return False
    if start < end:
        return start <= value < end
    return value >= start or value < end


def _day_phase(local_hour: float) -> DayPhase:
    if 5.0 <= local_hour < 12.0:
        return DayPhase.MORNING
    if 12.0 <= local_hour < 18.0:
        return DayPhase.AFTERNOON
    if 18.0 <= local_hour < 23.0:
        return DayPhase.EVENING
    return DayPhase.LATE_NIGHT


def _elapsed(now_ms: int, previous_ms: int | None) -> int | None:
    return None if previous_ms is None else max(0, now_ms - previous_ms)


def _conversation_gap(
    elapsed_user_ms: int | None,
    config: PerceptionConfig,
) -> ConversationGap:
    if elapsed_user_ms is None:
        return ConversationGap.FIRST_CONTACT
    if elapsed_user_ms <= config.fresh_gap_minutes * 60_000:
        return ConversationGap.CONTINUOUS
    if elapsed_user_ms <= int(config.long_gap_hours * 3_600_000):
        return ConversationGap.RESUMED
    return ConversationGap.LONG_GAP


def _cue_strength(text: str, cues: Sequence[str]) -> float:
    hits = sum(cue in text for cue in cues)
    if hits == 0:
        return 0.0
    return _clip(0.55 + 0.15 * (hits - 1))


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clip_signed(value: float) -> float:
    return max(-1.0, min(1.0, value))


__all__ = ["MODEL_ID", "OPERATOR_VERSION", "EnvironmentPerceptionService"]
