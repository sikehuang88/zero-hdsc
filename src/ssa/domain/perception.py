"""Structured environment and situation perception models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


class DayPhase(StrEnum):
    LATE_NIGHT = "late_night"
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"


class ConversationGap(StrEnum):
    FIRST_CONTACT = "first_contact"
    CONTINUOUS = "continuous"
    RESUMED = "resumed"
    LONG_GAP = "long_gap"


class SituationMode(StrEnum):
    ANSWER = "answer"
    ACT = "act"
    CLARIFY = "clarify"
    SUPPORT = "support"
    REPAIR = "repair"
    REMINISCE = "reminisce"
    EXPLORE = "explore"
    ACKNOWLEDGE = "acknowledge"


class TimePerception(BaseModel):
    """A UTC observation projected onto the configured human clock."""

    observed_at_ms: int = Field(ge=0)
    utc_iso: str
    local_iso: str
    timezone: str
    human_clock: str
    weekday: int = Field(ge=0, le=6)
    is_weekend: bool
    local_hour: float = Field(ge=0.0, lt=24.0)
    day_phase: DayPhase
    is_quiet_hours: bool
    circadian_sin: float = Field(ge=-1.0, le=1.0)
    circadian_cos: float = Field(ge=-1.0, le=1.0)
    elapsed_since_last_event_ms: int | None = Field(default=None, ge=0)
    elapsed_since_last_user_ms: int | None = Field(default=None, ge=0)
    conversation_gap: ConversationGap


class SemanticState(BaseModel):
    """Deterministic semantic control signals extracted from the current event."""

    question: float = Field(ge=0.0, le=1.0)
    directive: float = Field(ge=0.0, le=1.0)
    self_disclosure: float = Field(ge=0.0, le=1.0)
    autobiographical: float = Field(ge=0.0, le=1.0)
    relationship: float = Field(ge=0.0, le=1.0)
    temporal_reference: float = Field(ge=0.0, le=1.0)
    information_gap: float = Field(ge=0.0, le=1.0)


class AffectState(BaseModel):
    """Appraisal-derived affect vector for the current user event."""

    valence: float = Field(ge=-1.0, le=1.0)
    positive: float = Field(ge=0.0, le=1.0)
    negative: float = Field(ge=0.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    urgency: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    intensity: float = Field(ge=0.0, le=1.0)


class ContextState(BaseModel):
    """Conversation, trace, relationship, and continuity signals."""

    trace_support: float = Field(ge=0.0, le=1.0)
    semantic_continuity: float = Field(ge=0.0, le=1.0)
    memory_density: float = Field(ge=0.0, le=1.0)
    conversation_freshness: float = Field(ge=0.0, le=1.0)
    topic_shift: float = Field(ge=0.0, le=1.0)
    coherence: float = Field(ge=0.0, le=1.0)
    relationship_closeness: float = Field(ge=0.0, le=1.0)
    relationship_tension: float = Field(ge=0.0, le=1.0)
    unresolved_pressure: float = Field(ge=0.0, le=1.0)
    recent_turn_count: int = Field(ge=0)


class ResponsePosture(BaseModel):
    """Continuous response controls produced by the cross operator."""

    warmth: float = Field(ge=0.0, le=1.0)
    directness: float = Field(ge=0.0, le=1.0)
    exploration: float = Field(ge=0.0, le=1.0)
    caution: float = Field(ge=0.0, le=1.0)
    temporal_sensitivity: float = Field(ge=0.0, le=1.0)


class SituationPerception(BaseModel):
    """Auditable output of the P1 semantic-affect-context-time operator."""

    model_id: str
    operator_version: str
    time: TimePerception
    semantic: SemanticState
    affect: AffectState
    context: ContextState
    cross_terms: dict[str, float]
    mode_distribution: dict[SituationMode, float]
    primary_mode: SituationMode
    confidence: float = Field(ge=0.0, le=1.0)
    posture: ResponsePosture
    explanation: str

    @field_validator("cross_terms")
    @classmethod
    def _cross_terms_are_bounded(cls, values: dict[str, float]) -> dict[str, float]:
        if any(not 0.0 <= value <= 1.0 for value in values.values()):
            raise ValueError("cross terms must be in [0, 1]")
        return values

    @model_validator(mode="after")
    def _mode_distribution_is_complete(self) -> SituationPerception:
        if set(self.mode_distribution) != set(SituationMode):
            raise ValueError("mode distribution must contain every situation mode")
        if any(value < 0.0 or value > 1.0 for value in self.mode_distribution.values()):
            raise ValueError("mode probabilities must be in [0, 1]")
        total = sum(self.mode_distribution.values())
        if abs(total - 1.0) > 1e-8:
            raise ValueError("mode probabilities must sum to one")
        maximum = max(self.mode_distribution, key=self.mode_distribution.__getitem__)
        if maximum != self.primary_mode:
            raise ValueError("primary mode must maximize the mode distribution")
        return self


class StoredPerception(BaseModel):
    """Immutable persistence envelope for one situation perception."""

    id: str
    correlation_id: str
    conversation_id: str
    query_event_id: str
    result: SituationPerception
    created_at_ms: int = Field(ge=0)


__all__ = [
    "AffectState",
    "ContextState",
    "ConversationGap",
    "DayPhase",
    "ResponsePosture",
    "SemanticState",
    "SituationMode",
    "SituationPerception",
    "StoredPerception",
    "TimePerception",
]
