"""Cause-linked layered emotion and vocal-performance plans."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

RelationshipDirection = Literal["approach", "maintain", "withdraw", "push_away"]


class EmotionComponent(BaseModel):
    """One named feeling directed at a concrete target."""

    name: str = Field(min_length=1, max_length=64)
    intensity: float = Field(ge=0.0, le=1.0)
    target: str = Field(default="owner relationship", min_length=1, max_length=160)


class VoicePerformanceSegment(BaseModel):
    """Acoustic intent for one phase of an emotional reply."""

    id: str = Field(min_length=1, max_length=160)
    sequence: int = Field(ge=0, le=15)
    phase: str = Field(min_length=1, max_length=64)
    recipe: str = Field(min_length=1, max_length=64)
    intensity: float = Field(ge=0.0, le=1.0)
    rate: float = Field(ge=0.72, le=1.18)
    breathiness: float = Field(ge=0.0, le=1.0)
    tremor: float = Field(ge=0.0, le=1.0)
    pitch_stability: float = Field(ge=0.0, le=1.0)
    energy: float = Field(ge=0.0, le=1.0)
    pause_before_ms: int = Field(ge=0, le=2_500)
    ending: str = Field(min_length=1, max_length=64)
    emphasis: list[str] = Field(default_factory=list, max_length=8)


class ExpressionDynamics(BaseModel):
    """Contextual projection from an internal feeling into observable behaviour."""

    trigger_summary: str = Field(min_length=1, max_length=240)
    irritability_spillover: float = Field(ge=0.0, le=1.0)
    expressibility: float = Field(ge=0.0, le=1.0)
    temporal_fatigue: float = Field(ge=0.0, le=1.0)
    care_capacity: float = Field(ge=0.0, le=1.0)
    aestheticization_budget: float = Field(ge=0.0, le=1.0)
    relationship_direction: RelationshipDirection
    recurrence_count: int = Field(ge=0, le=64)
    repair_readiness: float = Field(ge=0.0, le=1.0)
    observable_behaviors: list[str] = Field(min_length=1, max_length=8)
    persistence_trajectory: list[str] = Field(min_length=1, max_length=8)

    @classmethod
    def neutral(cls) -> ExpressionDynamics:
        return cls(
            trigger_summary="no strong affective trigger",
            irritability_spillover=0.0,
            expressibility=0.8,
            temporal_fatigue=0.0,
            care_capacity=0.75,
            aestheticization_budget=0.45,
            relationship_direction="maintain",
            recurrence_count=0,
            repair_readiness=0.8,
            observable_behaviors=["ordinary conversational variation"],
            persistence_trajectory=["settled_presence"],
        )


class EmotionFrame(BaseModel):
    """Layered subjective interpretation for one interaction turn."""

    id: str = Field(min_length=1, max_length=160)
    conversation_id: str = Field(min_length=1, max_length=160)
    correlation_id: str = Field(min_length=1, max_length=160)
    query_event_id: str = Field(min_length=1, max_length=160)
    primary: EmotionComponent
    secondary: list[EmotionComponent] = Field(default_factory=list, max_length=4)
    surface_mask: EmotionComponent | None = None
    inhibition: float = Field(ge=0.0, le=1.0)
    certainty: float = Field(ge=0.0, le=1.0)
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    cause_summary: str = Field(min_length=1, max_length=240)
    inner_conflict: str = Field(min_length=1, max_length=240)
    regulation_strategy: str = Field(min_length=1, max_length=200)
    action_tendency: str = Field(min_length=1, max_length=200)
    expression_dynamics: ExpressionDynamics = Field(default_factory=ExpressionDynamics.neutral)
    trajectory: list[str] = Field(min_length=1, max_length=8)
    voice_segments: list[VoicePerformanceSegment] = Field(min_length=1, max_length=8)
    source_event_ids: list[str] = Field(default_factory=list, max_length=16)
    source_trace_ids: list[str] = Field(default_factory=list, max_length=16)
    created_at_ms: int = Field(ge=0)

    @field_validator("trajectory")
    @classmethod
    def _trajectory_is_nonempty(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if item.strip()]
        if not normalized:
            raise ValueError("trajectory must contain a phase")
        return normalized

    @model_validator(mode="after")
    def _components_and_segments_are_coherent(self) -> EmotionFrame:
        names = [self.primary.name, *(item.name for item in self.secondary)]
        if len(names) != len(set(names)):
            raise ValueError("emotion components must have unique names")
        sequences = [item.sequence for item in self.voice_segments]
        if sequences != list(range(len(sequences))):
            raise ValueError("voice segment sequences must be contiguous from zero")
        return self

    def presentation_data(self) -> dict[str, object]:
        """Return the bounded frontend/TTS projection without private evidence text."""
        return {
            "frame_id": self.id,
            "primary": self.primary.model_dump(mode="json"),
            "secondary": [item.model_dump(mode="json") for item in self.secondary],
            "surface_mask": (
                self.surface_mask.model_dump(mode="json") if self.surface_mask is not None else None
            ),
            "inhibition": self.inhibition,
            "certainty": self.certainty,
            "valence": self.valence,
            "arousal": self.arousal,
            "cause_summary": self.cause_summary,
            "inner_conflict": self.inner_conflict,
            "regulation_strategy": self.regulation_strategy,
            "action_tendency": self.action_tendency,
            "expression_dynamics": self.expression_dynamics.model_dump(mode="json"),
            "trajectory": list(self.trajectory),
            "voice_segments": [item.model_dump(mode="json") for item in self.voice_segments],
        }


__all__ = [
    "EmotionComponent",
    "EmotionFrame",
    "ExpressionDynamics",
    "RelationshipDirection",
    "VoicePerformanceSegment",
]
