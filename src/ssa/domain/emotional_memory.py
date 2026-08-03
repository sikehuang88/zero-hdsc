"""Long-term subjective emotional-memory records."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from ssa.domain.lifecycle import EmotionType


class EmotionalMemoryStatus(StrEnum):
    ACTIVE = "active"
    SETTLED = "settled"
    ARCHIVED = "archived"


class EmotionalMemory(BaseModel):
    """One evidence-linked record of how an interaction felt to the agent."""

    id: str
    conversation_id: str
    correlation_id: str
    origin_trace_id: str
    emotion_type: EmotionType
    target: str
    trigger_summary: str
    felt_summary: str
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    intensity: float = Field(ge=0.0, le=1.0)
    action_tendency: str
    source_event_ids: list[str] = Field(default_factory=list)
    source_trace_ids: list[str] = Field(default_factory=list)
    status: EmotionalMemoryStatus = EmotionalMemoryStatus.ACTIVE
    recall_count: int = Field(default=0, ge=0)
    last_recalled_at_ms: int | None = Field(default=None, ge=0)
    created_at_ms: int = Field(ge=0)
    updated_at_ms: int = Field(ge=0)

    @field_validator(
        "id",
        "conversation_id",
        "correlation_id",
        "origin_trace_id",
        "target",
        "trigger_summary",
        "felt_summary",
        "action_tendency",
    )
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("emotional-memory text fields must not be empty")
        return value


__all__ = ["EmotionalMemory", "EmotionalMemoryStatus"]
