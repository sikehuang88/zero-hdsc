"""Appraisal domain model — what an event means for her, the relationship, and goals.

Pipeline §16 (M07):
- `AppraisalResult`: structured evaluation of an event's significance.
- All numeric fields are bounded; goal_congruence and valence_signal
  range [-1, 1], others [0, 1].
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class AppraisalResult(BaseModel):
    """Structured appraisal of an event (pipeline §16.2)."""

    novelty: float
    goal_congruence: float  # [-1, 1]
    controllability: float
    certainty: float
    self_agency: float
    user_agency: float
    external_agency: float
    relationship_relevance: float
    urgency: float
    valence_signal: float  # [-1, 1]
    arousal_signal: float
    supported_event_ids: list[str] = []
    supported_memory_ids: list[str] = []
    explanation: str = ""

    @field_validator("novelty", "controllability", "certainty", "self_agency",
                     "user_agency", "external_agency", "relationship_relevance",
                     "urgency", "arousal_signal")
    @classmethod
    def _check_0_1(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"value must be in [0, 1], got {v}")
        return v

    @field_validator("goal_congruence", "valence_signal")
    @classmethod
    def _check_neg1_1(cls, v: float) -> float:
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"value must be in [-1, 1], got {v}")
        return v

    @classmethod
    def neutral(cls) -> AppraisalResult:
        """Neutral fallback appraisal (pipeline §16.4).

        Used when LLM fails, times out, or returns invalid JSON.
        Must NOT raise trust, closeness, or important long-term conclusions.
        """
        return cls(
            novelty=0.5,
            goal_congruence=0.0,
            controllability=0.5,
            certainty=0.2,
            self_agency=0.3,
            user_agency=0.3,
            external_agency=0.3,
            relationship_relevance=0.3,
            urgency=0.2,
            valence_signal=0.0,
            arousal_signal=0.2,
            supported_event_ids=[],
            supported_memory_ids=[],
            explanation="Neutral fallback — LLM appraisal unavailable.",
        )


__all__ = ["AppraisalResult"]
