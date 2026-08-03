"""Versioned, evidence-backed self beliefs.

Pipeline section 19 (M10) treats identity claims as revisable conclusions,
not persona constants. Each persisted value is an immutable version linked to
the events that support or contradict it.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator


class SelfBeliefStatus(StrEnum):
    """Lifecycle states for a self belief (pipeline section 19.2)."""

    CANDIDATE = "candidate"
    ACTIVE = "active"
    CHALLENGED = "challenged"
    REVISED = "revised"
    ARCHIVED = "archived"


class SelfBeliefEvidenceRelation(StrEnum):
    """How an immutable event relates to one belief version."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"


class SelfBeliefCandidate(BaseModel):
    """A model-proposed claim that passed structural evidence validation."""

    claim: str
    confidence: float
    evidence_event_ids: list[str]
    change_reason: str
    model: str
    prompt_version: str

    @field_validator("claim", "change_reason", "model", "prompt_version")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("confidence")
    @classmethod
    def _confidence_in_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {value}")
        return value

    @field_validator("evidence_event_ids")
    @classmethod
    def _independent_evidence(cls, values: list[str]) -> list[str]:
        unique = list(dict.fromkeys(values))
        if len(unique) < 2:
            raise ValueError("a self-belief candidate needs at least two independent events")
        if any(not event_id.strip() for event_id in unique):
            raise ValueError("evidence event IDs must not be empty")
        return unique


class SelfBelief(BaseModel):
    """One immutable version in a logical self-belief lineage."""

    id: str
    lineage_id: str
    claim: str
    confidence: float
    status: SelfBeliefStatus
    version: int = Field(ge=1)
    evidence_event_ids: list[str] = Field(default_factory=list)
    counterevidence_event_ids: list[str] = Field(default_factory=list)
    previous_id: str | None = None
    cause_event_id: str | None
    change_reason: str
    model: str | None
    prompt_version: str | None
    candidate_since_ms: int = Field(ge=0)
    activated_at_ms: int | None = Field(default=None, ge=0)
    created_at_ms: int = Field(ge=0)
    updated_at_ms: int = Field(ge=0)

    @field_validator("id", "lineage_id", "claim", "change_reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("cause_event_id", "model", "prompt_version")
    @classmethod
    def _optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("confidence")
    @classmethod
    def _confidence_in_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {value}")
        return value

    @field_validator("evidence_event_ids", "counterevidence_event_ids")
    @classmethod
    def _unique_event_ids(cls, values: list[str]) -> list[str]:
        if any(not event_id.strip() for event_id in values):
            raise ValueError("evidence event IDs must not be empty")
        if len(values) != len(set(values)):
            raise ValueError("evidence event IDs must be unique")
        return values

    @model_validator(mode="after")
    def _validate_version_shape(self) -> Self:
        if self.version == 1 and self.previous_id is not None:
            raise ValueError("the first self-belief version must not have a predecessor")
        if self.version > 1 and self.previous_id is None:
            raise ValueError("later self-belief versions must reference their predecessor")
        overlap = set(self.evidence_event_ids) & set(self.counterevidence_event_ids)
        if overlap:
            raise ValueError(f"events cannot both support and contradict a belief: {overlap}")
        if self.updated_at_ms < self.created_at_ms:
            raise ValueError("updated_at_ms must not precede created_at_ms")
        if self.activated_at_ms is not None and self.activated_at_ms < self.candidate_since_ms:
            raise ValueError("activated_at_ms must not precede candidate_since_ms")
        if (
            self.status
            in {
                SelfBeliefStatus.ACTIVE,
                SelfBeliefStatus.CHALLENGED,
                SelfBeliefStatus.REVISED,
            }
            and self.activated_at_ms is None
        ):
            raise ValueError(f"{self.status.value} beliefs require activated_at_ms")
        return self

    def to_export_dict(self) -> dict[str, Any]:
        """Return a JSON-ready record for the identity package."""
        return self.model_dump(mode="json")


class SelfBeliefEvidence(BaseModel):
    """A normalized evidence link for audit and foreign-key integrity."""

    belief_version_id: str
    event_id: str
    relation: SelfBeliefEvidenceRelation
    recorded_at_ms: int = Field(ge=0)


_ALLOWED_TRANSITIONS: dict[SelfBeliefStatus, frozenset[SelfBeliefStatus]] = {
    SelfBeliefStatus.CANDIDATE: frozenset(
        {
            SelfBeliefStatus.CANDIDATE,
            SelfBeliefStatus.ACTIVE,
            SelfBeliefStatus.ARCHIVED,
        }
    ),
    SelfBeliefStatus.ACTIVE: frozenset(
        {
            SelfBeliefStatus.ACTIVE,
            SelfBeliefStatus.CHALLENGED,
            SelfBeliefStatus.ARCHIVED,
        }
    ),
    SelfBeliefStatus.CHALLENGED: frozenset(
        {
            SelfBeliefStatus.CHALLENGED,
            SelfBeliefStatus.REVISED,
            SelfBeliefStatus.ARCHIVED,
        }
    ),
    # A revised current claim can accumulate evidence or be challenged again.
    SelfBeliefStatus.REVISED: frozenset(
        {
            SelfBeliefStatus.REVISED,
            SelfBeliefStatus.CHALLENGED,
            SelfBeliefStatus.ARCHIVED,
        }
    ),
    SelfBeliefStatus.ARCHIVED: frozenset(),
}


def validate_self_belief_transition(
    previous: SelfBeliefStatus,
    current: SelfBeliefStatus,
) -> None:
    """Reject lifecycle jumps that bypass evidence review."""
    if current not in _ALLOWED_TRANSITIONS[previous]:
        raise ValueError(f"invalid self-belief transition: {previous.value} -> {current.value}")


__all__ = [
    "SelfBelief",
    "SelfBeliefCandidate",
    "SelfBeliefEvidence",
    "SelfBeliefEvidenceRelation",
    "SelfBeliefStatus",
    "validate_self_belief_transition",
]
