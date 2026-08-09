"""Contracts for machine-verifiable predictions and calibration reports."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field, field_validator, model_validator

from ssa.domain.enums import Actor, SourceKind

JsonScalar: TypeAlias = str | int | float | bool | None


class PredictionClaimKind(StrEnum):
    USER_BEHAVIOR = "user_behavior"
    PREFERENCE = "preference"
    SCHEDULE = "schedule"
    WORLD_FACT = "world_fact"


class PredictionVerifierKind(StrEnum):
    EVENT_MATCH = "event_match"
    EXTERNAL_TRUTH = "external_truth"
    TOOL_RESULT = "tool_result"


class PredictionStatus(StrEnum):
    PENDING = "pending"
    RESOLVED_TRUE = "resolved_true"
    RESOLVED_FALSE = "resolved_false"
    EXPIRED = "expired"
    UNVERIFIABLE = "unverifiable"


class EventMatchVerifierSpec(BaseModel):
    """A deterministic predicate over immutable events in a fixed time window."""

    actor: Actor | None = None
    source_kind: SourceKind | None = None
    event_type: str | None = None
    content_equals: str | None = None
    content_contains_all: list[str] = Field(default_factory=list, max_length=16)
    content_contains_any: list[str] = Field(default_factory=list, max_length=16)
    metadata_equals: dict[str, JsonScalar] = Field(default_factory=dict)

    @field_validator("event_type", "content_equals")
    @classmethod
    def _optional_text_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("event matcher text values must not be blank")
        return normalized

    @field_validator("content_contains_all", "content_contains_any")
    @classmethod
    def _terms_are_bounded(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item or len(item) > 200 for item in normalized):
            raise ValueError("event matcher terms must contain 1 to 200 characters")
        if len(normalized) != len(set(normalized)):
            raise ValueError("event matcher terms must be unique")
        return normalized

    @model_validator(mode="after")
    def _has_constraint(self) -> EventMatchVerifierSpec:
        if not any(
            (
                self.actor is not None,
                self.source_kind is not None,
                self.event_type is not None,
                self.content_equals is not None,
                bool(self.content_contains_all),
                bool(self.content_contains_any),
                bool(self.metadata_equals),
            )
        ):
            raise ValueError("event_match verifier requires at least one structural constraint")
        return self


class ExternalTruthVerifierSpec(BaseModel):
    provider: str
    field_path: list[str] = Field(min_length=1, max_length=12)
    operator: Literal["equals", "contains", "less_than", "greater_than"] = "equals"
    expected: JsonScalar

    @field_validator("provider")
    @classmethod
    def _provider_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("external truth provider must not be blank")
        return normalized

    @field_validator("field_path")
    @classmethod
    def _field_path_not_blank(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("external truth field path must not contain blank components")
        return normalized


class ToolResultVerifierSpec(BaseModel):
    tool_name: str
    call_id: str | None = None
    require_ok: bool = True
    metadata_equals: dict[str, JsonScalar] = Field(default_factory=dict)

    @field_validator("tool_name")
    @classmethod
    def _tool_name_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("tool result verifier requires a tool name")
        return normalized


VerifierSpec: TypeAlias = (
    EventMatchVerifierSpec | ExternalTruthVerifierSpec | ToolResultVerifierSpec
)


class GroundedPrediction(BaseModel):
    id: str
    conversation_id: str
    dedup_key: str
    claim_text: str
    claim_kind: PredictionClaimKind
    verifier_kind: PredictionVerifierKind
    verifier_spec: dict[str, Any]
    stated_confidence: float = Field(ge=0.0, le=1.0)
    base_rate_prior: float = Field(default=0.5, ge=0.0, le=1.0)
    source_event_ids: list[str] = Field(default_factory=list)
    source_trace_ids: list[str] = Field(default_factory=list)
    created_at_ms: int = Field(ge=0)
    resolve_after_ms: int = Field(ge=0)
    expires_at_ms: int = Field(ge=0)
    status: PredictionStatus = PredictionStatus.PENDING
    resolved_at_ms: int | None = Field(default=None, ge=0)
    resolved_by_event_id: str | None = None
    brier_contribution: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("conversation_id", "dedup_key", "claim_text")
    @classmethod
    def _required_text_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("prediction text fields must not be blank")
        return normalized

    @field_validator("source_event_ids", "source_trace_ids")
    @classmethod
    def _source_ids_unique(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("prediction source ids must not be blank")
        if len(value) != len(set(value)):
            raise ValueError("prediction source ids must be unique")
        return value

    @model_validator(mode="after")
    def _validate_shape(self) -> GroundedPrediction:
        if self.expires_at_ms <= self.resolve_after_ms:
            raise ValueError("prediction expiry must follow its resolution start")
        _validate_verifier_spec(self.verifier_kind, self.verifier_spec)
        resolved = self.status in {
            PredictionStatus.RESOLVED_TRUE,
            PredictionStatus.RESOLVED_FALSE,
        }
        if resolved and (self.resolved_at_ms is None or self.brier_contribution is None):
            raise ValueError("resolved predictions require resolution time and Brier contribution")
        if not resolved and self.brier_contribution is not None:
            raise ValueError("unscored predictions must not have a Brier contribution")
        if self.status == PredictionStatus.PENDING and self.resolved_at_ms is not None:
            raise ValueError("pending predictions must not have a resolution time")
        if self.status == PredictionStatus.RESOLVED_FALSE and self.resolved_by_event_id is not None:
            raise ValueError("false predictions cannot be resolved by a matching event")
        return self

    def parsed_verifier(self) -> VerifierSpec:
        return _validate_verifier_spec(self.verifier_kind, self.verifier_spec)


class PredictionCalibrationBucket(BaseModel):
    lower_bound: float = Field(ge=0.0, le=1.0)
    upper_bound: float = Field(ge=0.0, le=1.0)
    count: int = Field(ge=0)
    mean_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    observed_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    brier_score: float | None = Field(default=None, ge=0.0, le=1.0)


class PredictionCalibrationReport(BaseModel):
    conversation_id: str
    resolved_count: int = Field(ge=0)
    pending_count: int = Field(ge=0)
    unverifiable_count: int = Field(ge=0)
    brier_score: float | None = Field(default=None, ge=0.0, le=1.0)
    baseline_brier_score: float | None = Field(default=None, ge=0.0, le=1.0)
    brier_skill_score: float | None = None
    buckets: tuple[PredictionCalibrationBucket, ...]


def _validate_verifier_spec(
    kind: PredictionVerifierKind,
    value: dict[str, Any],
) -> VerifierSpec:
    if kind == PredictionVerifierKind.EVENT_MATCH:
        return EventMatchVerifierSpec.model_validate(value)
    if kind == PredictionVerifierKind.EXTERNAL_TRUTH:
        return ExternalTruthVerifierSpec.model_validate(value)
    return ToolResultVerifierSpec.model_validate(value)


__all__ = [
    "EventMatchVerifierSpec",
    "ExternalTruthVerifierSpec",
    "GroundedPrediction",
    "PredictionCalibrationBucket",
    "PredictionCalibrationReport",
    "PredictionClaimKind",
    "PredictionStatus",
    "PredictionVerifierKind",
    "ToolResultVerifierSpec",
    "VerifierSpec",
]
