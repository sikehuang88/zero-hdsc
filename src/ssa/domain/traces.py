"""Append-only trace-space models and activation projections."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ssa.domain.enums import SourceKind


class Trace(BaseModel):
    """One immutable content-bearing point in the HDSC trace space."""

    id: str
    conversation_id: str
    correlation_id: str
    input_event_id: str
    output_event_id: str | None = None
    content: str
    content_type: str = "episode"
    source_kind: SourceKind
    importance: float
    valence: float
    arousal: float
    is_internal: bool = False
    embedding_model: str
    embedding_dim: int = Field(gt=0)
    vec_rowid: int = Field(gt=0)
    created_at_ms: int = Field(ge=0)

    @field_validator("id", "conversation_id", "correlation_id", "input_event_id", "content")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("trace text fields must not be empty")
        return value

    @field_validator("importance", "arousal")
    @classmethod
    def _unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("trace value must be in [0, 1]")
        return value

    @field_validator("valence")
    @classmethod
    def _signed_interval(cls, value: float) -> float:
        if not -1.0 <= value <= 1.0:
            raise ValueError("trace valence must be in [-1, 1]")
        return value


class TraceLink(BaseModel):
    """Directed association used for bounded radiation activation."""

    source_trace_id: str
    target_trace_id: str
    link_type: str = "semantic"
    weight: float
    created_at_ms: int = Field(ge=0)


class ActivatedTrace(BaseModel):
    """Query-time activation; the underlying trace remains immutable."""

    trace: Trace
    rank: int = Field(ge=1)
    score: float = Field(ge=0.0)
    semantic_similarity: float = Field(ge=0.0, le=1.0)
    freshness: float = Field(ge=0.0, le=1.0)
    importance_factor: float = Field(ge=0.0, le=1.0)
    activation_kind: Literal["main", "radiation"]


class TraceNode(BaseModel):
    """Two-dimensional read model for terminal visualization."""

    trace_id: str
    content: str
    content_type: str
    source_kind: SourceKind
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    importance: float
    freshness: float
    activation_score: float = Field(ge=0.0)
    activation_kind: Literal["main", "radiation"] | None = None
    created_at_ms: int


class TraceSpaceStabilityAudit(BaseModel):
    """Read model for the H2 shadow certificate; never used as prompt context."""

    shadow_model_id: str = "hdsc-h2-bounded-active-shadow"
    phase: Literal["awaiting-input", "replay", "post-write", "failed"]
    local_gate: Literal["not-run", "pass", "fail"]
    closed_loop_gate: Literal["not-measured", "conditional-pass", "fail"]
    evaluated_archive_count: int | None = Field(default=None, ge=0)
    active_unit: Literal["semantic-cluster"] = "semantic-cluster"
    active_count: int | None = Field(default=None, ge=0)
    active_capacity: int = Field(ge=1)
    active_mass: float | None = Field(default=None, ge=0.0, le=1.0)
    mass_budget: float = 1.0
    null_mass: float | None = Field(default=None, ge=0.0, le=1.0)
    mass_residual: float | None = None
    contraction_bound: float | None = Field(default=None, ge=0.0, lt=1.0)
    active_set_churn: float | None = Field(default=None, ge=0.0, le=1.0)
    joint_spectral_radius: float | None = Field(default=None, ge=0.0)
    certified_radius: float | None = Field(default=None, ge=0.0)
    duplicate_count: int = Field(default=0, ge=0)
    dropped_mass: float = Field(default=0.0, ge=0.0)
    certificate_reason: str = ""


class TraceSpaceSnapshot(BaseModel):
    """Inspectable spatial projection plus the latest activated region."""

    nodes: list[TraceNode] = Field(default_factory=list)
    links: list[TraceLink] = Field(default_factory=list)
    total_traces: int = 0
    latest_query: str = ""
    projection: str = "embedding-pca"
    activation_model: str = "legacy-ssa-a0"
    serving_model_id: str = "legacy-ssa-a0"
    legacy_activated_count: int | None = Field(default=None, ge=0)
    shadow_model_id: str | None = "hdsc-h2-bounded-active-shadow"
    shadow_status: Literal["disabled", "awaiting-replay", "awaiting-input", "passed", "failed"] = (
        "awaiting-input"
    )
    shadow_affects_prompt: bool = False
    stability_audit: TraceSpaceStabilityAudit | None = None

    @property
    def visible_count(self) -> int:
        return len(self.nodes)

    @property
    def activated_count(self) -> int:
        if self.legacy_activated_count is not None:
            return self.legacy_activated_count
        return sum(node.activation_kind is not None for node in self.nodes)


__all__ = [
    "ActivatedTrace",
    "Trace",
    "TraceLink",
    "TraceNode",
    "TraceSpaceSnapshot",
    "TraceSpaceStabilityAudit",
]
