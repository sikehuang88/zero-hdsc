"""Typed directed multi-relation memory graph contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EngramRelation(StrEnum):
    """The relation types declared by HDSC §5.10.2."""

    SEMANTIC = "semantic"
    ENTITY = "entity"
    CATEGORY = "category"
    SPATIAL = "spatial"
    EPISODIC = "episodic"
    PREFERENCE = "preference"
    EVIDENCE = "evidence"
    REVISION = "revision"
    TEMPORAL_FORWARD = "temporal_forward"


EngramNodeType = Literal["trace", "entity", "category", "community"]


class EngramNode(BaseModel):
    node_id: str = Field(min_length=1, max_length=240)
    node_type: EngramNodeType = "trace"
    conversation_id: str = Field(min_length=1, max_length=160)
    created_at_ms: int = Field(ge=0)

    @field_validator("node_id", "conversation_id")
    @classmethod
    def _node_text_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("engram node identifiers must not be blank")
        return normalized


class EngramEdge(BaseModel):
    """One directed support in W^(r,a); reverse edges need separate evidence."""

    edge_id: str = Field(min_length=1, max_length=240)
    src: str = Field(min_length=1, max_length=240)
    dst: str = Field(min_length=1, max_length=240)
    rel_type: EngramRelation
    support: float = Field(gt=0.0)
    action: str = Field(default="default", min_length=1, max_length=80)
    valid_from_ms: int = Field(ge=0)
    valid_to_ms: int | None = Field(default=None, ge=0)
    source_event_id: str | None = Field(default=None, max_length=240)
    trust_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    likelihood_ratio: float = Field(default=1.0, gt=0.0)

    @field_validator("edge_id", "src", "dst", "action")
    @classmethod
    def _edge_text_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("engram edge identifiers must not be blank")
        return normalized

    @field_validator("source_event_id")
    @classmethod
    def _optional_source_event_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @property
    def active(self) -> bool:
        return self.valid_to_ms is None

    def transport_weight(self, gate: float) -> float:
        """Apply query gate and evidence confidence without inventing reverse support."""
        return self.support * gate * self.trust_weight * self.likelihood_ratio


class EngramPathStep(BaseModel):
    edge_src: str = Field(min_length=1, max_length=240)
    edge_dst: str = Field(min_length=1, max_length=240)
    rel_type: EngramRelation
    source_event_id: str | None = Field(default=None, max_length=240)
    support: float = Field(gt=0.0)


class EngramActivation(BaseModel):
    """Query-time mass with its best contributing typed transport path.

    ``hops`` is the length of that retained transport path, not a shortest-path
    distance. A node can therefore report a longer path when it received more
    mass through that path than through a shorter alternative.
    """

    node_id: str = Field(min_length=1, max_length=240)
    mass: float = Field(ge=0.0)
    hops: int = Field(ge=0)
    path: list[EngramPathStep] = Field(default_factory=list, max_length=32)
    is_assertable: bool


class EngramTransportAudit(BaseModel):
    mode: Literal["truncated_power", "ppr", "bounded_active"]
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    seed_mass: float = Field(ge=0.0)
    propagated_mass: float = Field(ge=0.0)
    null_mass: float = Field(ge=0.0)
    conservation_residual: float
    max_hops: int = Field(ge=0)
    path_grounded_count: int = Field(ge=0)


class EngramQueryResult(BaseModel):
    activations: list[EngramActivation] = Field(default_factory=list, max_length=256)
    audit: EngramTransportAudit


__all__ = [
    "EngramActivation",
    "EngramEdge",
    "EngramNode",
    "EngramNodeType",
    "EngramPathStep",
    "EngramQueryResult",
    "EngramRelation",
    "EngramTransportAudit",
]
