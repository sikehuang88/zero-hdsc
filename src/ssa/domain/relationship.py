"""Relationship state and evidence-bearing transition inputs.

Pipeline section 18 (M09) models relationship state as a versioned snapshot.
Numeric changes are computed by ``RelationshipService``; the input models in
this module make the event or promise evidence for those changes explicit.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, PrivateAttr, field_validator

from ssa.domain.enums import ActionIntent


class RelationshipEventKind(StrEnum):
    """Deterministic relationship-relevant event categories."""

    NEUTRAL = "neutral"
    PROMISE_CREATED = "promise_created"
    USER_PROMISE_KEPT = "user_promise_kept"
    AGENT_PROMISE_KEPT = "agent_promise_kept"
    IMPORTANT_IGNORED = "important_ignored"
    CONFLICT = "conflict"
    REPAIR = "repair"
    SHARED_PROJECT_PROGRESS = "shared_project_progress"
    SHARED_RITUAL = "shared_ritual"


def _validate_identifier(value: str, field_name: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _validate_identifier_list(values: list[str], field_name: str) -> list[str]:
    for value in values:
        _validate_identifier(value, field_name)
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


class RelationshipSignal(BaseModel):
    """A normalized user/world event used to build a transient preview.

    ``event_id`` is mandatory even for a numerically neutral signal. This keeps
    every calculated transition reproducible from an immutable source event.
    The optional IDs name relationship memories affected by that event.
    """

    event_id: str
    kind: RelationshipEventKind = RelationshipEventKind.NEUTRAL
    major: bool = False
    commitment_ids: list[str] = Field(default_factory=list)
    unresolved_memory_ids: list[str] = Field(default_factory=list)
    shared_ritual_ids: list[str] = Field(default_factory=list)

    @field_validator("event_id")
    @classmethod
    def _event_id_not_empty(cls, value: str) -> str:
        return _validate_identifier(value, "event_id")

    @field_validator("commitment_ids", "unresolved_memory_ids", "shared_ritual_ids")
    @classmethod
    def _ids_are_unique(cls, values: list[str], info: object) -> list[str]:
        field_name = getattr(info, "field_name", "identifier list")
        return _validate_identifier_list(values, str(field_name))


class RelationshipAction(BaseModel):
    """Agent action effects applied when a preview is finalized."""

    event_id: str
    intent: ActionIntent
    major: bool = False
    fulfilled_commitment_ids: list[str] = Field(default_factory=list)
    resolved_memory_ids: list[str] = Field(default_factory=list)
    shared_ritual_ids: list[str] = Field(default_factory=list)

    @field_validator("event_id")
    @classmethod
    def _event_id_not_empty(cls, value: str) -> str:
        return _validate_identifier(value, "event_id")

    @field_validator("fulfilled_commitment_ids", "resolved_memory_ids", "shared_ritual_ids")
    @classmethod
    def _ids_are_unique(cls, values: list[str], info: object) -> list[str]:
        field_name = getattr(info, "field_name", "identifier list")
        return _validate_identifier_list(values, str(field_name))


class RelationshipState(BaseModel):
    """A bounded, versioned relationship snapshot (pipeline section 18.1)."""

    version: int
    trust: float
    closeness: float
    tension: float
    reciprocity: float
    repair_debt: float
    shared_ritual_ids: list[str] = Field(default_factory=list)
    active_commitment_ids: list[str] = Field(default_factory=list)
    unresolved_memory_ids: list[str] = Field(default_factory=list)
    updated_at_ms: int

    # Preview-only provenance. It is deliberately excluded from persistence and
    # allows ``finalize(preview, ActionIntent)`` to retain evidence linkage.
    _transition_event_id: str | None = PrivateAttr(default=None)
    _transition_major: bool = PrivateAttr(default=False)
    _transition_origin: tuple[float, float, float, float, float] | None = PrivateAttr(default=None)

    @field_validator("version")
    @classmethod
    def _version_is_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("version must be >= 1")
        return value

    @field_validator("trust", "closeness", "tension", "reciprocity", "repair_debt")
    @classmethod
    def _value_is_bounded(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"value must be in [0, 1], got {value}")
        return value

    @field_validator("shared_ritual_ids", "active_commitment_ids", "unresolved_memory_ids")
    @classmethod
    def _ids_are_unique(cls, values: list[str], info: object) -> list[str]:
        field_name = getattr(info, "field_name", "identifier list")
        return _validate_identifier_list(values, str(field_name))

    @field_validator("updated_at_ms")
    @classmethod
    def _timestamp_is_nonnegative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("updated_at_ms must be non-negative")
        return value

    @classmethod
    def initial(cls, now_ms: int) -> RelationshipState:
        """Create a conservative initial relationship snapshot."""
        return cls(
            version=1,
            trust=0.5,
            closeness=0.25,
            tension=0.0,
            reciprocity=0.5,
            repair_debt=0.0,
            updated_at_ms=now_ms,
        )


__all__ = [
    "RelationshipAction",
    "RelationshipEventKind",
    "RelationshipSignal",
    "RelationshipState",
]
