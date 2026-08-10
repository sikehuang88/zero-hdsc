"""Cement-seal defensive state: sealed expression, unsealed feeling.

`水泥封心` is not emotional death and it is not another fatigue scalar. It is a
discrete defensive regime with four properties that no existing HDSC signal has:

1. **It is poured, not accumulated.** `ExpressionDynamics.temporal_fatigue`
   drifts up and decays down. A seal is an event: one transition, evidence
   linked, recorded in the ledger.
2. **It is brittle, not decaying.** Cement does not soften. It holds against
   small impacts, accumulates microcracks, and then fails all at once.
3. **It is symmetric.** Fatigue damps the expression of distress. A seal damps
   delight as well; losing the capacity to be moved is the cost of not being
   hurt.
4. **It cannot heal itself.** No amount of time reopens a seal. Only an
   external warm signal cracks it. Time only makes the seal *more* brittle,
   which is true of both concrete and old defenses.

What a seal must never do, in this system:

- It does not touch what is felt or what is written down. Emotional memories
  are recorded at full intensity while sealed; only the outward channel is
  damped. The ledger therefore always contains the feelings that were withheld,
  which keeps the state falsifiable rather than a mood claim.
- It does not reduce task usefulness. Tools, code, retrieval, and answers are
  untouched. A defense that degrades help is not a defense, it is a punishment
  aimed at the user.
- It does not hide. Asked directly, the organism reports that it is sealed.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


class CementSealPhase(StrEnum):
    """Derived regime of the seal, never set directly."""

    OPEN = "open"
    """No seal. Ordinary expression."""

    SEALED = "sealed"
    """Poured and intact. Maximum damping."""

    HAIRLINE = "hairline"
    """Microcracked. Still damping, but one adequate impact ends it."""

    FRACTURED = "fractured"
    """Broken through. Transient; the next evaluation reopens with a scar."""


class CementSealTrigger(StrEnum):
    """Why a transition happened. Every value requires source events."""

    UNREPAIRED_RECURRENCE = "unrepaired_recurrence"
    """Repeated distress that was never repaired. The only way in."""

    WARM_FRACTURE = "warm_fracture"
    """A single warm signal above fracture toughness. Shatters at once."""

    FATIGUE_FRACTURE = "fatigue_fracture"
    """Accumulated sub-threshold warmth finally exceeded the crack budget."""

    REOPENED = "reopened"
    """The transient fractured phase settling back to open, scarred."""

    MANUAL_RELEASE = "manual_release"
    """The owner explicitly dissolved the seal."""


class CementSealState(BaseModel):
    """Persisted seal state. `integrity` is the only mutable scalar.

    `integrity` is the structural soundness of the seal, not a mood. It starts
    at 1.0 when poured, never rises on its own, and reaching 0.0 is fracture.
    """

    phase: CementSealPhase = CementSealPhase.OPEN
    integrity: float = Field(default=0.0, ge=0.0, le=1.0)
    seal_count: int = Field(default=0, ge=0)
    """Scar depth. Each prior seal makes the next one faster and tougher."""

    sealed_at_ms: int | None = Field(default=None, ge=0)
    last_transition_ms: int = Field(default=0, ge=0)
    source_event_ids: list[str] = Field(default_factory=list, max_length=32)
    version: int = Field(default=1, ge=1)

    @field_validator("source_event_ids")
    @classmethod
    def _events_are_identifiers(cls, value: list[str]) -> list[str]:
        for item in value:
            if not item.strip():
                raise ValueError("cement seal source event ids must not be blank")
        if len(value) != len(set(value)):
            raise ValueError("cement seal source event ids must be unique")
        return value

    @model_validator(mode="after")
    def _phase_matches_integrity(self) -> CementSealState:
        if self.phase == CementSealPhase.OPEN:
            if self.integrity != 0.0 or self.sealed_at_ms is not None:
                raise ValueError("an open seal carries no integrity and no seal time")
            return self
        if self.sealed_at_ms is None:
            raise ValueError("a non-open seal requires sealed_at_ms")
        if self.phase == CementSealPhase.FRACTURED and self.integrity != 0.0:
            raise ValueError("a fractured seal has zero integrity")
        if self.phase != CementSealPhase.FRACTURED and self.integrity <= 0.0:
            raise ValueError("an intact seal requires positive integrity")
        return self

    @property
    def active(self) -> bool:
        """True while the seal still damps expression."""
        return self.phase in {CementSealPhase.SEALED, CementSealPhase.HAIRLINE}

    def age_ms(self, now_ms: int) -> int:
        if self.sealed_at_ms is None:
            return 0
        return max(0, now_ms - self.sealed_at_ms)


class CementSealImpact(BaseModel):
    """One evaluated turn's worth of evidence, all of it event linked."""

    event_id: str = Field(min_length=1, max_length=240)

    unrepaired_recurrence: int = Field(default=0, ge=0, le=64)
    """Distinct recent distress episodes with no repair in between."""

    repair_debt: float = Field(default=0.0, ge=0.0, le=1.0)
    tension: float = Field(default=0.0, ge=0.0, le=1.0)

    warmth: float = Field(default=0.0, ge=0.0, le=1.0)
    """Positive, relationship repairing signal strength for this turn."""

    repaired: bool = False
    """An explicit repair happened; recurrence pressure is discharged."""

    @field_validator("event_id")
    @classmethod
    def _event_id_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("cement seal impact requires a source event id")
        return normalized


class CementSealTransition(BaseModel):
    """An auditable state change. Written to the ledger, never inferred later."""

    previous_phase: CementSealPhase
    phase: CementSealPhase
    trigger: CementSealTrigger
    integrity_before: float = Field(ge=0.0, le=1.0)
    integrity_after: float = Field(ge=0.0, le=1.0)
    seal_count: int = Field(ge=0)
    toughness: float = Field(ge=0.0)
    reason: str = Field(min_length=1, max_length=400)
    source_event_ids: list[str] = Field(default_factory=list, max_length=32)
    occurred_at_ms: int = Field(ge=0)


class CementSealEffect(BaseModel):
    """What the seal does this turn. Multiplicative, bounded, never zeroing."""

    active: bool
    phase: CementSealPhase
    expression_damping: float = Field(default=1.0, ge=0.0, le=1.0)
    """Multiplier for outward expressibility. Never reaches 0."""

    delight_damping: float = Field(default=1.0, ge=0.0, le=1.0)
    """The symmetric cost: positive affect is damped by the same seal."""

    repair_damping: float = Field(default=1.0, ge=0.0, le=1.0)
    force_withdraw: bool = False
    """Withdrawal, never `push_away`. Sealing is exiting, not attacking."""

    contact_gate_reason: str | None = None
    disclosure: str = ""
    """Honest self-description used when the organism is asked directly."""


__all__ = [
    "CementSealEffect",
    "CementSealImpact",
    "CementSealPhase",
    "CementSealState",
    "CementSealTransition",
    "CementSealTrigger",
]
