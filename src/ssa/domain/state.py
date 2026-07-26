"""Organism state — energy, needs, emotion dimensions.

Pipeline §17 (M08):
- `OrganismState`: versioned, immutable snapshot of internal state.
- `StateEngine`: deterministic preview + finalize (no LLM calls).
- valence range [-1, 1], others [0, 1].
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, field_validator

from ssa.config import StateConfig
from ssa.domain.appraisal import AppraisalResult
from ssa.domain.enums import ActionIntent


class OrganismState(BaseModel):
    """Versioned organism state snapshot (pipeline §17.1)."""

    version: int
    energy: float
    connection_need: float
    autonomy_need: float
    curiosity: float
    safety: float
    valence: float  # [-1, 1]
    arousal: float
    updated_at_ms: int

    @field_validator("energy", "connection_need", "autonomy_need",
                     "curiosity", "safety", "arousal")
    @classmethod
    def _check_0_1(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"value must be in [0, 1], got {v}")
        return v

    @field_validator("valence")
    @classmethod
    def _check_neg1_1(cls, v: float) -> float:
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"value must be in [-1, 1], got {v}")
        return v

    @classmethod
    def initial(cls, now_ms: int) -> OrganismState:
        """Create the initial state for a new database."""
        return cls(
            version=1,
            energy=0.7,
            connection_need=0.4,
            autonomy_need=0.3,
            curiosity=0.6,
            safety=0.7,
            valence=0.0,
            arousal=0.2,
            updated_at_ms=now_ms,
        )


@runtime_checkable
class StateEngine(Protocol):
    """Pipeline §17.1 StateEngine protocol."""

    def preview(
        self,
        old: OrganismState,
        appraisal: AppraisalResult,
        now_ms: int,
    ) -> OrganismState: ...

    def finalize(
        self,
        preview: OrganismState,
        intent: ActionIntent,
    ) -> OrganismState: ...


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class DeterministicStateEngine:
    """Production state engine — pure functions, no LLM (pipeline §17).

    Pipeline §17.2 update order:
    1. Time-based decay/recovery.
    2. Appraisal impact.
    3. Action cost and need satisfaction (in finalize).
    4. Clip.
    """

    def __init__(self, config: StateConfig) -> None:
        self._config = config

    def preview(
        self,
        old: OrganismState,
        appraisal: AppraisalResult,
        now_ms: int,
    ) -> OrganismState:
        """Generate a non-persisted candidate state (pipeline §17.2 steps 1-5)."""
        elapsed_ms = max(0, now_ms - old.updated_at_ms)
        elapsed_hours = elapsed_ms / (1000 * 60 * 60)

        # Step 1: time-based decay/recovery.
        # Energy recovers over time.
        energy_recovery = elapsed_hours * 0.05  # 5% per hour
        decayed_energy = _clip(old.energy + energy_recovery)

        # Arousal decays over time.
        arousal_decay = elapsed_hours * self._config.arousal_decay_per_hour
        decayed_arousal = _clip(old.arousal - arousal_decay)

        # Connection need increases without interaction.
        connection_increase = elapsed_hours * 0.03
        decayed_connection = _clip(old.connection_need + connection_increase)

        # Step 2: appraisal impact.
        c = self._config

        # Valence: EMA of appraisal.valence_signal.
        new_valence = _clip(
            old.valence * c.valence_decay + appraisal.valence_signal * c.valence_signal_weight,
            -1.0, 1.0,
        )

        # Arousal: increases with appraisal signal.
        new_arousal = _clip(
            decayed_arousal + appraisal.arousal_signal * c.arousal_signal_weight
        )

        # Energy: consumed by high arousal and urgency.
        energy_cost = appraisal.urgency * 0.05 + appraisal.arousal_signal * 0.03
        new_energy = _clip(decayed_energy - energy_cost)

        # Connection need: decreases with relationship-relevant events.
        connection_change = appraisal.relationship_relevance * 0.1
        new_connection = _clip(decayed_connection - connection_change)

        # Curiosity: increases with novelty.
        new_curiosity = _clip(old.curiosity + (appraisal.novelty - 0.5) * 0.1)

        # Safety: affected by controllability and certainty.
        safety_delta = (appraisal.controllability - 0.5) * 0.05 + (appraisal.certainty - 0.5) * 0.03
        new_safety = _clip(old.safety + safety_delta)

        return OrganismState(
            version=old.version,  # version bumps on finalize
            energy=new_energy,
            connection_need=new_connection,
            autonomy_need=old.autonomy_need,  # not changed by appraisal alone
            curiosity=new_curiosity,
            safety=new_safety,
            valence=new_valence,
            arousal=new_arousal,
            updated_at_ms=now_ms,
        )

    def finalize(
        self,
        preview: OrganismState,
        intent: ActionIntent,
    ) -> OrganismState:
        """Add action effects and produce final state (pipeline §17.2 steps 6-8)."""
        energy = preview.energy
        connection = preview.connection_need
        autonomy = preview.autonomy_need
        curiosity = preview.curiosity

        # Action costs and need satisfaction (pipeline §17.3).
        if intent == ActionIntent.ANSWER or intent == ActionIntent.ACKNOWLEDGE:
            energy = _clip(energy - 0.02)
            connection = _clip(connection - 0.05)

        elif intent == ActionIntent.COMFORT:
            energy = _clip(energy - 0.03)
            connection = _clip(connection - 0.1)

        elif intent == ActionIntent.ASK:
            energy = _clip(energy - 0.02)
            curiosity = _clip(curiosity - 0.05)

        elif intent == ActionIntent.SHARE:
            energy = _clip(energy - 0.03)
            connection = _clip(connection - 0.08)
            autonomy = _clip(autonomy - 0.03)

        elif intent == ActionIntent.CHALLENGE:
            energy = _clip(energy - 0.04)

        elif intent == ActionIntent.REPAIR:
            energy = _clip(energy - 0.05)
            connection = _clip(connection - 0.05)

        elif intent == ActionIntent.PROJECT_WORK:
            energy = _clip(energy - 0.08)
            autonomy = _clip(autonomy - 0.1)
            curiosity = _clip(curiosity - 0.05)

        elif intent == ActionIntent.INITIATE:
            energy = _clip(energy - 0.03)
            connection = _clip(connection - 0.05)
            autonomy = _clip(autonomy - 0.02)

        elif intent == ActionIntent.DEFER:
            autonomy = _clip(autonomy + 0.05)

        elif intent == ActionIntent.REST:
            energy = _clip(energy + 0.1)
            curiosity = _clip(curiosity + 0.05)

        return OrganismState(
            version=preview.version + 1,
            energy=energy,
            connection_need=connection,
            autonomy_need=autonomy,
            curiosity=curiosity,
            safety=preview.safety,
            valence=preview.valence,
            arousal=preview.arousal,
            updated_at_ms=preview.updated_at_ms,
        )


__all__ = ["DeterministicStateEngine", "OrganismState", "StateEngine"]
