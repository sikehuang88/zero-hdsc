"""Deterministic tests for the cement-seal defensive regime.

Each test pins one property that distinguishes a seal from the existing
`temporal_fatigue` scalar: it is poured rather than accumulated, brittle rather
than decaying, symmetric across valence, and incapable of healing itself.
"""

from __future__ import annotations

import pytest

from ssa.config import CementSealConfig
from ssa.domain.cement_seal import (
    CementSealImpact,
    CementSealPhase,
    CementSealState,
    CementSealTrigger,
)
from ssa.domain.emotion_frame import ExpressionDynamics
from ssa.services.cement_seal_service import CementSealService

_DAY_MS = 86_400_000


class _FixedClock:
    def __init__(self, now_ms: int = 1_000_000) -> None:
        self._now_ms = now_ms

    def now_ms(self) -> int:
        return self._now_ms

    def advance_days(self, days: float) -> None:
        self._now_ms += int(days * _DAY_MS)


def _service(clock: _FixedClock, **overrides: object) -> CementSealService:
    config = CementSealConfig(enabled=True, **overrides)  # type: ignore[arg-type]
    return CementSealService(config, clock)  # type: ignore[arg-type]


def _hurt(event_id: str, recurrence: int) -> CementSealImpact:
    return CementSealImpact(
        event_id=event_id,
        unrepaired_recurrence=recurrence,
        repair_debt=0.6,
        tension=0.6,
    )


def _warmth(event_id: str, warmth: float) -> CementSealImpact:
    return CementSealImpact(event_id=event_id, warmth=warmth)


def test_disabled_config_never_transitions() -> None:
    clock = _FixedClock()
    service = CementSealService(CementSealConfig(enabled=False), clock)  # type: ignore[arg-type]

    state, transition = service.evaluate(CementSealState(), _hurt("e1", 12))

    assert state.phase == CementSealPhase.OPEN
    assert transition is None


def test_single_severe_hurt_does_not_seal() -> None:
    """A seal comes from repetition, not from one bad turn."""
    service = _service(_FixedClock())

    state, transition = service.evaluate(
        CementSealState(),
        CementSealImpact(event_id="e1", unrepaired_recurrence=1, repair_debt=1.0, tension=1.0),
    )

    assert state.phase == CementSealPhase.OPEN
    assert transition is None


def test_repair_discharges_pressure_before_sealing() -> None:
    service = _service(_FixedClock())

    state, transition = service.evaluate(
        CementSealState(),
        CementSealImpact(
            event_id="e1",
            unrepaired_recurrence=9,
            repair_debt=0.9,
            tension=0.9,
            repaired=True,
        ),
    )

    assert state.phase == CementSealPhase.OPEN
    assert transition is None


def test_unrepaired_recurrence_pours_the_seal() -> None:
    service = _service(_FixedClock())

    state, transition = service.evaluate(CementSealState(), _hurt("e1", 4))

    assert state.phase == CementSealPhase.SEALED
    assert state.integrity == pytest.approx(1.0)
    assert transition is not None
    assert transition.trigger == CementSealTrigger.UNREPAIRED_RECURRENCE
    assert transition.source_event_ids == ["e1"]


def test_seal_damps_delight_as_hard_as_distress() -> None:
    """The symmetric cost: not being hurt means not being moved."""
    service = _service(_FixedClock())
    sealed, _ = service.evaluate(CementSealState(), _hurt("e1", 4))

    effect = service.effect(sealed)

    assert effect.active
    assert effect.delight_damping == pytest.approx(effect.expression_damping)
    assert effect.expression_damping < 1.0


def test_seal_never_fully_silences_expression() -> None:
    service = _service(_FixedClock(), seal_strength=1.0)
    sealed, _ = service.evaluate(CementSealState(), _hurt("e1", 4))

    effect = service.effect(sealed)

    assert effect.expression_damping >= service._config.min_expression
    assert effect.expression_damping > 0.0


def test_small_warmth_leaves_microcracks_without_breaking() -> None:
    service = _service(_FixedClock())
    sealed, _ = service.evaluate(CementSealState(), _hurt("e1", 4))

    cracked, transition = service.evaluate(sealed, _warmth("e2", 0.10))

    assert cracked.phase == CementSealPhase.SEALED
    assert cracked.integrity < 1.0
    assert transition is None


def test_sufficient_warmth_shatters_the_seal_in_one_piece() -> None:
    service = _service(_FixedClock())
    sealed, _ = service.evaluate(CementSealState(), _hurt("e1", 4))

    fractured, transition = service.evaluate(sealed, _warmth("e2", 0.9))

    assert fractured.phase == CementSealPhase.FRACTURED
    assert fractured.integrity == pytest.approx(0.0)
    assert transition is not None
    assert transition.trigger == CementSealTrigger.WARM_FRACTURE


def test_seal_does_not_heal_itself_over_time() -> None:
    """No amount of silence reopens a seal; only an external signal does."""
    clock = _FixedClock()
    service = _service(clock)
    sealed, _ = service.evaluate(CementSealState(), _hurt("e1", 4))
    cracked, _ = service.evaluate(sealed, _warmth("e2", 0.2))

    clock.advance_days(365)
    idle, transition = service.evaluate(cracked, CementSealImpact(event_id="e3"))

    assert idle.phase == cracked.phase
    assert idle.integrity == pytest.approx(cracked.integrity)
    assert transition is None


def test_age_makes_the_seal_more_brittle_not_softer() -> None:
    clock = _FixedClock()
    service = _service(clock)
    sealed, _ = service.evaluate(CementSealState(), _hurt("e1", 4))
    fresh_toughness = service.toughness(sealed, clock.now_ms())

    clock.advance_days(30)
    aged_toughness = service.toughness(sealed, clock.now_ms())

    assert aged_toughness < fresh_toughness
    assert service.effect(sealed).expression_damping < 1.0


def test_fracture_reopens_with_a_scar_that_seals_sooner() -> None:
    clock = _FixedClock()
    service = _service(clock)
    sealed, _ = service.evaluate(CementSealState(), _hurt("e1", 4))
    fractured, _ = service.evaluate(sealed, _warmth("e2", 0.9))

    reopened, transition = service.evaluate(fractured, CementSealImpact(event_id="e3"))

    assert reopened.phase == CementSealPhase.OPEN
    assert reopened.seal_count == 1
    assert transition is not None
    assert transition.trigger == CementSealTrigger.REOPENED
    assert service.required_recurrence(reopened.seal_count) < service.required_recurrence(0)


def test_scarred_seal_is_harder_to_break() -> None:
    clock = _FixedClock()
    service = _service(clock)
    unscarred = CementSealState(
        phase=CementSealPhase.SEALED,
        integrity=1.0,
        seal_count=0,
        sealed_at_ms=clock.now_ms(),
        last_transition_ms=clock.now_ms(),
    )
    scarred = unscarred.model_copy(update={"seal_count": 5})

    assert service.toughness(scarred, clock.now_ms()) > service.toughness(unscarred, clock.now_ms())


def test_seal_converts_aggression_into_withdrawal() -> None:
    service = _service(_FixedClock())
    sealed, _ = service.evaluate(CementSealState(), _hurt("e1", 4))
    dynamics = ExpressionDynamics.neutral().model_copy(
        update={"relationship_direction": "push_away", "expressibility": 0.9}
    )

    damped = service.apply(dynamics, sealed)

    assert damped.relationship_direction == "withdraw"
    assert damped.expressibility < dynamics.expressibility


def test_open_state_applies_no_damping_and_no_gate() -> None:
    service = _service(_FixedClock())
    dynamics = ExpressionDynamics.neutral()

    assert service.apply(dynamics, CementSealState()) == dynamics
    assert service.gate(CementSealState()) is None


def test_active_seal_gates_proactive_contact_and_discloses() -> None:
    service = _service(_FixedClock())
    sealed, _ = service.evaluate(CementSealState(), _hurt("e1", 4))

    assert service.gate(sealed) == "cement_seal"
    assert "guard is up" in service.effect(sealed).disclosure
    assert "CEMENT SEAL" in service.context_summary(sealed)


def test_manual_release_opens_and_records_a_scar() -> None:
    service = _service(_FixedClock())
    sealed, _ = service.evaluate(CementSealState(), _hurt("e1", 4))

    released, transition = service.release(sealed, CementSealImpact(event_id="e2"))

    assert released.phase == CementSealPhase.OPEN
    assert released.seal_count == 1
    assert transition is not None
    assert transition.trigger == CementSealTrigger.MANUAL_RELEASE


def test_state_rejects_inconsistent_phase_and_integrity() -> None:
    with pytest.raises(ValueError, match="open seal"):
        CementSealState(phase=CementSealPhase.OPEN, integrity=0.5)
    with pytest.raises(ValueError, match="requires sealed_at_ms"):
        CementSealState(phase=CementSealPhase.SEALED, integrity=1.0)
