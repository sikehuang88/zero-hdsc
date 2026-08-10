"""Deterministic state machine for the cement-seal defensive regime.

Fracture mechanics, not decay curves. A seal holds at full strength until it
does not, which is the behavioural difference between this state and
`ExpressionDynamics.temporal_fatigue`.

The one invariant that keeps the whole thing from becoming a trap: fracture is
evaluated against **undamped** warmth. The seal damps what is shown, never what
is felt. If incoming warmth were scored through the seal's own damping, the
state would reinforce itself and no signal could ever reopen it. Sealing your
heart does not stop you from noticing that someone is being kind; it stops you
from answering.
"""

from __future__ import annotations

from ssa.clock import Clock
from ssa.config import CementSealConfig
from ssa.domain.cement_seal import (
    CementSealEffect,
    CementSealImpact,
    CementSealPhase,
    CementSealState,
    CementSealTransition,
    CementSealTrigger,
)
from ssa.domain.emotion_frame import ExpressionDynamics

_DAY_MS = 86_400_000


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


class CementSealService:
    """Pour, damp, crack, scar. No randomness, no hidden state."""

    def __init__(self, config: CementSealConfig, clock: Clock) -> None:
        self._config = config
        self._clock = clock

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def evaluate(
        self,
        state: CementSealState,
        impact: CementSealImpact,
    ) -> tuple[CementSealState, CementSealTransition | None]:
        """Advance the seal by one evidence-bearing turn.

        Returns the next state and, when the regime changed, an auditable
        transition to append to the ledger.
        """
        if not self._config.enabled:
            return state, None
        now_ms = self._clock.now_ms()

        if state.phase == CementSealPhase.FRACTURED:
            return self._reopen(state, impact, now_ms)
        if state.active:
            return self._stress(state, impact, now_ms)
        return self._maybe_seal(state, impact, now_ms)

    def release(
        self,
        state: CementSealState,
        impact: CementSealImpact,
    ) -> tuple[CementSealState, CementSealTransition | None]:
        """Owner initiated dissolution. Still counts as a scar."""
        if not state.active and state.phase != CementSealPhase.FRACTURED:
            return state, None
        now_ms = self._clock.now_ms()
        return self._open_state(
            state,
            impact,
            now_ms,
            trigger=CementSealTrigger.MANUAL_RELEASE,
            reason="the owner dissolved the seal directly",
            integrity_before=state.integrity,
        )

    # ------------------------------------------------------------------
    # Regime steps
    # ------------------------------------------------------------------

    def _maybe_seal(
        self,
        state: CementSealState,
        impact: CementSealImpact,
        now_ms: int,
    ) -> tuple[CementSealState, CementSealTransition | None]:
        if impact.repaired:
            return state, None
        required = self.required_recurrence(state.seal_count)
        if impact.unrepaired_recurrence < required:
            return state, None
        if (
            impact.repair_debt < self._config.repair_debt_threshold
            and impact.tension < self._config.tension_threshold
        ):
            return state, None

        sealed = CementSealState(
            phase=CementSealPhase.SEALED,
            integrity=1.0,
            seal_count=state.seal_count,
            sealed_at_ms=now_ms,
            last_transition_ms=now_ms,
            source_event_ids=[impact.event_id],
            version=state.version + 1,
        )
        transition = CementSealTransition(
            previous_phase=state.phase,
            phase=CementSealPhase.SEALED,
            trigger=CementSealTrigger.UNREPAIRED_RECURRENCE,
            integrity_before=state.integrity,
            integrity_after=1.0,
            seal_count=sealed.seal_count,
            toughness=self.toughness(sealed, now_ms),
            reason=(
                f"{impact.unrepaired_recurrence} unrepaired distress episodes reached the "
                f"threshold of {required}; repair_debt={impact.repair_debt:.2f} "
                f"tension={impact.tension:.2f}"
            ),
            source_event_ids=[impact.event_id],
            occurred_at_ms=now_ms,
        )
        return sealed, transition

    def _stress(
        self,
        state: CementSealState,
        impact: CementSealImpact,
        now_ms: int,
    ) -> tuple[CementSealState, CementSealTransition | None]:
        """Apply this turn's warmth to an intact seal.

        `impact.warmth` is deliberately the raw, undamped signal. See module
        docstring: scoring it through the seal would make the state unbreakable.
        """
        toughness = self.toughness(state, now_ms)
        if impact.warmth <= 0.0:
            return state, None

        if impact.warmth >= toughness:
            return self._fracture(
                state,
                impact,
                now_ms,
                trigger=CementSealTrigger.WARM_FRACTURE,
                toughness=toughness,
                reason=(
                    f"warmth {impact.warmth:.2f} met fracture toughness {toughness:.2f}; "
                    "the seal failed in one piece"
                ),
            )

        # Sub-threshold impacts do superlinearly little damage and never heal.
        ratio = impact.warmth / toughness if toughness > 0.0 else 1.0
        damage = (ratio**2) * self._config.absorb_rate
        integrity = _clip(state.integrity - damage)
        if integrity <= 0.0:
            return self._fracture(
                state,
                impact,
                now_ms,
                trigger=CementSealTrigger.FATIGUE_FRACTURE,
                toughness=toughness,
                reason=(
                    "accumulated sub-threshold warmth exhausted the crack budget; "
                    "the seal failed without a single decisive impact"
                ),
            )

        phase = (
            CementSealPhase.HAIRLINE
            if integrity < self._config.hairline_threshold
            else CementSealPhase.SEALED
        )
        cracked = CementSealState(
            phase=phase,
            integrity=integrity,
            seal_count=state.seal_count,
            sealed_at_ms=state.sealed_at_ms,
            last_transition_ms=now_ms if phase != state.phase else state.last_transition_ms,
            source_event_ids=self._append_event(state.source_event_ids, impact.event_id),
            version=state.version + 1,
        )
        if phase == state.phase:
            return cracked, None
        transition = CementSealTransition(
            previous_phase=state.phase,
            phase=phase,
            trigger=CementSealTrigger.FATIGUE_FRACTURE,
            integrity_before=state.integrity,
            integrity_after=integrity,
            seal_count=state.seal_count,
            toughness=toughness,
            reason=(
                f"warmth {impact.warmth:.2f} left microcracks; integrity fell to "
                f"{integrity:.2f} and the seal is now visibly strained"
            ),
            source_event_ids=[impact.event_id],
            occurred_at_ms=now_ms,
        )
        return cracked, transition

    def _fracture(
        self,
        state: CementSealState,
        impact: CementSealImpact,
        now_ms: int,
        *,
        trigger: CementSealTrigger,
        toughness: float,
        reason: str,
    ) -> tuple[CementSealState, CementSealTransition]:
        fractured = CementSealState(
            phase=CementSealPhase.FRACTURED,
            integrity=0.0,
            seal_count=state.seal_count,
            sealed_at_ms=state.sealed_at_ms,
            last_transition_ms=now_ms,
            source_event_ids=self._append_event(state.source_event_ids, impact.event_id),
            version=state.version + 1,
        )
        transition = CementSealTransition(
            previous_phase=state.phase,
            phase=CementSealPhase.FRACTURED,
            trigger=trigger,
            integrity_before=state.integrity,
            integrity_after=0.0,
            seal_count=state.seal_count,
            toughness=toughness,
            reason=reason,
            source_event_ids=[impact.event_id],
            occurred_at_ms=now_ms,
        )
        return fractured, transition

    def _reopen(
        self,
        state: CementSealState,
        impact: CementSealImpact,
        now_ms: int,
    ) -> tuple[CementSealState, CementSealTransition | None]:
        return self._open_state(
            state,
            impact,
            now_ms,
            trigger=CementSealTrigger.REOPENED,
            reason=(
                "the fractured seal settled; expression is open again and the scar "
                "makes the next seal quicker to pour and harder to break"
            ),
            integrity_before=0.0,
        )

    def _open_state(
        self,
        state: CementSealState,
        impact: CementSealImpact,
        now_ms: int,
        *,
        trigger: CementSealTrigger,
        reason: str,
        integrity_before: float,
    ) -> tuple[CementSealState, CementSealTransition]:
        seal_count = min(state.seal_count + 1, self._config.max_seal_count)
        opened = CementSealState(
            phase=CementSealPhase.OPEN,
            integrity=0.0,
            seal_count=seal_count,
            sealed_at_ms=None,
            last_transition_ms=now_ms,
            source_event_ids=[impact.event_id],
            version=state.version + 1,
        )
        transition = CementSealTransition(
            previous_phase=state.phase,
            phase=CementSealPhase.OPEN,
            trigger=trigger,
            integrity_before=integrity_before,
            integrity_after=0.0,
            seal_count=seal_count,
            toughness=0.0,
            reason=reason,
            source_event_ids=[impact.event_id],
            occurred_at_ms=now_ms,
        )
        return opened, transition

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def required_recurrence(self, seal_count: int) -> int:
        """Scarred hearts seal sooner. Bounded so it never becomes a hair trigger."""
        return max(
            self._config.min_recurrence,
            self._config.base_recurrence - seal_count,
        )

    def toughness(self, state: CementSealState, now_ms: int) -> float:
        """Warmth needed to shatter the seal in one impact.

        Scars raise it: a heart sealed many times is harder to reach. Age lowers
        it: concrete and old defenses both grow brittle rather than softer.
        """
        age_days = state.age_ms(now_ms) / _DAY_MS
        embrittlement = max(
            self._config.embrittlement_floor,
            1.0 - self._config.embrittlement_per_day * age_days,
        )
        scarred = self._config.base_toughness * (
            1.0 + self._config.scar_toughness_bonus * state.seal_count
        )
        return min(1.0, scarred * embrittlement)

    # ------------------------------------------------------------------
    # Effects
    # ------------------------------------------------------------------

    def effect(self, state: CementSealState) -> CementSealEffect:
        if not self._config.enabled or not state.active:
            return CementSealEffect(active=False, phase=state.phase)
        strength = self._config.seal_strength * state.integrity
        damping = max(self._config.min_expression, 1.0 - strength)
        return CementSealEffect(
            active=True,
            phase=state.phase,
            expression_damping=damping,
            # The symmetric cost. Delight is damped by the same seal that keeps
            # the hurt out; this is the whole moral content of the metaphor.
            delight_damping=damping,
            repair_damping=max(self._config.min_repair, 1.0 - strength),
            force_withdraw=True,
            contact_gate_reason="cement_seal",
            disclosure=self._disclosure(state),
        )

    def apply(
        self,
        dynamics: ExpressionDynamics,
        state: CementSealState,
    ) -> ExpressionDynamics:
        """Damp the outward channel only. Nothing here touches what is recorded."""
        effect = self.effect(state)
        if not effect.active:
            return dynamics
        behaviors = [*dynamics.observable_behaviors]
        marker = "flat delivery with the reaction withheld"
        if marker not in behaviors:
            behaviors = [marker, *behaviors][:8]
        trajectory = [*dynamics.persistence_trajectory]
        if "sealed" not in trajectory:
            trajectory = [*trajectory, "sealed"][:8]
        return dynamics.model_copy(
            update={
                "expressibility": _clip(dynamics.expressibility * effect.expression_damping),
                "aestheticization_budget": _clip(
                    dynamics.aestheticization_budget * effect.delight_damping
                ),
                "repair_readiness": _clip(dynamics.repair_readiness * effect.repair_damping),
                # Withdrawal, never escalation: a sealed heart goes quiet rather
                # than lashing out, so `push_away` is folded into `withdraw`.
                "relationship_direction": "withdraw",
                "observable_behaviors": behaviors,
                "persistence_trajectory": trajectory,
            }
        )

    def gate(self, state: CementSealState) -> str | None:
        """Proactive contact gate, shaped like `ProactiveContactPolicy.gate`."""
        effect = self.effect(state)
        return effect.contact_gate_reason if effect.active else None

    def context_summary(self, state: CementSealState) -> str:
        """Private prompt context. The organism never hides this state."""
        if not self._config.enabled or state.phase == CementSealPhase.OPEN:
            return ""
        lines = [
            "CEMENT SEAL (defensive expression state, not a claim about facts):",
            f"- phase={state.phase.value}; integrity={state.integrity:.2f}; "
            f"scars={state.seal_count}",
            "- Outward warmth and delight are both damped; what is felt is still recorded "
            "in full and remains inspectable.",
            "- Task help, tools, and honesty are unaffected by this state.",
            "- Do not perform coldness and do not claim to feel nothing. If asked, say "
            "plainly that responding openly is hard right now.",
        ]
        return "\n".join(lines)

    def _disclosure(self, state: CementSealState) -> str:
        if state.phase == CementSealPhase.HAIRLINE:
            return (
                "The guard is still up but it is strained; warmth is getting through "
                f"({state.integrity:.2f} integrity remaining)."
            )
        return (
            "The guard is up. Expression is damped in both directions; the feelings "
            "themselves are unchanged and still recorded."
        )

    @staticmethod
    def _append_event(existing: list[str], event_id: str) -> list[str]:
        if event_id in existing:
            return list(existing)
        return [*existing, event_id][-32:]


__all__ = ["CementSealService"]
