"""Deterministic two-stage relationship evolution service."""

from __future__ import annotations

import math
from collections.abc import Mapping

from ssa.clock import Clock, SystemClock
from ssa.config import RelationshipConfig
from ssa.domain.appraisal import AppraisalResult
from ssa.domain.enums import ActionIntent, Actor
from ssa.domain.events import Event
from ssa.domain.relationship import (
    RelationshipAction,
    RelationshipEventKind,
    RelationshipSignal,
    RelationshipState,
)

_DIRECT_EVENT_KINDS: dict[str, RelationshipEventKind] = {
    "relationship.promise": RelationshipEventKind.PROMISE_CREATED,
    "relationship.promise_created": RelationshipEventKind.PROMISE_CREATED,
    "relationship.user_promise_kept": RelationshipEventKind.USER_PROMISE_KEPT,
    "relationship.promise.user_kept": RelationshipEventKind.USER_PROMISE_KEPT,
    "relationship.agent_promise_kept": RelationshipEventKind.AGENT_PROMISE_KEPT,
    "relationship.promise.agent_kept": RelationshipEventKind.AGENT_PROMISE_KEPT,
    "relationship.important_ignored": RelationshipEventKind.IMPORTANT_IGNORED,
    "relationship.ignored": RelationshipEventKind.IMPORTANT_IGNORED,
    "relationship.conflict": RelationshipEventKind.CONFLICT,
    "relationship.repair": RelationshipEventKind.REPAIR,
    "relationship.project_progress": RelationshipEventKind.SHARED_PROJECT_PROGRESS,
    "world.project_progress": RelationshipEventKind.SHARED_PROJECT_PROGRESS,
    "relationship.ritual": RelationshipEventKind.SHARED_RITUAL,
}

_ACTOR_DEPENDENT_PROMISE_EVENTS = {
    "relationship.promise_fulfilled",
    "relationship.promise_kept",
    "relationship.promise.fulfilled",
    "relationship.promise.kept",
}


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _add_unique(current: list[str], additions: list[str]) -> list[str]:
    result = list(current)
    known = set(result)
    for item in additions:
        if item not in known:
            result.append(item)
            known.add(item)
    return result


def _remove_ids(current: list[str], removals: list[str]) -> list[str]:
    removed = set(removals)
    return [item for item in current if item not in removed]


def _apply_transition_cap(
    current: float,
    delta: float,
    origin: float,
    cap: float,
) -> float:
    target_delta = max(-cap, min(cap, current + delta - origin))
    value = _clip(origin + target_delta)
    if abs(value - origin) > cap:
        value = math.nextafter(value, origin)
    return value


class RelationshipService:
    """Map events and actions into bounded relationship snapshots.

    ``preview`` never persists and keeps the old version. ``finalize`` applies
    the selected agent action and increments exactly once. The entire turn is
    capped relative to the state that entered ``preview``.
    """

    def __init__(
        self,
        config: RelationshipConfig | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._config = config or RelationshipConfig()
        self._clock = clock or SystemClock()
        if not math.isfinite(self._config.normal_delta_cap):
            raise ValueError("normal_delta_cap must be finite")
        if not math.isfinite(self._config.major_delta_cap):
            raise ValueError("major_delta_cap must be finite")
        if self._config.normal_delta_cap < 0.0:
            raise ValueError("normal_delta_cap must be non-negative")
        if self._config.major_delta_cap < self._config.normal_delta_cap:
            raise ValueError("major_delta_cap must be >= normal_delta_cap")
        if self._config.major_delta_cap > 1.0:
            raise ValueError("major_delta_cap must be <= 1")

    def signal_from_event(self, event: Event) -> RelationshipSignal:
        """Normalize an immutable Event without inspecting free-form content."""
        metadata: Mapping[str, object] = event.metadata
        event_type = event.event_type.casefold()
        kind = self._kind_from_metadata(metadata)
        if kind is None:
            kind = _DIRECT_EVENT_KINDS.get(event_type)
        if kind is None and event_type in _ACTOR_DEPENDENT_PROMISE_EVENTS:
            if event.actor == Actor.USER:
                kind = RelationshipEventKind.USER_PROMISE_KEPT
            elif event.actor == Actor.AGENT:
                kind = RelationshipEventKind.AGENT_PROMISE_KEPT
        if kind is None:
            kind = RelationshipEventKind.NEUTRAL

        return RelationshipSignal(
            event_id=event.id,
            kind=kind,
            major=metadata.get("major") is True,
            commitment_ids=self._metadata_ids(
                metadata,
                "commitment_ids",
                "commitment_id",
                "promise_memory_ids",
                "promise_memory_id",
            ),
            unresolved_memory_ids=self._metadata_ids(
                metadata,
                "unresolved_memory_ids",
                "unresolved_memory_id",
            ),
            shared_ritual_ids=self._metadata_ids(
                metadata,
                "shared_ritual_ids",
                "shared_ritual_id",
                "ritual_ids",
                "ritual_id",
            ),
        )

    def preview(
        self,
        old: RelationshipState,
        event: Event | RelationshipSignal,
        appraisal: AppraisalResult | None = None,
        now_ms: int | None = None,
    ) -> RelationshipState:
        """Build a transient relationship state from one evidence-bearing event."""
        signal = event if isinstance(event, RelationshipSignal) else self.signal_from_event(event)
        cap = self._cap(signal.major)

        trust_delta = 0.0
        closeness_delta = 0.0
        tension_delta = 0.0
        reciprocity_delta = 0.0
        repair_debt_delta = 0.0

        if signal.kind == RelationshipEventKind.USER_PROMISE_KEPT:
            trust_delta += cap * 0.67
            reciprocity_delta += cap * 0.33
            repair_debt_delta -= cap * 0.33
        elif signal.kind == RelationshipEventKind.AGENT_PROMISE_KEPT:
            reciprocity_delta += cap * 0.67
        elif signal.kind == RelationshipEventKind.IMPORTANT_IGNORED:
            trust_delta -= cap * 0.50
            closeness_delta -= cap * 0.33
            tension_delta += cap * 0.67
            repair_debt_delta += cap * 0.67
        elif signal.kind == RelationshipEventKind.CONFLICT:
            trust_delta -= cap * 0.33
            closeness_delta -= cap * 0.17
            tension_delta += cap * 0.67
            repair_debt_delta += cap * 0.67
        elif signal.kind == RelationshipEventKind.REPAIR:
            trust_delta += cap * 0.17
            closeness_delta += cap * 0.17
            tension_delta -= cap * 0.67
            repair_debt_delta -= cap * 0.67
        elif signal.kind == RelationshipEventKind.SHARED_PROJECT_PROGRESS:
            closeness_delta += cap * 0.67
            reciprocity_delta += cap * 0.50
        elif signal.kind == RelationshipEventKind.SHARED_RITUAL:
            closeness_delta += cap * 0.50
            reciprocity_delta += cap * 0.17
        elif signal.kind == RelationshipEventKind.NEUTRAL and appraisal is not None:
            evidence_supported = signal.event_id in appraisal.supported_event_ids
            evidence_weight = appraisal.relationship_relevance * appraisal.certainty
            if evidence_supported and appraisal.valence_signal > 0.0:
                closeness_delta += cap * 0.17 * appraisal.valence_signal * evidence_weight
                trust_delta += cap * 0.08 * appraisal.valence_signal * evidence_weight
            elif evidence_supported and appraisal.valence_signal < 0.0:
                negative_signal = abs(appraisal.valence_signal) * evidence_weight
                tension_delta += cap * 0.17 * negative_signal
                repair_debt_delta += cap * 0.08 * negative_signal

        commitments = list(old.active_commitment_ids)
        unresolved = list(old.unresolved_memory_ids)
        rituals = list(old.shared_ritual_ids)
        if signal.kind == RelationshipEventKind.PROMISE_CREATED:
            commitments = _add_unique(commitments, signal.commitment_ids)
        elif signal.kind in {
            RelationshipEventKind.USER_PROMISE_KEPT,
            RelationshipEventKind.AGENT_PROMISE_KEPT,
        }:
            commitments = _remove_ids(commitments, signal.commitment_ids)
        elif signal.kind in {
            RelationshipEventKind.IMPORTANT_IGNORED,
            RelationshipEventKind.CONFLICT,
        }:
            unresolved = _add_unique(unresolved, signal.unresolved_memory_ids)
        elif signal.kind == RelationshipEventKind.REPAIR:
            unresolved = _remove_ids(unresolved, signal.unresolved_memory_ids)
        elif signal.kind == RelationshipEventKind.SHARED_RITUAL:
            rituals = _add_unique(rituals, signal.shared_ritual_ids)

        timestamp = self._clock.now_ms() if now_ms is None else now_ms
        preview = RelationshipState(
            version=old.version,
            trust=_apply_transition_cap(old.trust, trust_delta, old.trust, cap),
            closeness=_apply_transition_cap(old.closeness, closeness_delta, old.closeness, cap),
            tension=_apply_transition_cap(old.tension, tension_delta, old.tension, cap),
            reciprocity=_apply_transition_cap(
                old.reciprocity, reciprocity_delta, old.reciprocity, cap
            ),
            repair_debt=_apply_transition_cap(
                old.repair_debt, repair_debt_delta, old.repair_debt, cap
            ),
            shared_ritual_ids=rituals,
            active_commitment_ids=commitments,
            unresolved_memory_ids=unresolved,
            updated_at_ms=timestamp,
        )
        preview._transition_event_id = signal.event_id
        preview._transition_major = signal.major
        preview._transition_origin = self._numeric_values(old)
        return preview

    def finalize(
        self,
        preview: RelationshipState,
        action: RelationshipAction | ActionIntent,
    ) -> RelationshipState:
        """Apply agent effects and return the single committable next version."""
        if isinstance(action, ActionIntent):
            if preview._transition_event_id is None:
                raise ValueError("A bare ActionIntent requires a preview produced by this service")
            normalized_action = RelationshipAction(
                event_id=preview._transition_event_id,
                intent=action,
                major=preview._transition_major,
            )
        else:
            normalized_action = action

        cap = self._cap(normalized_action.major or preview._transition_major)
        origin = preview._transition_origin or self._numeric_values(preview)

        trust_delta = 0.0
        closeness_delta = 0.0
        tension_delta = 0.0
        reciprocity_delta = 0.0
        repair_debt_delta = 0.0

        fulfilled = [
            item
            for item in normalized_action.fulfilled_commitment_ids
            if item in preview.active_commitment_ids
        ]
        resolved = [
            item
            for item in normalized_action.resolved_memory_ids
            if item in preview.unresolved_memory_ids
        ]

        if normalized_action.intent == ActionIntent.REPAIR:
            has_repair_target = bool(resolved) or preview.tension > 0.0 or preview.repair_debt > 0.0
            if has_repair_target:
                trust_delta += cap * 0.17
                closeness_delta += cap * 0.10
                tension_delta -= cap * 0.33
                repair_debt_delta -= cap * 0.33
        elif normalized_action.intent == ActionIntent.PROJECT_WORK:
            closeness_delta += cap * 0.33
            reciprocity_delta += cap * 0.33
        elif normalized_action.intent == ActionIntent.SHARE:
            closeness_delta += cap * 0.17
            reciprocity_delta += cap * 0.17
        elif normalized_action.intent == ActionIntent.COMFORT:
            closeness_delta += cap * 0.10
            reciprocity_delta += cap * 0.10
        elif normalized_action.intent in {
            ActionIntent.ACKNOWLEDGE,
            ActionIntent.ANSWER,
            ActionIntent.ASK,
            ActionIntent.CHALLENGE,
            ActionIntent.INITIATE,
        }:
            reciprocity_delta += cap * 0.10

        if fulfilled:
            reciprocity_delta += cap * 0.67
        if resolved:
            trust_delta += cap * 0.17
            tension_delta -= cap * 0.67
            repair_debt_delta -= cap * 0.67

        commitments = _remove_ids(preview.active_commitment_ids, fulfilled)
        unresolved = _remove_ids(preview.unresolved_memory_ids, resolved)
        rituals = _add_unique(preview.shared_ritual_ids, normalized_action.shared_ritual_ids)

        return RelationshipState(
            version=preview.version + 1,
            trust=_apply_transition_cap(preview.trust, trust_delta, origin[0], cap),
            closeness=_apply_transition_cap(preview.closeness, closeness_delta, origin[1], cap),
            tension=_apply_transition_cap(preview.tension, tension_delta, origin[2], cap),
            reciprocity=_apply_transition_cap(
                preview.reciprocity, reciprocity_delta, origin[3], cap
            ),
            repair_debt=_apply_transition_cap(
                preview.repair_debt, repair_debt_delta, origin[4], cap
            ),
            shared_ritual_ids=rituals,
            active_commitment_ids=commitments,
            unresolved_memory_ids=unresolved,
            updated_at_ms=preview.updated_at_ms,
        )

    def _cap(self, major: bool) -> float:
        return self._config.major_delta_cap if major else self._config.normal_delta_cap

    @staticmethod
    def _numeric_values(
        state: RelationshipState,
    ) -> tuple[float, float, float, float, float]:
        return (
            state.trust,
            state.closeness,
            state.tension,
            state.reciprocity,
            state.repair_debt,
        )

    @staticmethod
    def _kind_from_metadata(
        metadata: Mapping[str, object],
    ) -> RelationshipEventKind | None:
        raw_kind = metadata.get("relationship_kind")
        if not isinstance(raw_kind, str):
            return None
        try:
            return RelationshipEventKind(raw_kind)
        except ValueError:
            return None

    @staticmethod
    def _metadata_ids(
        metadata: Mapping[str, object],
        *keys: str,
    ) -> list[str]:
        result: list[str] = []
        known: set[str] = set()
        for key in keys:
            raw_value = metadata.get(key)
            candidates: list[object]
            if isinstance(raw_value, str):
                candidates = [raw_value]
            elif isinstance(raw_value, list):
                candidates = list(raw_value)
            else:
                continue
            for candidate in candidates:
                if isinstance(candidate, str) and candidate.strip() and candidate not in known:
                    result.append(candidate)
                    known.add(candidate)
        return result


__all__ = ["RelationshipService"]
