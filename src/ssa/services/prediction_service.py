"""Grounded prediction capture, deterministic resolution, and calibration reporting."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ssa.clock import Clock
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, IncomingSignal, normalize_signal
from ssa.domain.predictions import (
    EventMatchVerifierSpec,
    GroundedPrediction,
    PredictionCalibrationBucket,
    PredictionCalibrationReport,
    PredictionClaimKind,
    PredictionStatus,
    PredictionVerifierKind,
)
from ssa.ids import IdGenerator
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.prediction_repository import PredictionRepository


@dataclass(frozen=True)
class PredictionResolutionResult:
    predictions: tuple[GroundedPrediction, ...] = ()
    events: tuple[Event, ...] = ()


class GroundedPredictionService:
    """Own the objective prediction channel; no model judgment enters resolution."""

    def __init__(
        self,
        *,
        predictions: PredictionRepository,
        events: SqliteEventRepository,
        clock: Clock,
        ids: IdGenerator,
    ) -> None:
        self.repository = predictions
        self._events = events
        self._clock = clock
        self._ids = ids

    def now_ms(self) -> int:
        return self._clock.now_ms()

    def record(
        self,
        *,
        conversation_id: str,
        claim_text: str,
        claim_kind: PredictionClaimKind,
        verifier_kind: PredictionVerifierKind,
        verifier_spec: Mapping[str, Any],
        stated_confidence: float,
        resolve_after_ms: int,
        expires_at_ms: int,
        base_rate_prior: float = 0.5,
        source_event_ids: list[str] | None = None,
        source_trace_ids: list[str] | None = None,
        dedup_key: str | None = None,
    ) -> GroundedPrediction:
        source_events = list(source_event_ids or ())
        for event_id in source_events:
            event = self._events.get(event_id)
            if event is None or event.conversation_id != conversation_id:
                raise ValueError(
                    f"prediction source event {event_id!r} is outside the conversation"
                )
        canonical_spec = dict(verifier_spec)
        computed_dedup = dedup_key or _prediction_dedup_key(
            conversation_id=conversation_id,
            claim_text=claim_text,
            verifier_kind=verifier_kind,
            verifier_spec=canonical_spec,
            resolve_after_ms=resolve_after_ms,
            expires_at_ms=expires_at_ms,
        )
        prediction = GroundedPrediction(
            id=self._ids.new(),
            conversation_id=conversation_id,
            dedup_key=computed_dedup,
            claim_text=claim_text,
            claim_kind=claim_kind,
            verifier_kind=verifier_kind,
            verifier_spec=canonical_spec,
            stated_confidence=stated_confidence,
            base_rate_prior=base_rate_prior,
            source_event_ids=source_events,
            source_trace_ids=list(source_trace_ids or ()),
            created_at_ms=self._clock.now_ms(),
            resolve_after_ms=resolve_after_ms,
            expires_at_ms=expires_at_ms,
        )
        return self.repository.insert(prediction)

    def source_event_ids_for_correlation(
        self,
        conversation_id: str,
        correlation_id: str,
    ) -> list[str]:
        return [
            event.id
            for event in self._events.find_by_correlation(correlation_id)
            if event.conversation_id == conversation_id
        ]

    def resolve_due(
        self,
        conversation_id: str,
        *,
        limit: int = 100,
    ) -> PredictionResolutionResult:
        now_ms = self._clock.now_ms()
        resolved: list[GroundedPrediction] = []
        emitted: list[Event] = []
        for prediction in self.repository.list_due(
            now_ms,
            conversation_id=conversation_id,
            limit=limit,
        ):
            terminal: GroundedPrediction | None = None
            if prediction.verifier_kind == PredictionVerifierKind.EVENT_MATCH:
                spec = EventMatchVerifierSpec.model_validate(prediction.verifier_spec)
                candidates = self._events.between_by_conversation(
                    prediction.conversation_id,
                    start_ms=prediction.resolve_after_ms,
                    end_ms=min(now_ms, prediction.expires_at_ms),
                )
                match = next(
                    (
                        event
                        for event in candidates
                        if event.id not in prediction.source_event_ids
                        and _event_matches(event, spec)
                    ),
                    None,
                )
                if match is not None:
                    terminal = self.repository.resolve(
                        prediction,
                        status=PredictionStatus.RESOLVED_TRUE,
                        resolved_at_ms=now_ms,
                        resolved_by_event_id=match.id,
                    )
                elif now_ms >= prediction.expires_at_ms:
                    terminal = self.repository.resolve(
                        prediction,
                        status=PredictionStatus.RESOLVED_FALSE,
                        resolved_at_ms=now_ms,
                    )
            elif now_ms >= prediction.expires_at_ms:
                # These specifications stay explicit and auditable until their deterministic
                # provider adapters are connected. No LLM is used as a substitute verifier.
                terminal = self.repository.resolve(
                    prediction,
                    status=PredictionStatus.UNVERIFIABLE,
                    resolved_at_ms=now_ms,
                )

            if terminal is None:
                continue
            resolved.append(terminal)
            emitted.append(self._append_resolution_event(terminal))
        return PredictionResolutionResult(tuple(resolved), tuple(emitted))

    def calibration_report(self, conversation_id: str) -> PredictionCalibrationReport:
        predictions = self.repository.resolved(conversation_id)
        counts = self.repository.status_counts(conversation_id)
        buckets: list[PredictionCalibrationBucket] = []
        for index in range(10):
            lower = index / 10.0
            upper = (index + 1) / 10.0
            members = [
                prediction
                for prediction in predictions
                if min(int(prediction.stated_confidence * 10), 9) == index
            ]
            if not members:
                buckets.append(
                    PredictionCalibrationBucket(
                        lower_bound=lower,
                        upper_bound=upper,
                        count=0,
                    )
                )
                continue
            outcomes = [
                1.0 if item.status == PredictionStatus.RESOLVED_TRUE else 0.0 for item in members
            ]
            buckets.append(
                PredictionCalibrationBucket(
                    lower_bound=lower,
                    upper_bound=upper,
                    count=len(members),
                    mean_confidence=sum(item.stated_confidence for item in members) / len(members),
                    observed_rate=sum(outcomes) / len(outcomes),
                    brier_score=sum(float(item.brier_contribution or 0.0) for item in members)
                    / len(members),
                )
            )
        if predictions:
            brier = sum(float(item.brier_contribution or 0.0) for item in predictions) / len(
                predictions
            )
            baseline = sum(
                (
                    item.base_rate_prior
                    - (1.0 if item.status == PredictionStatus.RESOLVED_TRUE else 0.0)
                )
                ** 2
                for item in predictions
            ) / len(predictions)
            skill = 1.0 - brier / baseline if baseline > 0.0 else None
        else:
            brier = None
            baseline = None
            skill = None
        return PredictionCalibrationReport(
            conversation_id=conversation_id,
            resolved_count=len(predictions),
            pending_count=counts.get(PredictionStatus.PENDING, 0),
            unverifiable_count=(
                counts.get(PredictionStatus.UNVERIFIABLE, 0)
                + counts.get(PredictionStatus.EXPIRED, 0)
            ),
            brier_score=brier,
            baseline_brier_score=baseline,
            brier_skill_score=skill,
            buckets=tuple(buckets),
        )

    def _append_resolution_event(self, prediction: GroundedPrediction) -> Event:
        existing = self._events.find_by_channel_message(
            "prediction",
            f"{prediction.id}:{prediction.status.value}",
        )
        if existing is not None:
            return existing
        return self._events.append(
            normalize_signal(
                IncomingSignal(
                    actor=Actor.SYSTEM,
                    signal_type="learning.prediction_resolved",
                    content=(
                        f"A grounded prediction resolved as {prediction.status.value}: "
                        f"{prediction.claim_text}"
                    ),
                    channel="prediction",
                    channel_message_id=f"{prediction.id}:{prediction.status.value}",
                    conversation_id=prediction.conversation_id,
                    metadata={
                        "prediction_id": prediction.id,
                        "status": prediction.status.value,
                        "brier_contribution": prediction.brier_contribution,
                        "resolved_by_event_id": prediction.resolved_by_event_id,
                    },
                ),
                event_id=self._ids.new(),
                correlation_id=self._ids.new(),
                now_ms=self._clock.now_ms(),
                source_kind=SourceKind.SYSTEM_DERIVED,
            )
        )


def _event_matches(event: Event, spec: EventMatchVerifierSpec) -> bool:
    if spec.actor is not None and event.actor != spec.actor:
        return False
    if spec.source_kind is not None and event.source_kind != spec.source_kind:
        return False
    if spec.event_type is not None and event.event_type != spec.event_type:
        return False
    content = event.content.casefold()
    if spec.content_equals is not None and content != spec.content_equals.casefold():
        return False
    if spec.content_contains_all and not all(
        term.casefold() in content for term in spec.content_contains_all
    ):
        return False
    if spec.content_contains_any and not any(
        term.casefold() in content for term in spec.content_contains_any
    ):
        return False
    for path, expected in spec.metadata_equals.items():
        if _metadata_value(event.metadata, path) != expected:
            return False
    return True


def _metadata_value(metadata: Mapping[str, Any], path: str) -> Any:
    current: Any = metadata
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return object()
        current = current[component]
    return current


def _prediction_dedup_key(
    *,
    conversation_id: str,
    claim_text: str,
    verifier_kind: PredictionVerifierKind,
    verifier_spec: Mapping[str, Any],
    resolve_after_ms: int,
    expires_at_ms: int,
) -> str:
    payload = json.dumps(
        {
            "conversation_id": conversation_id,
            "claim_text": claim_text.strip(),
            "verifier_kind": verifier_kind.value,
            "verifier_spec": dict(verifier_spec),
            "resolve_after_ms": resolve_after_ms,
            "expires_at_ms": expires_at_ms,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"prediction:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


__all__ = [
    "GroundedPredictionService",
    "PredictionResolutionResult",
]
