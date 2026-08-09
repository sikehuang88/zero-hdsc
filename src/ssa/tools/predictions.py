"""Model tool for emitting structured, machine-verifiable predictions."""

from __future__ import annotations

from typing import Any

from ssa.domain.predictions import PredictionVerifierKind
from ssa.services.prediction_service import GroundedPredictionService
from ssa.tools.models import GroundedPredictionArguments, ToolAutonomyContext, ToolOutcome
from ssa.tools.registry import ToolCapability, ToolRegistry

_MINUTE_MS = 60_000


class GroundedPredictionExecutor:
    def __init__(self, service: GroundedPredictionService) -> None:
        self._service = service

    def register_into(self, registry: ToolRegistry) -> None:
        registry.register(
            ToolCapability(
                name="record_grounded_prediction",
                description=(
                    "Record a concrete future claim only when it can be checked by immutable "
                    "events. Supply actor/type/content/metadata constraints as a machine-readable "
                    "predicate, a calibrated confidence, a baseline rate, and a finite window."
                ),
                arguments_model=GroundedPredictionArguments,
                handler=self.record,
            )
        )

    async def record(
        self,
        raw_arguments: dict[str, Any],
        context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = GroundedPredictionArguments.model_validate(raw_arguments)
        now_ms = self._service.now_ms()
        prediction = self._service.record(
            conversation_id=context.conversation_id,
            claim_text=arguments.claim_text,
            claim_kind=arguments.claim_kind,
            verifier_kind=PredictionVerifierKind.EVENT_MATCH,
            verifier_spec=arguments.verifier_spec.model_dump(mode="json", exclude_none=True),
            stated_confidence=arguments.stated_confidence,
            base_rate_prior=arguments.base_rate_prior,
            resolve_after_ms=now_ms + arguments.resolve_after_minutes * _MINUTE_MS,
            expires_at_ms=now_ms + arguments.expires_after_minutes * _MINUTE_MS,
            source_event_ids=self._service.source_event_ids_for_correlation(
                context.conversation_id,
                context.correlation_id,
            ),
        )
        return ToolOutcome(
            ok=True,
            output=f"recorded grounded prediction {prediction.id}",
            metadata={
                "prediction_id": prediction.id,
                "status": prediction.status.value,
                "resolve_after_ms": prediction.resolve_after_ms,
                "expires_at_ms": prediction.expires_at_ms,
            },
        )


__all__ = ["GroundedPredictionExecutor"]
