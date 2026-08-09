"""Persistence, deterministic resolution, and calibration for grounded predictions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ssa.adapters.llm import FunctionCall, ToolCall
from ssa.clock import FrozenClock
from ssa.config import DatabaseConfig
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, IncomingSignal, normalize_signal
from ssa.domain.predictions import (
    EventMatchVerifierSpec,
    PredictionClaimKind,
    PredictionStatus,
    PredictionVerifierKind,
)
from ssa.ids import SequentialIdGenerator
from ssa.services.prediction_service import GroundedPredictionService
from ssa.storage.database import Database
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.prediction_repository import PredictionRepository
from ssa.tools.kernel import ToolKernel
from ssa.tools.models import ToolAutonomyContext
from ssa.tools.predictions import GroundedPredictionExecutor
from ssa.tools.registry import ToolRegistry


def _setup(
    tmp_path: Path,
) -> tuple[
    Database,
    FrozenClock,
    SequentialIdGenerator,
    SqliteEventRepository,
    GroundedPredictionService,
]:
    database = Database(DatabaseConfig(path=str(tmp_path / "predictions.db")))
    database.initialize()
    clock = FrozenClock(2_000_000_000_000)
    ids = SequentialIdGenerator("prediction")
    events = SqliteEventRepository(database.connection)
    service = GroundedPredictionService(
        predictions=PredictionRepository(database.connection),
        events=events,
        clock=clock,
        ids=ids,
    )
    return database, clock, ids, events, service


def _event(
    events: SqliteEventRepository,
    ids: SequentialIdGenerator,
    clock: FrozenClock,
    content: str,
    *,
    actor: Actor = Actor.USER,
    correlation_id: str | None = None,
) -> Event:
    return events.append(
        normalize_signal(
            IncomingSignal(
                actor=actor,
                signal_type=f"{actor.value}.message",
                content=content,
                channel="prediction-fixture",
                channel_message_id=ids.new(),
                conversation_id="conversation-1",
            ),
            event_id=ids.new(),
            correlation_id=correlation_id or ids.new(),
            now_ms=clock.now_ms(),
            source_kind=(
                SourceKind.USER_OBSERVED if actor == Actor.USER else SourceKind.AGENT_OUTPUT
            ),
        )
    )


def test_event_match_predictions_resolve_true_and_false_with_calibration(
    tmp_path: Path,
) -> None:
    database, clock, ids, events, service = _setup(tmp_path)
    try:
        now_ms = clock.now_ms()
        positive = service.record(
            conversation_id="conversation-1",
            claim_text="The user will mention the deployment tomorrow.",
            claim_kind=PredictionClaimKind.USER_BEHAVIOR,
            verifier_kind=PredictionVerifierKind.EVENT_MATCH,
            verifier_spec={
                "actor": "user",
                "event_type": "user.message",
                "content_contains_all": ["deployment"],
            },
            stated_confidence=0.8,
            base_rate_prior=0.6,
            resolve_after_ms=now_ms + 100,
            expires_at_ms=now_ms + 1_000,
        )
        negative = service.record(
            conversation_id="conversation-1",
            claim_text="The user will mention an absent keyword.",
            claim_kind=PredictionClaimKind.USER_BEHAVIOR,
            verifier_kind=PredictionVerifierKind.EVENT_MATCH,
            verifier_spec={"actor": "user", "content_contains_all": ["ABSENT_KEYWORD"]},
            stated_confidence=0.7,
            base_rate_prior=0.2,
            resolve_after_ms=now_ms + 100,
            expires_at_ms=now_ms + 1_000,
        )

        assert service.resolve_due("conversation-1").predictions == ()
        clock.set_ms(now_ms + 200)
        matching = _event(events, ids, clock, "The deployment is ready.")
        first = service.resolve_due("conversation-1")

        assert first.predictions[0].id == positive.id
        assert first.predictions[0].status == PredictionStatus.RESOLVED_TRUE
        assert first.predictions[0].resolved_by_event_id == matching.id
        assert first.predictions[0].brier_contribution == pytest.approx(0.04)
        assert len(first.events) == 1

        clock.set_ms(now_ms + 1_000)
        second = service.resolve_due("conversation-1")

        assert second.predictions[0].id == negative.id
        assert second.predictions[0].status == PredictionStatus.RESOLVED_FALSE
        assert second.predictions[0].brier_contribution == pytest.approx(0.49)

        report = service.calibration_report("conversation-1")
        assert report.resolved_count == 2
        assert report.pending_count == 0
        assert report.brier_score == pytest.approx(0.265)
        assert report.baseline_brier_score == pytest.approx(0.10)
        assert report.brier_skill_score == pytest.approx(-1.65)
        assert len(report.buckets) == 10
        assert report.buckets[7].count == 1
        assert report.buckets[8].count == 1
    finally:
        database.close()


def test_verifier_requires_machine_checkable_constraints() -> None:
    with pytest.raises(ValueError, match="at least one structural constraint"):
        EventMatchVerifierSpec()


async def test_prediction_tool_records_structured_event_match(tmp_path: Path) -> None:
    database, clock, ids, events, service = _setup(tmp_path)
    try:
        correlation_id = "turn-correlation"
        source = _event(
            events,
            ids,
            clock,
            "I will report back tomorrow.",
            correlation_id=correlation_id,
        )
        registry = ToolRegistry()
        GroundedPredictionExecutor(service).register_into(registry)
        kernel = ToolKernel(registry)
        result = await kernel.execute(
            ToolCall(
                id="tool-prediction",
                function=FunctionCall(
                    name="record_grounded_prediction",
                    arguments=json.dumps(
                        {
                            "claim_text": "The user will report back.",
                            "claim_kind": "user_behavior",
                            "verifier_spec": {
                                "actor": "user",
                                "content_contains_any": ["report", "反馈"],
                            },
                            "stated_confidence": 0.75,
                            "base_rate_prior": 0.4,
                            "resolve_after_minutes": 60,
                            "expires_after_minutes": 2_880,
                        }
                    ),
                ),
            ),
            ToolAutonomyContext(
                correlation_id=correlation_id,
                conversation_id="conversation-1",
                energy=0.7,
                valence=0.1,
                arousal=0.3,
                trust=0.6,
                tension=0.1,
                situation_mode="chat",
                situation_confidence=0.8,
            ),
        )

        assert result.ok is True
        prediction = service.repository.get(str(result.metadata["prediction_id"]))
        assert prediction is not None
        assert prediction.source_event_ids == [source.id]
        assert prediction.verifier_kind == PredictionVerifierKind.EVENT_MATCH
    finally:
        database.close()
