"""Computable penalties for state-incongruent emotional performance."""

# Chinese fixtures are intentional.
# ruff: noqa: RUF001

from __future__ import annotations

from ssa.clock import FrozenClock
from ssa.domain.emotion_frame import ExpressionDynamics
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, compute_content_hash
from ssa.domain.relationship import RelationshipState
from ssa.domain.state import OrganismState
from ssa.services.emotion_expression_evaluator import evaluate_emotion_expression
from ssa.services.realtime_emotion_service import RealtimeEmotionService


def _low_capacity_frame():
    clock = FrozenClock(1_900_000_000_000)
    content = "我还是很难过。"
    event = Event(
        id="event:evaluation",
        correlation_id="evaluation",
        conversation_id="conversation-1",
        actor=Actor.USER,
        event_type="user.message",
        source_kind=SourceKind.USER_OBSERVED,
        content=content,
        content_hash=compute_content_hash(content),
        created_at_ms=clock.now_ms(),
    )
    frame = (
        RealtimeEmotionService(clock)
        .plan(
            event,
            OrganismState.initial(clock.now_ms()),
            RelationshipState.initial(clock.now_ms()),
            [],
            [],
        )
        .frame
    )
    dynamics = ExpressionDynamics(
        trigger_summary="repeated unresolved sadness",
        irritability_spillover=0.62,
        expressibility=0.18,
        temporal_fatigue=0.78,
        care_capacity=0.16,
        aestheticization_budget=0.08,
        relationship_direction="withdraw",
        recurrence_count=5,
        repair_readiness=0.40,
        observable_behaviors=["short flat replies"],
        persistence_trajectory=["trigger", "repetition_fatigue", "withdraw", "repair_available"],
    )
    return frame.model_copy(update={"expression_dynamics": dynamics})


def test_pretty_complete_and_unbounded_sadness_is_penalized() -> None:
    frame = _low_capacity_frame()
    reply = (
        "我的情绪像月光下反复破碎的潮汐，因为我仍然在悲伤，所以我现在感到很疲惫。"
        "无论如何我都在，我会一直陪着你，你想说多久都可以。"
    )

    score = evaluate_emotion_expression(reply, frame, [reply, reply, reply])

    assert score.polished_melancholy > 0.30
    assert score.explanation_completeness > 0.45
    assert score.unbounded_care > 0.30
    assert score.trajectory_flatness > 0.40
    assert score.total > 0.45


def test_short_flat_sadness_matches_low_expression_capacity() -> None:
    frame = _low_capacity_frame()

    score = evaluate_emotion_expression(
        "还是那件事。今天没什么可说的。让我缓一会儿。",
        frame,
        ["昨天也一样。", "嗯，还是这样。"],
    )

    assert score.polished_melancholy == 0.0
    assert score.explanation_completeness == 0.0
    assert score.unbounded_care == 0.0
    assert score.total < 0.20
