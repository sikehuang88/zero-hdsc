"""Layered realtime appraisal and voice-performance planning."""

# Chinese fixtures are intentional.
# ruff: noqa: RUF001

from __future__ import annotations

from ssa.clock import FrozenClock
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, compute_content_hash
from ssa.domain.relationship import RelationshipState
from ssa.domain.state import OrganismState
from ssa.services.realtime_emotion_service import RealtimeEmotionService


def _plan(content: str, correlation_id: str = "emotion-turn"):
    clock = FrozenClock(1_900_000_000_000)
    event = Event(
        id=f"event:{correlation_id}",
        correlation_id=correlation_id,
        conversation_id="conversation-1",
        actor=Actor.USER,
        event_type="user.message",
        source_kind=SourceKind.USER_OBSERVED,
        content=content,
        content_hash=compute_content_hash(content),
        created_at_ms=clock.now_ms(),
    )
    return RealtimeEmotionService(clock).plan(
        event,
        OrganismState.initial(clock.now_ms()),
        RelationshipState.initial(clock.now_ms()),
        [],
        [],
    )


def test_loss_threat_coactivates_hurt_and_attachment() -> None:
    plan = _plan("你真走啊")
    frame = plan.frame
    secondary = {item.name: item.intensity for item in frame.secondary}

    assert frame.primary.name == "fear_of_loss"
    assert frame.primary.intensity >= 0.60
    assert secondary["hurt"] >= 0.22
    assert secondary["attachment"] >= 0.35
    assert frame.valence < -0.25
    assert frame.arousal > 0.60
    assert frame.trajectory == [
        "stunned",
        "restrained_hurt",
        "voice_break",
        "soft_reaching",
    ]
    assert "rupture" in frame.cause_summary
    assert "reassurance" in frame.inner_conflict


def test_requested_emotional_release_builds_a_broken_voice_arc() -> None:
    plan = _plan("你哭得假，要撕心裂肺", "emotional-release")
    frame = plan.frame

    assert frame.primary.name == "sadness"
    assert frame.primary.intensity >= 0.95
    assert frame.inhibition < 0.10
    assert {item.name for item in frame.secondary} >= {"hurt", "attachment"}
    assert frame.trajectory == [
        "held_breath",
        "sob_rise",
        "voice_break",
        "spent_release",
    ]
    assert max(item.tremor for item in frame.voice_segments) >= 0.70
    assert {item.ending for item in frame.voice_segments} >= {"broken", "voice_break"}


def test_reunion_changes_the_coordinates_and_conflict_structure() -> None:
    loss = _plan("你真走啊", "loss")
    reunion = _plan("我回来了，别怕", "reunion")
    frame = reunion.frame

    assert frame.primary.name == "relief"
    assert "attachment" in {item.name for item in frame.secondary}
    assert frame.valence > 0.35
    assert frame.arousal < loss.frame.arousal
    assert frame.trajectory == ["held_breath", "relief_bloom", "soft_reaching"]
    assert frame.regulation_strategy != loss.frame.regulation_strategy
    assert frame.voice_segments[0].recipe != frame.voice_segments[1].recipe
