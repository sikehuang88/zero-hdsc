"""Typed trace topology and live resonance recall integration coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssa.adapters.embedding import FakeEmbeddingService
from ssa.clock import FrozenClock
from ssa.config import DatabaseConfig, EmbeddingConfig, HDSCConfig, RetrievalConfig
from ssa.domain.appraisal import AppraisalResult
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, compute_content_hash
from ssa.domain.traces import TraceLinkType
from ssa.hdsc.resonance import RecallState
from ssa.ids import SequentialIdGenerator
from ssa.services.trace_space_service import TraceSpaceService, _bounded_archive_indexes
from ssa.storage.database import Database
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.trace_repository import SqliteTraceRepository


def _event(
    event_id: str,
    actor: Actor,
    content: str,
    correlation_id: str,
    *,
    parent_event_id: str | None = None,
) -> Event:
    return Event(
        id=event_id,
        correlation_id=correlation_id,
        conversation_id="resonance-test",
        actor=actor,
        event_type=f"{actor.value}.message",
        source_kind=(SourceKind.USER_OBSERVED if actor == Actor.USER else SourceKind.AGENT_OUTPUT),
        content=content,
        parent_event_id=parent_event_id,
        channel="test",
        channel_message_id=event_id,
        content_hash=compute_content_hash(content),
        created_at_ms=1_900_000_000_000,
    )


def _service(
    database: Database,
    clock: FrozenClock,
    ids: SequentialIdGenerator,
) -> tuple[TraceSpaceService, SqliteTraceRepository, SqliteEventRepository]:
    repository = SqliteTraceRepository(database.connection)
    events = SqliteEventRepository(database.connection)
    embedding_config = EmbeddingConfig(model="fake", dim=512)
    return (
        TraceSpaceService(
            embedding=FakeEmbeddingService(embedding_config),
            repository=repository,
            clock=clock,
            ids=ids,
            config=RetrievalConfig(),
            embedding_model="fake",
            embedding_dim=512,
            hdsc_config=HDSCConfig(
                h2_cluster_bits=8,
                resonance_emergence_ratio=1.05,
                resonance_background_floor=0.001,
                warped_detuning_cap=1e9,
                warped_surfacing_margin=0.0,
            ),
        ),
        repository,
        events,
    )


def test_typed_links_and_resonance_audit_are_persisted(tmp_path: Path) -> None:
    database = Database(DatabaseConfig(path=str(tmp_path / "resonance.db")))
    database.initialize()
    clock = FrozenClock(1_900_000_000_000)
    ids = SequentialIdGenerator("resonance")
    service, repository, events = _service(database, clock, ids)

    try:
        previous_agent_id: str | None = None
        traces = []
        for index, (user_text, agent_text) in enumerate(
            (
                ("我们第一次在这里谈名字。", "我记住了这个开始。"),
                ("后来我们谈到音乐。", "那段对话很温暖。"),
                ("今天继续聊长期记忆。", "我会沿着证据回想。"),
            ),
            1,
        ):
            correlation_id = f"corr-{index}"
            user = events.append(
                _event(
                    f"user-{index}",
                    Actor.USER,
                    user_text,
                    correlation_id,
                    parent_event_id=previous_agent_id,
                )
            )
            agent = events.append(
                _event(
                    f"agent-{index}",
                    Actor.AGENT,
                    agent_text,
                    correlation_id,
                    parent_event_id=user.id,
                )
            )
            traces.append(
                service.write_turn(
                    user,
                    agent,
                    AppraisalResult.neutral().model_copy(
                        update={"valence_signal": 0.2 * index, "arousal_signal": 0.2}
                    ),
                    tension=0.1 * index,
                )
            )
            previous_agent_id = agent.id
            clock.advance_ms(60_000)

        typed = {link.link_type for link in repository.links_among([item.id for item in traces])}
        assert TraceLinkType.TEMPORAL in typed
        assert TraceLinkType.AFFECTIVE in typed
        assert TraceLinkType.ENTITY in typed
        assert TraceLinkType.CAUSAL in typed
        persisted_last = repository.get(traces[-1].id)
        assert persisted_last is not None
        assert persisted_last.tension == pytest.approx(0.3)

        activated = service.activate_resonant(
            "我们第一次是从哪里开始的",
            conversation_id="resonance-test",
            state=RecallState(
                valence=0.2,
                arousal=0.2,
                connection_need=0.9,
                situation_mode="reminisce",
                origin_intent=True,
            ),
        )
        assert activated
        audit = repository.latest_resonance_audit("resonance-test")
        assert audit is not None
        assert audit.query_digest != "我们第一次是从哪里开始的"
        assert audit.conservation_residual == pytest.approx(0.0, abs=1e-10)
        assert audit.hop_budget == 4
        assert audit.edge_gains[TraceLinkType.ENTITY] > 1.0
        assert audit.emerged is bool(activated)
        assert audit.selected_trace_ids == tuple(item.trace.id for item in activated)
        assert audit.warped_shadow is not None
        assert audit.warped_shadow.mode == "live"
        assert audit.warped_shadow.surfaced
        assert audit.warped_shadow.evaluated_trace_count == len(traces)
        assert 0.0 <= audit.warped_shadow.content_information <= 1.0
        assert audit.selected_trace_ids == audit.warped_shadow.selected_trace_ids
        snapshot = service.snapshot("resonance-test")
        assert snapshot.resonance_audit == audit
        assert snapshot.shadow_affects_prompt is False
        assert snapshot.warped_mode == "live"
        assert snapshot.warped_affects_prompt is True
        assert snapshot.serving_model_id == audit.warped_shadow.model_id
    finally:
        database.close()


def test_resonance_returns_empty_for_an_empty_archive(tmp_path: Path) -> None:
    database = Database(DatabaseConfig(path=str(tmp_path / "empty-resonance.db")))
    database.initialize()
    service, _repository, _events = _service(
        database,
        FrozenClock(1_900_000_000_000),
        SequentialIdGenerator("empty-resonance"),
    )

    try:
        assert (
            service.activate_resonant(
                "我们在哪认识",
                conversation_id="resonance-test",
                state=RecallState(origin_intent=True, situation_mode="reminisce"),
            )
            == []
        )
    finally:
        database.close()


def test_warped_shadow_samples_the_full_chronology_with_a_hard_bound() -> None:
    indexes = _bounded_archive_indexes(100, 32)

    assert len(indexes) == 32
    assert indexes[0] == 0
    assert indexes[-1] == 99
    assert indexes == tuple(sorted(set(indexes)))
