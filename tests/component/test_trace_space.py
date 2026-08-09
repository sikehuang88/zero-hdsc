"""Component tests for append-only trace writing and spatial activation."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssa.adapters.embedding import FakeEmbeddingService
from ssa.clock import FrozenClock
from ssa.config import DatabaseConfig, EmbeddingConfig, RetrievalConfig
from ssa.domain.appraisal import AppraisalResult
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, compute_content_hash
from ssa.domain.traces import TraceLink, TraceLinkType
from ssa.ids import SequentialIdGenerator
from ssa.services.trace_space_service import TraceSpaceService
from ssa.storage.database import Database
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.trace_repository import SqliteTraceRepository


def _event(
    event_id: str,
    actor: Actor,
    content: str,
    *,
    correlation_id: str,
    parent_event_id: str | None = None,
) -> Event:
    return Event(
        id=event_id,
        correlation_id=correlation_id,
        conversation_id="trace-test",
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


def test_trace_write_activation_radiation_projection_and_audit(tmp_path: Path) -> None:
    database = Database(DatabaseConfig(path=str(tmp_path / "traces.db")))
    database.initialize()
    repository = SqliteTraceRepository(database.connection)
    events = SqliteEventRepository(database.connection)
    clock = FrozenClock(1_900_000_000_000)
    ids = SequentialIdGenerator("trace")
    embedding_config = EmbeddingConfig(model="fake", dim=512)
    service = TraceSpaceService(
        embedding=FakeEmbeddingService(embedding_config),
        repository=repository,
        clock=clock,
        ids=ids,
        config=RetrievalConfig(),
        embedding_model="fake",
        embedding_dim=512,
    )

    try:
        first_user = events.append(
            _event("user-1", Actor.USER, "I named you SO", correlation_id="corr-1")
        )
        first_agent = events.append(
            _event(
                "agent-1",
                Actor.AGENT,
                "I will remember the name SO.",
                correlation_id="corr-1",
                parent_event_id=first_user.id,
            )
        )
        zero_novelty = AppraisalResult.neutral().model_copy(update={"novelty": 0.0})
        first = service.write_turn(first_user, first_agent, zero_novelty)
        clock.advance_ms(60_000)

        second_user = events.append(
            _event("user-2", Actor.USER, "We discussed identity", correlation_id="corr-2")
        )
        second_agent = events.append(
            _event(
                "agent-2",
                Actor.AGENT,
                "Identity should change slowly.",
                correlation_id="corr-2",
                parent_event_id=second_user.id,
            )
        )
        second = service.write_turn(second_user, second_agent, zero_novelty)
        repository.add_link(
            TraceLink(
                source_trace_id=first.id,
                target_trace_id=second.id,
                weight=1.0,
                created_at_ms=clock.now_ms(),
            )
        )

        activated = service.activate(
            first.content,
            conversation_id="trace-test",
            max_main=1,
        )
        assert activated[0].trace.id == first.id
        assert activated[0].activation_kind == "main"
        assert any(item.activation_kind == "radiation" for item in activated)

        service.record_activation(
            conversation_id="trace-test",
            correlation_id="corr-2",
            query_event_id=second_user.id,
            activations=activated,
        )
        replayed = repository.latest_activations("trace-test")
        assert [item.trace.id for item in replayed] == [item.trace.id for item in activated]

        snapshot = service.snapshot("trace-test", latest_query=first.content)
        assert snapshot.total_traces == 2
        assert snapshot.activated_count == len(activated)
        assert snapshot.serving_model_id == "legacy-ssa-a0"
        assert snapshot.shadow_model_id == "hdsc-h2-bounded-active-shadow"
        assert snapshot.shadow_affects_prompt is False
        assert snapshot.shadow_status == "passed"
        assert snapshot.stability_audit is not None
        assert snapshot.stability_audit.evaluated_archive_count == 2
        assert snapshot.stability_audit.active_count is not None
        assert snapshot.stability_audit.active_count <= snapshot.stability_audit.active_capacity
        assert snapshot.stability_audit.mass_residual == pytest.approx(0.0, abs=1e-10)
        assert snapshot.stability_audit.closed_loop_gate == "not-measured"
        live_h2_state = service._h2_states["trace-test"]
        service.replay_h2_shadow("trace-test")
        assert service._h2_states["trace-test"] == live_h2_state
        assert snapshot.projection == "embedding-pca"
        assert len(snapshot.nodes) == 2
        assert snapshot.links
        temporal = [link for link in snapshot.links if link.link_type == TraceLinkType.TEMPORAL]
        assert (first.id, second.id) in [
            (link.source_trace_id, link.target_trace_id) for link in temporal
        ]
        assert repository.links_from(first.id, link_types=("semantic",))
        assert repository.links_from(second.id, link_types=("temporal",))
        node_ids, directed_rates = service.directed_rate_matrix("trace-test")
        first_index = node_ids.index(first.id)
        second_index = node_ids.index(second.id)
        assert directed_rates[second_index, first_index] > 0.0
        assert directed_rates[first_index, second_index] > 0.0
        assert directed_rates[second_index, first_index] != pytest.approx(
            directed_rates[first_index, second_index]
        )
        assert all(0.0 <= node.x <= 1.0 and 0.0 <= node.y <= 1.0 for node in snapshot.nodes)
    finally:
        database.close()


def test_h2_snapshot_does_not_pass_with_an_unprocessed_archive_tail(tmp_path: Path) -> None:
    database = Database(DatabaseConfig(path=str(tmp_path / "watermark.db")))
    database.initialize()
    repository = SqliteTraceRepository(database.connection)
    events = SqliteEventRepository(database.connection)
    service = TraceSpaceService(
        embedding=FakeEmbeddingService(EmbeddingConfig(model="fake", dim=512)),
        repository=repository,
        clock=FrozenClock(1_900_000_000_000),
        ids=SequentialIdGenerator("trace"),
        config=RetrievalConfig(),
        embedding_model="fake",
        embedding_dim=512,
    )

    try:
        user = events.append(_event("user-tail", Actor.USER, "tail", correlation_id="tail"))
        agent = events.append(
            _event(
                "agent-tail",
                Actor.AGENT,
                "tail reply",
                correlation_id="tail",
                parent_event_id=user.id,
            )
        )
        service.write_turn(
            user,
            agent,
            AppraisalResult.neutral(),
            advance_shadow=False,
        )

        snapshot = service.snapshot("trace-test")
        assert snapshot.total_traces == 1
        assert snapshot.shadow_status == "awaiting-replay"
        assert snapshot.stability_audit is None

        service.replay_h2_shadow("trace-test")
        replayed = service.snapshot("trace-test")
        assert replayed.shadow_status == "passed"
        assert replayed.stability_audit is not None
        assert replayed.stability_audit.evaluated_archive_count == 1
    finally:
        database.close()


def test_h2_online_repository_failure_does_not_abort_trace_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Database(DatabaseConfig(path=str(tmp_path / "shadow-failure.db")))
    database.initialize()
    repository = SqliteTraceRepository(database.connection)
    events = SqliteEventRepository(database.connection)
    service = TraceSpaceService(
        embedding=FakeEmbeddingService(EmbeddingConfig(model="fake", dim=512)),
        repository=repository,
        clock=FrozenClock(1_900_000_000_000),
        ids=SequentialIdGenerator("trace"),
        config=RetrievalConfig(),
        embedding_model="fake",
        embedding_dim=512,
    )

    try:
        user = events.append(_event("user-fail", Actor.USER, "hello", correlation_id="fail"))
        agent = events.append(
            _event(
                "agent-fail",
                Actor.AGENT,
                "reply",
                correlation_id="fail",
                parent_event_id=user.id,
            )
        )

        with monkeypatch.context() as scoped:
            scoped.setattr(
                repository,
                "count",
                lambda _conversation_id=None: (_ for _ in ()).throw(RuntimeError("count")),
            )
            written = service.write_turn(user, agent, AppraisalResult.neutral())

        assert repository.get(written.id) == written
        snapshot = service.snapshot("trace-test")
        assert snapshot.shadow_status == "failed"
        assert snapshot.stability_audit is not None
        assert "RuntimeError: count" in snapshot.stability_audit.certificate_reason
        assert service.activate(written.content, conversation_id="trace-test")
    finally:
        database.close()


def test_h2_replay_failure_and_missing_vector_are_reported_in_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Database(DatabaseConfig(path=str(tmp_path / "replay-failure.db")))
    database.initialize()
    repository = SqliteTraceRepository(database.connection)
    events = SqliteEventRepository(database.connection)
    service = TraceSpaceService(
        embedding=FakeEmbeddingService(EmbeddingConfig(model="fake", dim=512)),
        repository=repository,
        clock=FrozenClock(1_900_000_000_000),
        ids=SequentialIdGenerator("trace"),
        config=RetrievalConfig(),
        embedding_model="fake",
        embedding_dim=512,
    )

    try:
        user = events.append(_event("user-replay", Actor.USER, "remember", correlation_id="replay"))
        agent = events.append(
            _event(
                "agent-replay",
                Actor.AGENT,
                "remembered",
                correlation_id="replay",
                parent_event_id=user.id,
            )
        )
        trace = service.write_turn(user, agent, AppraisalResult.neutral())

        with monkeypatch.context() as scoped:
            scoped.setattr(
                repository,
                "all_with_embeddings",
                lambda _conversation_id: (_ for _ in ()).throw(RuntimeError("replay")),
            )
            service.replay_h2_shadow("trace-test")

        failed = service.snapshot("trace-test")
        assert failed.shadow_status == "failed"
        assert service.activate(trace.content, conversation_id="trace-test")

        database.connection.execute("DELETE FROM trace_vec WHERE rowid = ?", (trace.vec_rowid,))
        service.replay_h2_shadow("trace-test")
        missing = service.snapshot("trace-test")
        assert missing.shadow_status == "failed"
        assert missing.stability_audit is not None
        assert "one persisted embedding" in missing.stability_audit.certificate_reason
    finally:
        database.close()


def test_h2_replay_rejects_a_changed_semantic_partition(tmp_path: Path) -> None:
    database = Database(DatabaseConfig(path=str(tmp_path / "partition.db")))
    database.initialize()
    repository = SqliteTraceRepository(database.connection)
    events = SqliteEventRepository(database.connection)
    base = TraceSpaceService(
        embedding=FakeEmbeddingService(EmbeddingConfig(model="fake", dim=512)),
        repository=repository,
        clock=FrozenClock(1_900_000_000_000),
        ids=SequentialIdGenerator("trace"),
        config=RetrievalConfig(),
        embedding_model="fake",
        embedding_dim=512,
    )

    try:
        user = events.append(
            _event("user-partition", Actor.USER, "partition", correlation_id="partition")
        )
        agent = events.append(
            _event(
                "agent-partition",
                Actor.AGENT,
                "partition reply",
                correlation_id="partition",
                parent_event_id=user.id,
            )
        )
        base.write_turn(user, agent, AppraisalResult.neutral())

        changed = TraceSpaceService(
            embedding=FakeEmbeddingService(EmbeddingConfig(model="fake-v2", dim=512)),
            repository=repository,
            clock=FrozenClock(1_900_000_000_000),
            ids=SequentialIdGenerator("changed"),
            config=RetrievalConfig(),
            embedding_model="fake-v2",
            embedding_dim=512,
        )
        changed.replay_h2_shadow("trace-test")
        snapshot = changed.snapshot("trace-test")

        assert snapshot.shadow_status == "failed"
        assert snapshot.stability_audit is not None
        assert "partition does not match" in snapshot.stability_audit.certificate_reason
    finally:
        database.close()
