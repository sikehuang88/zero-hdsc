"""Offline agency gates, persistence, evidence, and external-study coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ssa.adapters.embedding import FakeEmbeddingService
from ssa.adapters.llm import FakeLLMAdapter
from ssa.clock import FrozenClock
from ssa.config import DatabaseConfig, ExternalTruthConfig, OfflineAgencyConfig, Settings
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import IncomingSignal, normalize_signal
from ssa.domain.lifecycle import OfflineActionKind, OfflineEpisode, OfflineEpisodeStatus
from ssa.domain.relationship import RelationshipState
from ssa.domain.state import OrganismState
from ssa.domain.traces import Trace
from ssa.ids import SequentialIdGenerator
from ssa.services.offline_agency_service import OfflineAgencyService
from ssa.storage.database import Database
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.lifecycle_repository import (
    BackgroundUsageRepository,
    GoalRepository,
    OfflineAgencyRepository,
)
from ssa.storage.trace_repository import SqliteTraceRepository, trace_vec_rowid
from ssa.tools.external_truth import ExternalTruthExecutor, JsonResponse


class _TruthTransport:
    def __init__(self) -> None:
        self.calls = 0

    async def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout_seconds: int,
        max_response_bytes: int,
    ) -> JsonResponse:
        self.calls += 1
        return JsonResponse(
            data={
                "category": "ai",
                "articles": [
                    {
                        "headline": "A verifiable external development",
                        "url": "https://example.test/article",
                        "source": "fixture",
                    }
                ],
            },
            final_url=url,
            status_code=200,
        )


def _setup(
    tmp_path: Path,
    *,
    config: OfflineAgencyConfig | None = None,
) -> tuple[
    Database,
    FrozenClock,
    SequentialIdGenerator,
    FakeLLMAdapter,
    OfflineAgencyService,
    SqliteEventRepository,
    SqliteTraceRepository,
    OfflineAgencyRepository,
    _TruthTransport,
]:
    settings = Settings(
        database=DatabaseConfig(path=str(tmp_path / "offline.db")),
        offline_agency=config
        or OfflineAgencyConfig(min_user_absence_minutes=20, external_observation_every_cycles=4),
    )
    database = Database(settings.database)
    database.initialize()
    clock = FrozenClock(1_900_000_000_000)
    ids = SequentialIdGenerator("offline")
    llm = FakeLLMAdapter()
    events = SqliteEventRepository(database.connection)
    traces = SqliteTraceRepository(database.connection)
    episodes = OfflineAgencyRepository(database.connection)
    truth_transport = _TruthTransport()
    truth = ExternalTruthExecutor(
        ExternalTruthConfig(min_request_interval_ms=0),
        transport=truth_transport,
    )
    service = OfflineAgencyService(
        episodes=episodes,
        goals=GoalRepository(database.connection),
        traces=traces,
        events=events,
        usage=BackgroundUsageRepository(database.connection),
        external_truth=truth,
        llm=llm,
        clock=clock,
        ids=ids,
        config=settings.offline_agency,
        budget_config=settings.budget,
        llm_config=settings.llm,
        timezone="UTC",
    )
    return database, clock, ids, llm, service, events, traces, episodes, truth_transport


def _seed_user_and_trace(
    clock: FrozenClock,
    ids: SequentialIdGenerator,
    events: SqliteEventRepository,
    traces: SqliteTraceRepository,
    settings: Settings,
    *,
    age_minutes: int,
) -> Trace:
    event = events.append(
        normalize_signal(
            IncomingSignal(
                actor=Actor.USER,
                signal_type="user.message",
                content="Build a real evidence-grounded offline learning practice.",
                channel="fixture",
                channel_message_id=ids.new(),
                conversation_id="conversation-1",
            ),
            event_id=ids.new(),
            correlation_id=ids.new(),
            now_ms=clock.now_ms() - age_minutes * 60_000,
            source_kind=SourceKind.USER_OBSERVED,
        )
    )
    embedding = FakeEmbeddingService(settings.embedding)
    trace_id = ids.new()
    trace = Trace(
        id=trace_id,
        conversation_id="conversation-1",
        correlation_id=event.correlation_id,
        input_event_id=event.id,
        content=event.content,
        source_kind=SourceKind.SYSTEM_DERIVED,
        importance=0.9,
        valence=0.3,
        arousal=0.4,
        embedding_model=embedding.model_name,
        embedding_dim=embedding.dimension,
        vec_rowid=trace_vec_rowid(trace_id),
        created_at_ms=event.created_at_ms,
    )
    return traces.insert(trace, embedding.embed_one(trace.content).as_bytes())


@pytest.mark.asyncio
async def test_cycle_waits_until_user_is_absent(tmp_path: Path) -> None:
    database, clock, ids, _llm, service, events, traces, episodes, _truth = _setup(tmp_path)
    settings = Settings()
    try:
        _seed_user_and_trace(clock, ids, events, traces, settings, age_minutes=5)
        result = await service.run_cycle(
            "conversation-1",
            OrganismState.initial(clock.now_ms()),
            RelationshipState.initial(clock.now_ms()),
        )

        assert result.gate == "user_present"
        assert episodes.recent_episodes("conversation-1") == []
    finally:
        database.close()


@pytest.mark.asyncio
async def test_trace_reflection_persists_completed_episode_and_artifact(tmp_path: Path) -> None:
    database, clock, ids, llm, service, events, traces, _episodes, truth = _setup(tmp_path)
    settings = Settings()
    llm.set_response(
        "offline_learning_reflection",
        json.dumps(
            {
                "title": "Evidence before self-description",
                "summary": "Converted one archived request into an auditable learning note.",
                "learning_note": "Claims about offline growth require completed episode evidence.",
                "questions": ["Which action should be measured next?"],
                "next_action": "Compare the claim with the next completed episode.",
            }
        ),
    )
    try:
        trace = _seed_user_and_trace(clock, ids, events, traces, settings, age_minutes=30)
        result = await service.run_cycle(
            "conversation-1",
            OrganismState.initial(clock.now_ms()),
            RelationshipState.initial(clock.now_ms()),
        )

        assert result.gate == "completed"
        assert result.episode is not None
        assert result.episode.status == OfflineEpisodeStatus.COMPLETED
        assert result.episode.action_kind == OfflineActionKind.TRACE_REFLECTION
        assert result.episode.source_trace_ids == [trace.id]
        assert result.artifact is not None
        assert "Claims about offline growth" in result.artifact.content
        assert result.event is not None
        assert result.event.event_type == "lifecycle.offline_episode"
        assert truth.calls == 0
        context = service.context_summary("conversation-1")
        assert "action=trace_reflection; status=completed" in context
        assert f"artifact_id={result.artifact.id}" in context
        assert "Converted one archived request" in context
    finally:
        database.close()


def test_context_excludes_failed_and_unproven_episodes(tmp_path: Path) -> None:
    database, clock, ids, _llm, service, _events, _traces, episodes, _truth = _setup(tmp_path)
    try:
        for status in (OfflineEpisodeStatus.FAILED, OfflineEpisodeStatus.COMPLETED):
            episode_id = ids.new()
            episodes.insert_episode(
                OfflineEpisode(
                    id=episode_id,
                    conversation_id="conversation-1",
                    correlation_id=ids.new(),
                    dedup_key=f"offline:unproven:{episode_id}",
                    action_kind=OfflineActionKind.TRACE_REFLECTION,
                    motive="Fixture without a persisted artifact",
                    status=status,
                    summary="This summary must not become runtime evidence.",
                    error="fixture failure" if status == OfflineEpisodeStatus.FAILED else None,
                    started_at_ms=clock.now_ms(),
                    completed_at_ms=clock.now_ms(),
                )
            )

        context = service.context_summary("conversation-1")

        assert "no completed offline activity has been recorded" in context
        assert "This summary must not become runtime evidence" not in context
    finally:
        database.close()


@pytest.mark.asyncio
async def test_external_study_is_real_and_provider_grounded(tmp_path: Path) -> None:
    config = OfflineAgencyConfig(
        min_user_absence_minutes=1,
        external_observation_every_cycles=1,
    )
    database, clock, ids, _llm, service, events, traces, episodes, truth = _setup(
        tmp_path,
        config=config,
    )
    settings = Settings()
    try:
        trace = _seed_user_and_trace(clock, ids, events, traces, settings, age_minutes=30)
        result = await service.run_cycle(
            "conversation-1",
            OrganismState.initial(clock.now_ms()),
            RelationshipState.initial(clock.now_ms()),
        )

        assert result.gate == "completed"
        assert result.episode is not None
        assert result.episode.action_kind == OfflineActionKind.EXTERNAL_STUDY
        assert result.episode.provider == "noozra"
        assert result.episode.tool_name == "truth_news"
        assert result.episode.tool_output_hash is not None
        assert result.episode.source_trace_ids == []
        assert trace.id not in episodes.used_trace_ids("conversation-1")
        assert result.artifact is not None
        assert result.artifact.metadata["truth_status"] == "provider_reported"
        assert result.event is not None
        assert result.artifact.evidence_event_ids == [result.event.id]
        assert truth.calls == 1
        assert len(episodes.recent_episodes("conversation-1")) == 1
    finally:
        database.close()
