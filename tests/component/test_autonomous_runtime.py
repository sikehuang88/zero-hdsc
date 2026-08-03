"""Unified P1-P6 autonomous runtime integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssa.adapters.llm import FakeLLMAdapter
from ssa.clock import FrozenClock
from ssa.config import DatabaseConfig, Settings, WorldConfig
from ssa.runtime.autonomous import AutonomousRuntime
from ssa.storage.database import Database


@pytest.mark.asyncio
async def test_runtime_bootstrap_is_persistent_and_runs_every_module(tmp_path: Path) -> None:
    settings = Settings(
        database=DatabaseConfig(path=str(tmp_path / "autonomous.db")),
        world=WorldConfig(observe_interval_seconds=300),
    )
    database = Database(settings.database)
    database.initialize()
    clock = FrozenClock(1_900_000_000_000)
    runtime = AutonomousRuntime(
        database=database,
        llm=FakeLLMAdapter(),
        settings=settings,
        conversation_id="conversation-1",
        clock=clock,
    )
    try:
        first = runtime.bootstrap()
        second = runtime.bootstrap()
        assert [job.id for job in first] == [job.id for job in second]
        assert {job.job_type for job in first} == {
            "agency.offline_cycle",
            "state.refresh",
            "health.check",
            "inner.heartbeat",
            "reflection.schedule",
            "learning.consolidate",
            "learning.evaluate_outcomes",
            "trace.resting_step",
            "goal.advance",
            "initiative.evaluate",
            "initiative.expire",
            "contact.evaluate",
            "outbox.deliver",
            "world.observe",
        }

        tick = await runtime.tick()
        assert tick.lifecycle.leased == 14
        assert tick.lifecycle.failed == 0
        assert runtime.states.latest() is not None
        assert runtime.goals.list_active("conversation-1")
        observations = runtime.world_observations.recent("conversation-1")
        assert [item.observation_type for item in observations] == ["clock"]
    finally:
        database.close()


@pytest.mark.asyncio
async def test_material_world_change_expedites_resting_and_initiative_jobs(
    tmp_path: Path,
) -> None:
    watched = tmp_path / "watched.txt"
    watched.write_text("initial", encoding="utf-8")
    settings = Settings(
        database=DatabaseConfig(path=str(tmp_path / "material.db")),
        world=WorldConfig(observe_interval_seconds=300, watched_paths=[str(watched)]),
    )
    database = Database(settings.database)
    database.initialize()
    clock = FrozenClock(1_900_000_000_000)
    runtime = AutonomousRuntime(
        database=database,
        llm=FakeLLMAdapter(),
        settings=settings,
        conversation_id="conversation-1",
        clock=clock,
    )
    try:
        runtime.bootstrap()
        await runtime.tick()
        resting = runtime.jobs.find_by_dedup("lifecycle:conversation-1:trace.resting_step")
        initiative = runtime.jobs.find_by_dedup("lifecycle:conversation-1:initiative.evaluate")
        assert resting is not None and resting.due_at_ms == clock.now_ms()
        assert initiative is not None and initiative.due_at_ms == clock.now_ms()
    finally:
        database.close()
