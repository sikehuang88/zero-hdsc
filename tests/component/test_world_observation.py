"""World observation persistence and deduplication tests."""

from __future__ import annotations

import json
from pathlib import Path

from ssa.clock import FrozenClock
from ssa.config import DatabaseConfig, Settings, WorldConfig
from ssa.domain.enums import Actor, SourceKind
from ssa.ids import SequentialIdGenerator
from ssa.services.world_observation_service import WorldObservationService
from ssa.storage.database import Database
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.lifecycle_repository import WorldObservationRepository


def test_clock_file_and_calendar_observations_are_deduplicated(tmp_path: Path) -> None:
    watched = tmp_path / "watched.txt"
    watched.write_text("v1", encoding="utf-8")
    calendar = tmp_path / "calendar.json"
    calendar.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "id": "meeting-1",
                        "title": "Shared project review",
                        "start": "2030-03-05T18:00:00Z",
                        "salience": 0.9,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        database=DatabaseConfig(path=str(tmp_path / "world.db")),
        world=WorldConfig(
            watched_paths=[str(watched)],
            calendar_json_paths=[str(calendar)],
        ),
    )
    database = Database(settings.database)
    database.initialize()
    clock = FrozenClock(1_900_000_000_000)
    events = SqliteEventRepository(database.connection)
    repository = WorldObservationRepository(database.connection)
    service = WorldObservationService(
        observations=repository,
        events=events,
        clock=clock,
        ids=SequentialIdGenerator("world"),
        config=settings.world,
        timezone="UTC",
    )
    try:
        first = service.observe("conversation-1")
        second = service.observe("conversation-1")

        assert {item.observation_type for item in first.observations} == {
            "clock",
            "file_change",
            "calendar",
        }
        assert first.material_count == 2
        assert second.observations == ()
        assert all(event.actor == Actor.WORLD for event in first.events)
        assert all(event.source_kind == SourceKind.WORLD_OBSERVED for event in first.events)
        assert len(repository.recent("conversation-1")) == 3

        watched.write_text("v2 changed", encoding="utf-8")
        changed = service.observe("conversation-1")
        assert [item.observation_type for item in changed.observations] == ["file_change"]
        assert changed.material_count == 1
    finally:
        database.close()
