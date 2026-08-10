"""SQLite persistence coverage for the cement-seal state and ledger."""

from __future__ import annotations

from pathlib import Path

from ssa.config import DatabaseConfig
from ssa.domain.cement_seal import (
    CementSealPhase,
    CementSealState,
    CementSealTransition,
    CementSealTrigger,
)
from ssa.storage.cement_seal_repository import CementSealRepository
from ssa.storage.database import Database


def test_cement_seal_state_and_transition_round_trip(tmp_path: Path) -> None:
    database_path = tmp_path / "cement-seal.db"
    database = Database(DatabaseConfig(path=str(database_path)))
    database.initialize()
    repository = CementSealRepository(str(database_path))
    try:
        assert database.schema_version == 26
        assert repository.get() == CementSealState()

        sealed = CementSealState(
            phase=CementSealPhase.SEALED,
            integrity=1.0,
            sealed_at_ms=100,
            last_transition_ms=100,
            source_event_ids=["event-1"],
            version=2,
        )
        transition = CementSealTransition(
            previous_phase=CementSealPhase.OPEN,
            phase=CementSealPhase.SEALED,
            trigger=CementSealTrigger.UNREPAIRED_RECURRENCE,
            integrity_before=0.0,
            integrity_after=1.0,
            seal_count=0,
            toughness=0.55,
            reason="repeated unrepaired distress crossed the configured threshold",
            source_event_ids=["event-1"],
            occurred_at_ms=100,
        )

        repository.save(sealed)
        transition_id = repository.append_transition(transition)

        assert repository.get() == sealed
        assert transition_id.startswith("cement:")
        assert repository.recent_transitions() == [transition]
    finally:
        database.close()
