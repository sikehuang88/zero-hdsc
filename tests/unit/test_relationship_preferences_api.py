"""Relationship preference persistence and HTTP contract coverage."""

from __future__ import annotations

from pathlib import Path

from ssa.config import DatabaseConfig, InitiativeConfig
from ssa.domain.relationship_preferences import (
    ProactiveFrequency,
    RelationshipPreferences,
    RelationshipSupportMode,
)
from ssa.interfaces.relationship_preferences_api import relationship_preferences_router
from ssa.services.proactive_contact_policy import ProactiveContactPolicy
from ssa.storage.database import Database
from ssa.storage.relationship_preferences_repository import RelationshipPreferencesRepository


def _initialized_database(tmp_path: Path) -> tuple[Database, Path]:
    path = tmp_path / "preferences.db"
    database = Database(DatabaseConfig(path=str(path)))
    database.initialize()
    return database, path


def test_relationship_preferences_round_trip(tmp_path: Path) -> None:
    database, path = _initialized_database(tmp_path)
    try:
        repository = RelationshipPreferencesRepository(str(path))
        defaults = repository.get()
        assert defaults.venom_intensity == 58
        assert defaults.support_mode == RelationshipSupportMode.ADAPTIVE

        saved = repository.save(
            RelationshipPreferences(
                venom_intensity=24,
                support_mode=RelationshipSupportMode.LISTEN,
                proactive_frequency=ProactiveFrequency.LOW,
                quiet_hours_enabled=True,
                quiet_hours_start="22:30",
                quiet_hours_end="09:15",
            )
        )

        assert saved.updated_at_ms > 0
        assert repository.get() == saved
    finally:
        database.close()


def test_relationship_preferences_http_contract(tmp_path: Path) -> None:
    database, path = _initialized_database(tmp_path)
    try:
        routes = {
            route.name: route.endpoint
            for route in relationship_preferences_router(str(path)).routes
        }
        updated = routes["save_preferences"](
            RelationshipPreferences(
                venom_intensity=72,
                support_mode=RelationshipSupportMode.CHALLENGE,
                proactive_frequency=ProactiveFrequency.HIGH,
                quiet_hours_enabled=False,
            )
        )

        assert updated.venom_intensity == 72
        assert routes["get_preferences"]() == updated
    finally:
        database.close()


def test_relationship_preferences_generate_behavior_context() -> None:
    gentle = RelationshipPreferences(
        venom_intensity=12,
        support_mode=RelationshipSupportMode.COMFORT,
    ).prompt_context()
    sharp = RelationshipPreferences(
        venom_intensity=88,
        support_mode=RelationshipSupportMode.CHALLENGE,
    ).prompt_context()

    assert "mostly gentle" in gentle
    assert "companionship before solutions" in gentle
    assert "strongly blunt" in sharp
    assert "self-deception" in sharp


def test_proactive_frequency_resolves_distinct_limits(tmp_path: Path) -> None:
    database, path = _initialized_database(tmp_path)
    repository = RelationshipPreferencesRepository(str(path))
    policy = ProactiveContactPolicy(
        preferences=repository,
        config=InitiativeConfig(),
        timezone="UTC",
    )
    try:
        expected = {
            ProactiveFrequency.LOW: (1, 720, 8),
            ProactiveFrequency.BALANCED: (3, 240, 4),
            ProactiveFrequency.HIGH: (5, 120, 2),
        }
        for frequency, limits in expected.items():
            repository.save(
                RelationshipPreferences(
                    proactive_frequency=frequency,
                    quiet_hours_enabled=False,
                )
            )
            current = policy.current()
            assert (current.daily_limit, current.cooldown_minutes, current.min_gap_hours) == limits
    finally:
        database.close()
