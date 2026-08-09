"""Romantic persona persistence, prompt projection, and HTTP contract coverage."""

from __future__ import annotations

from pathlib import Path

from ssa.config import DatabaseConfig
from ssa.domain.romantic_persona import (
    AffectionStyle,
    CareStyle,
    ConflictStyle,
    RelationshipStage,
    RomanticPersonaProfile,
)
from ssa.interfaces.romantic_persona_api import romantic_persona_router
from ssa.storage.database import Database
from ssa.storage.romantic_persona_repository import RomanticPersonaRepository


def _initialized_database(tmp_path: Path) -> tuple[Database, Path]:
    path = tmp_path / "persona.db"
    database = Database(DatabaseConfig(path=str(path)))
    database.initialize()
    return database, path


def test_romantic_persona_round_trip(tmp_path: Path) -> None:
    database, path = _initialized_database(tmp_path)
    try:
        repository = RomanticPersonaRepository(str(path))
        defaults = repository.get()
        assert defaults.character_name == "清雪"
        assert defaults.relationship_stage == RelationshipStage.COMMITTED

        saved = repository.save(
            RomanticPersonaProfile(
                character_name="清雪",
                owner_address="笨蛋",
                relationship_stage=RelationshipStage.LONG_TERM,
                affection_style=AffectionStyle.EXPRESSIVE,
                care_style=CareStyle.PROTECTIVE,
                conflict_style=ConflictStyle.SOFT_REPAIR,
                teasing_intensity=70,
                vulnerability_openness=64,
                custom_notes="记得把关心落到具体事情上。",
                version=defaults.version,
            )
        )

        assert saved.version == defaults.version + 1
        assert saved.updated_at_ms > 0
        assert repository.get() == saved
    finally:
        database.close()


def test_romantic_persona_http_contract(tmp_path: Path) -> None:
    database, path = _initialized_database(tmp_path)
    try:
        routes = {route.name: route.endpoint for route in romantic_persona_router(str(path)).routes}
        current = routes["get_romantic_persona"]()
        updated = routes["save_romantic_persona"](
            current.model_copy(update={"affection_style": AffectionStyle.RESTRAINED})
        )

        assert updated.affection_style == AffectionStyle.RESTRAINED
        assert routes["get_romantic_persona"]() == updated
    finally:
        database.close()


def test_stale_romantic_persona_save_keeps_version_monotonic(tmp_path: Path) -> None:
    database, path = _initialized_database(tmp_path)
    try:
        repository = RomanticPersonaRepository(str(path))
        stale = repository.get()
        first = repository.save(stale.model_copy(update={"teasing_intensity": 30}))
        second = repository.save(stale.model_copy(update={"teasing_intensity": 80}))

        assert second.version == first.version + 1
        assert repository.get() == second
    finally:
        database.close()


def test_romantic_persona_prompt_separates_identity_from_dynamic_state() -> None:
    context = RomanticPersonaProfile(
        affection_style=AffectionStyle.BALANCED,
        care_style=CareStyle.ATTENTIVE,
        conflict_style=ConflictStyle.DIRECT_REPAIR,
        teasing_intensity=62,
        vulnerability_openness=48,
    ).prompt_context()

    assert "Persistent romantic persona card" in context
    assert "exclusive girlfriend" in context
    assert "receive the owner's emotion or action" in context
    assert "Read behavior, not minds" in context
    assert "dynamic state" in context
    assert "do not overwrite this stable card" in context
    assert "never invent autobiographical memories" in context
