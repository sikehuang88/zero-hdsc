"""Local HTTP contract for relationship communication preferences."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ssa.domain.relationship_preferences import RelationshipPreferences
from ssa.storage.relationship_preferences_repository import RelationshipPreferencesRepository


def relationship_preferences_router(
    database_path: str,
    *,
    busy_timeout_ms: int = 5_000,
) -> APIRouter:
    repository = RelationshipPreferencesRepository(
        database_path,
        busy_timeout_ms=busy_timeout_ms,
    )
    router = APIRouter(prefix="/api/relationship", tags=["relationship-preferences"])

    @router.get("/preferences")
    def get_preferences() -> RelationshipPreferences:
        try:
            return repository.get()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="relationship preferences unavailable") from exc

    @router.put("/preferences")
    def save_preferences(body: RelationshipPreferences) -> RelationshipPreferences:
        try:
            return repository.save(body)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="relationship preferences could not be saved") from exc

    return router


__all__ = ["relationship_preferences_router"]
