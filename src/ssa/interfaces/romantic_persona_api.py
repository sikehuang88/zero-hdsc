"""Local HTTP contract for the persistent romantic persona card."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ssa.domain.romantic_persona import RomanticPersonaProfile
from ssa.storage.romantic_persona_repository import RomanticPersonaRepository


def romantic_persona_router(
    database_path: str,
    *,
    busy_timeout_ms: int = 5_000,
) -> APIRouter:
    repository = RomanticPersonaRepository(
        database_path,
        busy_timeout_ms=busy_timeout_ms,
    )
    router = APIRouter(prefix="/api/relationship", tags=["romantic-persona"])

    @router.get("/persona")
    def get_romantic_persona() -> RomanticPersonaProfile:
        try:
            return repository.get()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="romantic persona unavailable") from exc

    @router.put("/persona")
    def save_romantic_persona(body: RomanticPersonaProfile) -> RomanticPersonaProfile:
        try:
            return repository.save(body)
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail="romantic persona could not be saved"
            ) from exc

    return router


__all__ = ["romantic_persona_router"]
