"""Public HTTP contract for developer recruitment and private-beta reservations."""

from __future__ import annotations

import re
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from ssa.services.community_github import CommunityGithubService
from ssa.storage.community_application_repository import (
    CommunityApplicationInput,
    CommunityApplicationRepository,
)

_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_CAPACITY = {"developer": 80, "tester": 500}


class CommunityApplicationRequest(BaseModel):
    track: Literal["developer", "tester"]
    display_name: str = Field(min_length=2, max_length=80)
    email: str = Field(min_length=5, max_length=254)
    focus: str = Field(min_length=2, max_length=120)
    profile_url: str | None = Field(default=None, max_length=500)
    platform: str | None = Field(default=None, max_length=80)
    notes: str = Field(default="", max_length=2_000)
    consent: bool
    source: str = Field(default="community-page", max_length=80)

    @field_validator("display_name", "focus", "notes", "source")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("email")
    @classmethod
    def _valid_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _EMAIL_RE.fullmatch(normalized):
            raise ValueError("email address is invalid")
        return normalized

    @field_validator("profile_url")
    @classmethod
    def _valid_profile_url(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        normalized = value.strip()
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("profile URL must be an http(s) URL")
        return normalized

    @field_validator("platform")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        return value.strip() if value and value.strip() else None

    @model_validator(mode="after")
    def _track_fields(self) -> CommunityApplicationRequest:
        if not self.consent:
            raise ValueError("privacy consent is required")
        if self.track == "tester" and self.platform is None:
            raise ValueError("tester applications require a platform")
        return self


def community_router(
    database_path: str,
    *,
    busy_timeout_ms: int = 5_000,
    github_service: CommunityGithubService | None = None,
) -> APIRouter:
    repository = CommunityApplicationRepository(
        database_path,
        busy_timeout_ms=busy_timeout_ms,
    )
    github = github_service or CommunityGithubService()
    router = APIRouter(prefix="/api/community", tags=["community"])

    @router.get("/status")
    def status() -> dict[str, Any]:
        try:
            counts = repository.counts()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="community registry is initializing") from exc
        return {
            "stage": "closed-beta",
            "developer": {
                "reserved": counts["developer"],
                "capacity": _CAPACITY["developer"],
            },
            "tester": {
                "reserved": counts["tester"],
                "capacity": _CAPACITY["tester"],
            },
        }

    @router.get("/github")
    def github_activity() -> dict[str, Any]:
        return github.get_activity()

    @router.post("/applications", status_code=201)
    def apply(body: CommunityApplicationRequest) -> dict[str, Any]:
        try:
            receipt = repository.submit(
                CommunityApplicationInput(
                    track=body.track,
                    email=body.email,
                    display_name=body.display_name,
                    focus=body.focus,
                    profile_url=body.profile_url,
                    platform=body.platform,
                    notes=body.notes,
                    metadata={"source": body.source},
                )
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail="reservation could not be recorded") from exc
        return {
            "application_code": receipt.application_code,
            "track": receipt.track,
            "queue_position": receipt.queue_position,
            "created": receipt.created,
        }

    return router


__all__ = ["CommunityApplicationRequest", "community_router"]
