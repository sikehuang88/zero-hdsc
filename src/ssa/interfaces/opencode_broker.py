"""Local authenticated Broker endpoints used by the vendored ZERO bridge plugin."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from ssa.runtime.interactive import DigitalLifeSession
from ssa.services.coding_workspace import (
    CodingWorkspaceError,
    CodingWorkspaceNotFoundError,
    resolve_workspace_path,
    resolve_workspace_root,
    run_workspace_command,
    write_workspace_file,
)


class BrokerExecRequest(BaseModel):
    command: str = Field(min_length=1, max_length=16_384)
    root: str = Field(min_length=1, max_length=2_048)
    cwd: str | None = Field(default=None, max_length=4_096)
    timeout_seconds: int = Field(default=60, ge=1, le=180)

    @field_validator("command", "root")
    @classmethod
    def _required_values_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Broker request value must not be blank")
        return normalized


class BrokerWriteRequest(BaseModel):
    root: str = Field(min_length=1, max_length=2_048)
    path: str = Field(min_length=1, max_length=4_096)
    content: str = Field(max_length=2_097_152)


class BrokerWritebackRequest(BaseModel):
    sessionID: str = Field(min_length=1, max_length=200)
    root: str = Field(min_length=1, max_length=2_048)


def opencode_broker_router(
    session_provider: Callable[[], DigitalLifeSession],
    token: str,
) -> APIRouter:
    """Build a Broker with explicit Bearer authentication."""
    if not token.strip():
        raise ValueError("opencode Broker token must not be blank")
    router = APIRouter(prefix="/api/opencode-broker", tags=["opencode-broker"])

    def authenticate(request: Request) -> None:
        if request.headers.get("authorization", "") != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="invalid Broker token")

    @router.get("/context")
    async def context(request: Request, sessionID: str = "") -> PlainTextResponse:
        authenticate(request)
        active = session_provider()
        snapshot = active.snapshot()
        payload: dict[str, Any] = {
            "provenance": "HDSC derived control context; not user fact",
            "conversation_id": active.conversation_id,
            "requested_session_id": sessionID,
            "organism": snapshot.organism.model_dump(mode="json"),
            "relationship": snapshot.relationship.model_dump(mode="json"),
            "emotion_frames": [frame.presentation_data() for frame in snapshot.emotion_frames[:4]],
            "active_trace_count": snapshot.trace_space.total_traces,
        }
        return PlainTextResponse(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    @router.post("/exec")
    async def execute(request: Request, body: BrokerExecRequest) -> dict[str, Any]:
        authenticate(request)
        root = resolve_workspace_root(body.root)
        working_directory = root
        if body.cwd:
            _, working_directory = resolve_workspace_path(str(root), body.cwd)
        try:
            return await run_workspace_command(
                str(working_directory),
                body.command,
                timeout_seconds=body.timeout_seconds,
            )
        except CodingWorkspaceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/write")
    async def write(request: Request, body: BrokerWriteRequest) -> dict[str, Any]:
        authenticate(request)
        try:
            try:
                return write_workspace_file(
                    body.root,
                    body.path,
                    body.content,
                    expected_mtime_ns=None,
                )
            except CodingWorkspaceNotFoundError:
                root, target = resolve_workspace_path(body.root, body.path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("", encoding="utf-8")
                return write_workspace_file(
                    str(root),
                    target.relative_to(root).as_posix(),
                    body.content,
                    expected_mtime_ns=None,
                )
        except CodingWorkspaceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/session-writeback")
    async def session_writeback(
        request: Request,
        body: BrokerWritebackRequest,
    ) -> dict[str, str]:
        authenticate(request)
        event = session_provider().record_opencode_writeback(
            session_id=body.sessionID,
            workspace_root=body.root,
        )
        return {"event_id": event.id, "state": "recorded"}

    return router


__all__ = [
    "BrokerExecRequest",
    "BrokerWriteRequest",
    "BrokerWritebackRequest",
    "opencode_broker_router",
]
