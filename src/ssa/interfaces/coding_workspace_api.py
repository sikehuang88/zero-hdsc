"""HTTP API used by Zero's local Coding workspace UI."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from ssa.services.coding_workspace import (
    CodingWorkspaceBinaryError,
    CodingWorkspaceConflictError,
    CodingWorkspaceError,
    CodingWorkspaceNotFoundError,
    CodingWorkspaceTooLargeError,
    git_workspace_diff,
    git_workspace_status,
    list_workspace_tree,
    read_workspace_file,
    run_workspace_command,
    search_workspace,
    write_workspace_file,
)


class CodingFileWriteRequest(BaseModel):
    workspace_root: str = Field(min_length=1, max_length=2_048)
    path: str = Field(min_length=1, max_length=4_096)
    content: str = Field(max_length=2_097_152)
    expected_mtime_ns: int | None = Field(default=None, ge=0)


class CodingCommandRequest(BaseModel):
    workspace_root: str = Field(min_length=1, max_length=2_048)
    command: str = Field(min_length=1, max_length=16_384)
    timeout_seconds: int = Field(default=60, ge=1, le=180)

    @field_validator("command")
    @classmethod
    def _command_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("command must not be blank")
        return value


def coding_workspace_router() -> APIRouter:
    router = APIRouter(prefix="/api/coding", tags=["coding-workspace"])

    @router.get("/tree")
    async def tree(
        workspace_root: str = Query(min_length=1, max_length=2_048),
        path: str = Query(default=".", max_length=4_096),
        max_depth: int = Query(default=5, ge=1, le=8),
        max_entries: int = Query(default=1_500, ge=1, le=2_000),
    ) -> Any:
        return _call_workspace(
            list_workspace_tree,
            workspace_root,
            path=path,
            max_depth=max_depth,
            max_entries=max_entries,
        )

    @router.get("/search")
    async def search(
        workspace_root: str = Query(min_length=1, max_length=2_048),
        query: str = Query(min_length=1, max_length=500),
        path: str = Query(default=".", max_length=4_096),
        max_results: int = Query(default=120, ge=1, le=500),
    ) -> Any:
        try:
            return await search_workspace(
                workspace_root,
                query,
                path=path,
                max_results=max_results,
            )
        except CodingWorkspaceError as exc:
            raise _http_error(exc) from exc

    @router.get("/file")
    async def read_file(
        workspace_root: str = Query(min_length=1, max_length=2_048),
        path: str = Query(min_length=1, max_length=4_096),
    ) -> Any:
        return _call_workspace(read_workspace_file, workspace_root, path)

    @router.put("/file")
    async def write_file(body: CodingFileWriteRequest) -> Any:
        return _call_workspace(
            write_workspace_file,
            body.workspace_root,
            body.path,
            body.content,
            expected_mtime_ns=body.expected_mtime_ns,
        )

    @router.get("/git/status")
    async def git_status(
        workspace_root: str = Query(min_length=1, max_length=2_048),
    ) -> Any:
        try:
            return await git_workspace_status(workspace_root)
        except CodingWorkspaceError as exc:
            raise _http_error(exc) from exc

    @router.get("/git/diff")
    async def git_diff(
        workspace_root: str = Query(min_length=1, max_length=2_048),
        path: str = Query(min_length=1, max_length=4_096),
    ) -> Any:
        try:
            return await git_workspace_diff(workspace_root, path)
        except CodingWorkspaceError as exc:
            raise _http_error(exc) from exc

    @router.post("/command")
    async def command(body: CodingCommandRequest) -> Any:
        try:
            return await run_workspace_command(
                body.workspace_root,
                body.command,
                timeout_seconds=body.timeout_seconds,
            )
        except CodingWorkspaceError as exc:
            raise _http_error(exc) from exc

    return router


def _call_workspace(function: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except CodingWorkspaceError as exc:
        raise _http_error(exc) from exc


def _http_error(exc: CodingWorkspaceError) -> HTTPException:
    if isinstance(exc, CodingWorkspaceNotFoundError):
        status = 404
    elif isinstance(exc, CodingWorkspaceBinaryError):
        status = 415
    elif isinstance(exc, CodingWorkspaceTooLargeError):
        status = 413
    elif isinstance(exc, CodingWorkspaceConflictError):
        status = 409
    else:
        status = 400
    return HTTPException(status_code=status, detail=str(exc))


__all__ = ["CodingCommandRequest", "CodingFileWriteRequest", "coding_workspace_router"]
