"""Bounded delegation capabilities for a local headless opencode service."""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from ssa.config import OpencodeConfig
from ssa.tools.models import (
    OpencodeCheckArguments,
    OpencodeFireArguments,
    OpencodeSessionExportArguments,
    ToolAutonomyContext,
    ToolOutcome,
)
from ssa.tools.registry import ToolCapability, ToolRegistry


class OpencodeError(RuntimeError):
    """A bounded, user-readable opencode transport or job error."""


@dataclass(frozen=True)
class OpencodeJob:
    job_id: str
    session_id: str
    project_id: str
    model: str


@dataclass
class _JobState:
    job: OpencodeJob
    state: str = "running"
    summary: str = "opencode job is running"
    exit_code: int | None = None
    message_id: str | None = None
    task: asyncio.Task[None] | None = field(default=None, repr=False)


class OpencodeExecutor:
    """Expose fire/check/export without consuming the outer tool-loop budget."""

    def __init__(
        self,
        config: OpencodeConfig,
        server_password: str,
        *,
        server_username: str = "opencode",
        client: Any | None = None,
    ) -> None:
        if not server_password.strip() and client is None:
            raise ValueError("opencode requires OPENCODE_SERVER_PASSWORD")
        self._config = config
        self._client = client or _OpencodeHttpClient(config, server_password, server_username)

    def register_into(self, registry: ToolRegistry) -> None:
        registry.register(
            ToolCapability(
                name="opencode_fire",
                description=(
                    "Delegate a bounded coding task to a local headless opencode agent. "
                    "Returns a job id immediately; use opencode_check to poll. "
                    "The task is constrained to the supplied workspace."
                ),
                arguments_model=OpencodeFireArguments,
                handler=self.fire,
            )
        )
        registry.register(
            ToolCapability(
                name="opencode_check",
                description=(
                    "Poll a previously fired opencode job and return its current state, "
                    "bounded output, and verification summary."
                ),
                arguments_model=OpencodeCheckArguments,
                handler=self.check,
            )
        )
        registry.register(
            ToolCapability(
                name="opencode_export",
                description=(
                    "Export the bounded message trace and final status for a completed "
                    "opencode job."
                ),
                arguments_model=OpencodeSessionExportArguments,
                handler=self.export,
            )
        )

    async def fire(
        self,
        raw_arguments: dict[str, Any],
        context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = OpencodeFireArguments.model_validate(raw_arguments)
        project_dir = arguments.project_dir or context.workspace_root or self._config.project_dir
        if not project_dir.strip():
            return ToolOutcome(ok=False, error="opencode requires a workspace_root or project_dir")
        try:
            job = await self._client.create_session_and_prompt(
                task=arguments.task,
                project_dir=project_dir,
                model=arguments.model or self._config.default_model,
                conversation_id=context.conversation_id,
            )
        except OpencodeError as exc:
            return ToolOutcome(ok=False, error=str(exc))
        return ToolOutcome(
            ok=True,
            output=f"opencode job started: {job.job_id}",
            metadata={
                "opencode_delegation": True,
                "job_id": job.job_id,
                "session_id": job.session_id,
                "project_id": job.project_id,
                "provider": "opencode",
                "model": job.model,
                "conversation_id": context.conversation_id,
            },
        )

    async def check(
        self,
        raw_arguments: dict[str, Any],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = OpencodeCheckArguments.model_validate(raw_arguments)
        try:
            status = await self._client.poll(arguments.job_id, max_chars=arguments.max_chars)
        except OpencodeError as exc:
            return ToolOutcome(ok=False, error=str(exc))
        return ToolOutcome(
            ok=status["state"] not in {"error", "aborted"},
            output=str(status.get("summary", "")),
            exit_code=status.get("exit_code"),
            metadata={
                "opencode_delegation": True,
                "job_id": arguments.job_id,
                "session_id": status.get("session_id"),
                "project_id": status.get("project_id"),
                "state": status.get("state"),
                "provider": "opencode",
            },
        )

    async def export(
        self,
        raw_arguments: dict[str, Any],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = OpencodeSessionExportArguments.model_validate(raw_arguments)
        try:
            status = await self._client.export(
                arguments.job_id, max_chars=self._config.poll_max_output_chars
            )
        except OpencodeError as exc:
            return ToolOutcome(ok=False, error=str(exc))
        return ToolOutcome(
            ok=status["state"] not in {"error", "aborted"},
            output=str(status.get("summary", "")),
            exit_code=status.get("exit_code"),
            metadata={
                "opencode_delegation": True,
                "job_id": arguments.job_id,
                "session_id": status.get("session_id"),
                "project_id": status.get("project_id"),
                "state": status.get("state"),
                "provider": "opencode",
                "export": True,
            },
        )


class _OpencodeHttpClient:
    """Small stdlib HTTP adapter with an in-memory fire/check job table."""

    def __init__(self, config: OpencodeConfig, password: str, username: str) -> None:
        self._config = config
        self._base_url = config.base_url.rstrip("/")
        credentials = f"{username}:{password}".encode()
        self._authorization = "Basic " + base64.b64encode(credentials).decode("ascii")
        self._jobs: dict[str, _JobState] = {}
        self._jobs_lock = asyncio.Lock()
        self._capacity = asyncio.Semaphore(config.fire_max_concurrent)

    async def create_session_and_prompt(
        self,
        *,
        task: str,
        project_dir: str,
        model: str,
        conversation_id: str,
    ) -> OpencodeJob:
        directory = str(Path(project_dir).expanduser().resolve())
        project_id = await self._resolve_project_id(directory)
        session_payload = await self._request(
            "POST",
            f"/project/{quote(project_id, safe='')}/session",
            {"directory": directory},
        )
        session_id = _string_id(session_payload, "session id")
        job = OpencodeJob(
            job_id=f"opencode:{uuid.uuid4().hex}",
            session_id=session_id,
            project_id=project_id,
            model=model,
        )
        state = _JobState(job=job)
        async with self._jobs_lock:
            self._jobs[job.job_id] = state
        state.task = asyncio.create_task(
            self._run_job(state, task=task, model=model, conversation_id=conversation_id),
            name=f"opencode:{job.job_id}",
        )
        return job

    async def poll(self, job_id: str, *, max_chars: int) -> dict[str, Any]:
        state = await self._get_job(job_id)
        if state.state == "running":
            await self._refresh_messages(state, max_chars=max_chars)
        return self._status_payload(state, max_chars=max_chars)

    async def export(self, job_id: str, *, max_chars: int) -> dict[str, Any]:
        state = await self._get_job(job_id)
        await self._refresh_messages(state, max_chars=max_chars)
        payload = self._status_payload(state, max_chars=max_chars)
        payload["summary"] = f"job={job_id} state={state.state}\n{payload['summary']}"
        return payload

    async def _run_job(
        self,
        state: _JobState,
        *,
        task: str,
        model: str,
        conversation_id: str,
    ) -> None:
        del conversation_id
        async with self._capacity:
            try:
                response = await self._request(
                    "POST",
                    f"/project/{quote(state.job.project_id, safe='')}/session/"
                    f"{quote(state.job.session_id, safe='')}/message",
                    {
                        "model": _model_payload(model),
                        "parts": [{"type": "text", "text": task}],
                    },
                )
                state.message_id = _optional_id(response)
                state.state = "done"
                state.exit_code = 0
                state.summary = _message_text(response) or "opencode completed without text output"
            except asyncio.CancelledError:
                state.state = "aborted"
                state.summary = "opencode job was cancelled"
                raise
            except OpencodeError as exc:
                state.state = "error"
                state.exit_code = 1
                state.summary = str(exc)

    async def _refresh_messages(self, state: _JobState, *, max_chars: int) -> None:
        try:
            payload = await self._request(
                "GET",
                f"/project/{quote(state.job.project_id, safe='')}/session/"
                f"{quote(state.job.session_id, safe='')}/message",
            )
        except OpencodeError:
            if state.state == "running":
                return
            raise
        text = _message_text(payload)
        if text:
            state.summary = text[-max_chars:]

    async def _resolve_project_id(self, directory: str) -> str:
        try:
            payload = await self._request("GET", "/project")
        except OpencodeError:
            payload = []
        projects = (
            payload
            if isinstance(payload, list)
            else payload.get("projects", [])
            if isinstance(payload, Mapping)
            else []
        )
        for project in projects:
            if not isinstance(project, Mapping):
                continue
            candidate = str(
                project.get("worktree") or project.get("directory") or project.get("path") or ""
            )
            if candidate and Path(candidate).resolve() == Path(directory).resolve():
                return _string_id(project, "project id")
        initialized = await self._request("POST", "/project/init", {"directory": directory})
        return _string_id(initialized, "project id")

    async def _get_job(self, job_id: str) -> _JobState:
        async with self._jobs_lock:
            state = self._jobs.get(job_id)
        if state is None:
            raise OpencodeError(f"unknown opencode job: {job_id}")
        return state

    def _status_payload(self, state: _JobState, *, max_chars: int) -> dict[str, Any]:
        return {
            "job_id": state.job.job_id,
            "session_id": state.job.session_id,
            "project_id": state.job.project_id,
            "state": state.state,
            "summary": state.summary[-max_chars:],
            "exit_code": state.exit_code,
        }

    async def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        return await asyncio.to_thread(self._request_sync, method, path, payload)

    def _request_sync(self, method: str, path: str, payload: dict[str, Any] | None) -> Any:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            self._base_url + path,
            data=body,
            method=method,
            headers={
                "Accept": "application/json",
                "Authorization": self._authorization,
                **({"Content-Type": "application/json"} if body is not None else {}),
            },
        )
        try:
            with urlopen(request, timeout=self._config.request_timeout_seconds) as response:
                raw = response.read()
        except HTTPError as exc:
            detail = exc.read(4_096).decode("utf-8", errors="replace").strip()
            raise OpencodeError(f"opencode HTTP {exc.code}: {detail or exc.reason}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OpencodeError(f"opencode service unavailable: {exc}") from exc
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise OpencodeError("opencode returned invalid JSON") from exc


def _string_id(payload: Any, label: str) -> str:
    value = _optional_id(payload)
    if value is None:
        raise OpencodeError(f"opencode response did not include {label}")
    return value


def _optional_id(payload: Any) -> str | None:
    if isinstance(payload, Mapping):
        for key in ("id", "sessionID", "sessionId", "projectID", "projectId"):
            value = payload.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        nested = payload.get("info")
        if nested is not None:
            return _optional_id(nested)
    return None


def _model_payload(model: str) -> dict[str, str]:
    provider, separator, model_id = model.partition("/")
    if not separator:
        return {"providerID": "opencode", "modelID": model}
    return {"providerID": provider, "modelID": model_id}


def _message_text(payload: Any) -> str:
    values: list[str] = []
    messages = payload if isinstance(payload, list) else [payload]
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        parts = message.get("parts")
        if not isinstance(parts, list):
            parts = [message]
        for part in parts:
            if not isinstance(part, Mapping):
                continue
            for key in ("text", "content", "output"):
                value = part.get(key)
                if isinstance(value, str) and value.strip():
                    values.append(value.strip())
                    break
    return "\n".join(values)


__all__ = ["OpencodeError", "OpencodeExecutor", "OpencodeJob"]
