"""Bounded delegation capabilities for a local headless opencode service."""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from collections.abc import Mapping
from contextlib import suppress
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
    conversation_id: str = ""


@dataclass
class _JobState:
    job: OpencodeJob
    state: str = "running"
    summary: str = "opencode job is running"
    exit_code: int | None = None
    message_id: str | None = None
    conversation_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    capacity_acquired: bool = False
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
                "conversation_id": status.get("conversation_id"),
                "message_id": status.get("message_id"),
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
                "conversation_id": status.get("conversation_id"),
                "message_id": status.get("message_id"),
                "state": status.get("state"),
                "provider": "opencode",
                "export": True,
            },
        )


class _OpencodeHttpClient:
    """Small stdlib HTTP adapter with bounded, restart-aware job tracking."""

    def __init__(self, config: OpencodeConfig, password: str, username: str) -> None:
        self._config = config
        self._base_url = config.base_url.rstrip("/")
        credentials = f"{username}:{password}".encode()
        self._authorization = "Basic " + base64.b64encode(credentials).decode("ascii")
        self._jobs: dict[str, _JobState] = {}
        self._jobs_lock = asyncio.Lock()
        self._persist_lock = asyncio.Lock()
        self._capacity = asyncio.Semaphore(config.fire_max_concurrent)
        self._load_jobs()

    async def create_session_and_prompt(
        self,
        *,
        task: str,
        project_dir: str,
        model: str,
        conversation_id: str,
    ) -> OpencodeJob:
        directory = str(Path(project_dir).expanduser().resolve())
        await self._cleanup_jobs()
        acquired = False
        state: _JobState | None = None
        try:
            try:
                await asyncio.wait_for(
                    self._capacity.acquire(),
                    timeout=self._config.capacity_wait_seconds,
                )
            except TimeoutError as exc:
                raise OpencodeError("opencode capacity exhausted; retry later") from exc
            acquired = True
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
                conversation_id=conversation_id,
            )
            state = _JobState(
                job=job,
                conversation_id=conversation_id,
                capacity_acquired=True,
            )
            async with self._jobs_lock:
                self._jobs[job.job_id] = state
            await self._persist_jobs()
            state.task = asyncio.create_task(
                self._run_job(state, task=task, model=model),
                name=f"opencode:{job.job_id}",
            )
            return job
        except BaseException:
            if state is not None:
                async with self._jobs_lock:
                    self._jobs.pop(state.job.job_id, None)
                with suppress(OpencodeError):
                    await self._persist_jobs()
            if acquired:
                self._capacity.release()
            raise

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
    ) -> None:
        try:
            response = await self._request(
                "POST",
                self._message_path(state) + "/prompt_async",
                {
                    "model": _model_payload(model),
                    "parts": [{"type": "text", "text": task}],
                },
            )
            state.message_id = _optional_id(response)
            state.summary = "opencode task submitted; waiting for assistant output"
            state.updated_at = time.time()
            await self._persist_jobs()
            deadline = time.monotonic() + self._config.job_timeout_seconds
            while state.state == "running":
                await self._refresh_messages(
                    state,
                    max_chars=self._config.poll_max_output_chars,
                )
                if state.state != "running":
                    break
                if time.monotonic() >= deadline:
                    raise OpencodeError(
                        f"opencode job exceeded {self._config.job_timeout_seconds}s runtime budget"
                    )
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            state.state = "aborted"
            state.exit_code = 1
            state.summary = "opencode job was cancelled"
            raise
        except OpencodeError as exc:
            state.state = "error"
            state.exit_code = 1
            state.summary = str(exc)
        except Exception as exc:
            state.state = "error"
            state.exit_code = 1
            state.summary = f"opencode job failed: {type(exc).__name__}: {exc}"
        finally:
            state.updated_at = time.time()
            if state.capacity_acquired:
                state.capacity_acquired = False
                self._capacity.release()
            await self._persist_jobs()

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
        previous = (state.state, state.message_id, state.exit_code)
        text = _message_text(payload)
        if text:
            state.summary = text[-max_chars:]
        assistant_text, assistant_id, assistant_complete = _assistant_message(payload)
        if assistant_id is not None:
            state.message_id = assistant_id
        if assistant_text:
            state.summary = assistant_text[-max_chars:]
        if state.state == "running" and assistant_complete:
            state.state = "done"
            state.exit_code = 0
        current = (state.state, state.message_id, state.exit_code)
        if current != previous:
            state.updated_at = time.time()
            await self._persist_jobs()

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
        await self._cleanup_jobs()
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
            "conversation_id": state.conversation_id,
            "message_id": state.message_id,
            "state": state.state,
            "summary": state.summary[-max_chars:],
            "exit_code": state.exit_code,
        }

    def _message_path(self, state: _JobState) -> str:
        return (
            f"/project/{quote(state.job.project_id, safe='')}/session/"
            f"{quote(state.job.session_id, safe='')}/message"
        )

    def _load_jobs(self) -> None:
        path = _job_store_path(self._config.job_store_path)
        if path is None or not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload.get("jobs", []) if isinstance(payload, Mapping) else []
        except (OSError, ValueError, TypeError):
            return
        now = time.time()
        for record in records:
            if not isinstance(record, Mapping):
                continue
            job_id = str(record.get("job_id", "")).strip()
            session_id = str(record.get("session_id", "")).strip()
            project_id = str(record.get("project_id", "")).strip()
            if not job_id or not session_id or not project_id:
                continue
            state_name = str(record.get("state", "error"))
            if state_name == "running":
                state_name = "aborted"
            if state_name not in {"done", "error", "aborted"}:
                state_name = "error"
            try:
                updated_at = float(record.get("updated_at", now))
                created_at = float(record.get("created_at", now))
            except (TypeError, ValueError):
                updated_at = now
                created_at = now
            if state_name in {"done", "error", "aborted"} and (
                now - updated_at > self._config.job_retention_seconds
            ):
                continue
            job = OpencodeJob(
                job_id=job_id,
                session_id=session_id,
                project_id=project_id,
                model=str(record.get("model", "")),
                conversation_id=str(record.get("conversation_id", "")),
            )
            self._jobs[job_id] = _JobState(
                job=job,
                state=state_name,
                summary=(
                    "opencode job was interrupted by client restart"
                    if str(record.get("state", "")) == "running"
                    else str(record.get("summary", "opencode job restored"))
                ),
                exit_code=(
                    int(record["exit_code"])
                    if record.get("exit_code") is not None
                    else (1 if state_name == "aborted" else None)
                ),
                message_id=(str(record["message_id"]) if record.get("message_id") else None),
                conversation_id=str(record.get("conversation_id", "")),
                created_at=created_at,
                updated_at=updated_at,
            )

    async def _cleanup_jobs(self) -> None:
        now = time.time()
        async with self._jobs_lock:
            orphaned = []
            for state in self._jobs.values():
                if (
                    state.state == "running"
                    and state.task is None
                    and now - state.updated_at > 1.0
                ):
                    state.state = "error"
                    state.exit_code = 1
                    state.summary = "opencode job has no active worker task"
                    state.updated_at = now
                    orphaned.append(state.job.job_id)
            stale = [
                job_id
                for job_id, state in self._jobs.items()
                if state.state in {"done", "error", "aborted"}
                and now - state.updated_at > self._config.job_retention_seconds
            ]
            for job_id in stale:
                self._jobs.pop(job_id, None)
        if stale or orphaned:
            await self._persist_jobs()

    async def _persist_jobs(self) -> None:
        path = _job_store_path(self._config.job_store_path)
        if path is None:
            return
        async with self._persist_lock:
            async with self._jobs_lock:
                records = [
                    {
                        "job_id": state.job.job_id,
                        "session_id": state.job.session_id,
                        "project_id": state.job.project_id,
                        "model": state.job.model,
                        "conversation_id": state.conversation_id,
                        "state": state.state,
                        "summary": state.summary,
                        "exit_code": state.exit_code,
                        "message_id": state.message_id,
                        "created_at": state.created_at,
                        "updated_at": state.updated_at,
                    }
                    for state in self._jobs.values()
                ]
            await asyncio.to_thread(_write_job_store, path, records)

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


def _assistant_message(payload: Any) -> tuple[str, str | None, bool]:
    """Return the latest assistant text, id, and completion signal."""
    messages = payload if isinstance(payload, list) else [payload]
    latest_text = ""
    latest_id: str | None = None
    complete = False
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        info = message.get("info")
        info_map = info if isinstance(info, Mapping) else {}
        role = str(message.get("role") or info_map.get("role") or "").lower()
        if role != "assistant":
            continue
        text = _message_text(message)
        if text:
            latest_text = text
        latest_id = _optional_id(message) or latest_id
        timing = info_map.get("time")
        status = str(message.get("status") or info_map.get("status") or "").lower()
        complete = complete or bool(
            info_map.get("finish")
            or info_map.get("finishReason")
            or (isinstance(timing, Mapping) and timing.get("completed"))
            or status in {"done", "completed", "success"}
        )
    return latest_text, latest_id, complete


def _job_store_path(value: str) -> Path | None:
    normalized = value.strip()
    return Path(normalized).expanduser() if normalized else None


def _write_job_store(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps({"version": 1, "jobs": records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise OpencodeError(f"could not persist opencode job store: {exc}") from exc


__all__ = ["OpencodeError", "OpencodeExecutor", "OpencodeJob"]
