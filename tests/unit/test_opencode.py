"""Unit coverage for the bounded opencode delegation boundary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from ssa.config import Environment, OpencodeConfig, ToolConfig, load_settings
from ssa.interfaces.opencode_broker import opencode_broker_router
from ssa.tools.executors import build_default_tool_kernel
from ssa.tools.models import ToolAutonomyContext
from ssa.tools.opencode import OpencodeExecutor, OpencodeJob
from ssa.tools.registry import ToolRegistry


def _context() -> ToolAutonomyContext:
    return ToolAutonomyContext(
        correlation_id="corr-opencode",
        conversation_id="conv-opencode",
        energy=0.8,
        valence=0.1,
        arousal=0.2,
        trust=0.7,
        tension=0.0,
        situation_mode="coding",
        situation_confidence=0.9,
        interaction_mode="coding",
        workspace_root="E:/workspace",
    )


class _FakeOpencodeClient:
    async def create_session_and_prompt(self, **_kwargs: object) -> OpencodeJob:
        return OpencodeJob(
            job_id="opencode:job-1",
            session_id="session-1",
            project_id="project-1",
            model="hdsc/deepseek-v4-flash",
        )

    async def poll(self, job_id: str, *, max_chars: int) -> dict[str, object]:
        assert job_id == "opencode:job-1"
        assert max_chars == 4_000
        return {
            "session_id": "session-1",
            "project_id": "project-1",
            "state": "done",
            "summary": "diff verified",
            "exit_code": 0,
        }

    async def export(self, job_id: str, *, max_chars: int) -> dict[str, object]:
        assert job_id == "opencode:job-1"
        assert max_chars == 24_000
        return {
            "session_id": "session-1",
            "project_id": "project-1",
            "state": "done",
            "summary": "session trace",
            "exit_code": 0,
        }


@pytest.mark.asyncio
async def test_fire_check_export_are_bounded_and_marked() -> None:
    executor = OpencodeExecutor(
        OpencodeConfig(poll_max_output_chars=24_000),
        "",
        client=_FakeOpencodeClient(),
    )
    registry = ToolRegistry()
    executor.register_into(registry)
    context = _context()

    started = await executor.fire({"task": "run the focused tests"}, context)
    checked = await executor.check({"job_id": "opencode:job-1", "max_chars": 4_000}, context)
    exported = await executor.export({"job_id": "opencode:job-1"}, context)

    assert started.ok is True
    assert started.metadata["opencode_delegation"] is True
    assert started.metadata["job_id"] == "opencode:job-1"
    assert checked.ok is True
    assert checked.output == "diff verified"
    assert checked.metadata["state"] == "done"
    assert exported.output == "session trace"
    assert exported.metadata["export"] is True


def test_opencode_capabilities_are_gated_by_enabled_and_password() -> None:
    disabled = build_default_tool_kernel(
        ToolConfig(),
        opencode_config=OpencodeConfig(enabled=False),
        opencode_server_password="secret",
    )
    missing_password = build_default_tool_kernel(
        ToolConfig(),
        opencode_config=OpencodeConfig(enabled=True),
    )
    enabled = build_default_tool_kernel(
        ToolConfig(),
        opencode_config=OpencodeConfig(enabled=True),
        opencode_server_password="secret",
    )

    assert not {"opencode_fire", "opencode_check", "opencode_export"} & set(disabled.registry.names)
    assert not {"opencode_fire", "opencode_check", "opencode_export"} & set(
        missing_password.registry.names
    )
    assert {"opencode_fire", "opencode_check", "opencode_export"}.issubset(enabled.registry.names)


def test_opencode_settings_and_secret_are_environment_only() -> None:
    settings = load_settings(
        Environment.DEVELOPMENT,
        environ={
            "HDSC_OPENCODE_ENABLED": "true",
            "HDSC_OPENCODE_BASE_URL": "http://127.0.0.1:4096/",
            "HDSC_OPENCODE_FIRE_MAX_CONCURRENT": "3",
            "OPENCODE_SERVER_PASSWORD": "local-secret",
            "OPENCODE_SERVER_USERNAME": "zero",
            "HDSC_BROKER_TOKEN": "broker-secret",
        },
    )

    assert settings.opencode.enabled is True
    assert settings.opencode.base_url == "http://127.0.0.1:4096"
    assert settings.opencode.fire_max_concurrent == 3
    assert settings.secrets.opencode_server_password.get_secret_value() == "local-secret"
    assert settings.secrets.opencode_server_username == "zero"
    assert settings.secrets.broker_token.get_secret_value() == "broker-secret"
    assert "local-secret" not in json.dumps(settings.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_broker_auth_workspace_boundary_and_writeback(tmp_path: Path) -> None:
    class FakeSession:
        conversation_id = "conv-opencode"

        def record_opencode_writeback(self, *, session_id: str, workspace_root: str):
            return SimpleNamespace(id=f"writeback:{session_id}:{workspace_root}")

    app = FastAPI()
    app.include_router(opencode_broker_router(lambda: FakeSession(), "broker-secret"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://broker") as client:
        unauthorized = await client.post(
            "/api/opencode-broker/write",
            json={"root": str(workspace), "path": "main.py", "content": "print('ok')"},
        )
        written = await client.post(
            "/api/opencode-broker/write",
            headers={"authorization": "Bearer broker-secret"},
            json={"root": str(workspace), "path": "main.py", "content": "print('ok')"},
        )
        escaped = await client.post(
            "/api/opencode-broker/write",
            headers={"authorization": "Bearer broker-secret"},
            json={"root": str(workspace), "path": "../escape.py", "content": "bad"},
        )
        executed = await client.post(
            "/api/opencode-broker/exec",
            headers={"authorization": "Bearer broker-secret"},
            json={"root": str(workspace), "command": "Write-Output broker-ok"},
        )
        writeback = await client.post(
            "/api/opencode-broker/session-writeback",
            headers={"authorization": "Bearer broker-secret"},
            json={"sessionID": "session-1", "root": str(workspace)},
        )

    assert unauthorized.status_code == 401
    assert written.status_code == 200
    assert (workspace / "main.py").read_text(encoding="utf-8") == "print('ok')"
    assert escaped.status_code == 400
    assert executed.status_code == 200
    assert "broker-ok" in executed.json()["stdout"]
    assert writeback.json()["state"] == "recorded"
