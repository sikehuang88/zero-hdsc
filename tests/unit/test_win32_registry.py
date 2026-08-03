"""Unit coverage for the native Windows capability registry."""

from __future__ import annotations

import json
from typing import Any

import pytest

from ssa.adapters.llm import FunctionCall, ToolCall
from ssa.config import Win32Config
from ssa.tools.kernel import ToolKernel
from ssa.tools.models import ToolAutonomyContext
from ssa.tools.registry import ToolRegistry
from ssa.tools.win32_registry import Win32ToolExecutor


class _FakeWin32Backend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def active_window(self) -> dict[str, Any]:
        self.calls.append(("active_window", (), {}))
        return {
            "observed_at_ms": 1_900_000_000_000,
            "title": "Research - Editor",
            "process_id": 42,
            "process_name": "editor.exe",
            "idle_ms": 1200,
        }

    def visible_windows(self, *, limit: int, include_untitled: bool) -> dict[str, Any]:
        self.calls.append(
            ("visible_windows", (), {"limit": limit, "include_untitled": include_untitled})
        )
        return {"observed_at_ms": 1, "count": 1, "windows": [{"title": "Editor"}]}

    def processes(self, *, limit: int, include_paths: bool) -> dict[str, Any]:
        self.calls.append(("processes", (), {"limit": limit, "include_paths": include_paths}))
        return {"observed_at_ms": 2, "count": 1, "processes": [{"process_id": 42}]}

    def drives(self) -> dict[str, Any]:
        self.calls.append(("drives", (), {}))
        return {"observed_at_ms": 3, "count": 1, "drives": [{"root": "C:\\"}]}

    def list_directory(
        self,
        path: str,
        *,
        pattern: str,
        limit: int,
        include_hidden: bool,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "directory",
                (path,),
                {"pattern": pattern, "limit": limit, "include_hidden": include_hidden},
            )
        )
        return {
            "observed_at_ms": 4,
            "path": path,
            "count": 1,
            "entries": [{"name": "paper.pdf"}],
        }

    def clipboard_text(self, *, max_chars: int) -> dict[str, Any]:
        self.calls.append(("clipboard", (), {"max_chars": max_chars}))
        return {"observed_at_ms": 5, "text_available": True, "text": "draft"}


def _context() -> ToolAutonomyContext:
    return ToolAutonomyContext(
        correlation_id="win32-correlation",
        conversation_id="win32-conversation",
        energy=0.7,
        valence=0.0,
        arousal=0.2,
        trust=0.8,
        tension=0.1,
        situation_mode="observe",
        situation_confidence=0.9,
    )


def _call(name: str, arguments: dict[str, object] | None = None) -> ToolCall:
    return ToolCall(
        id=f"call-{name}",
        function=FunctionCall(name=name, arguments=json.dumps(arguments or {})),
    )


def _kernel(config: Win32Config, backend: _FakeWin32Backend) -> ToolKernel:
    registry = ToolRegistry()
    Win32ToolExecutor(config, backend=backend).register_into(registry)
    return ToolKernel(registry)


def test_registry_exposes_read_only_native_capabilities() -> None:
    backend = _FakeWin32Backend()
    kernel = _kernel(Win32Config(), backend)

    assert kernel.registry.names == (
        "win32_active_window",
        "win32_clipboard_text",
        "win32_drives",
        "win32_list_directory",
        "win32_processes",
        "win32_visible_windows",
    )
    assert all(
        item["function"]["name"].startswith("win32_") for item in kernel.definitions()
    )


@pytest.mark.asyncio
async def test_native_observations_are_structured_and_limits_are_clamped() -> None:
    backend = _FakeWin32Backend()
    kernel = _kernel(
        Win32Config(
            max_windows=3,
            max_processes=4,
            max_directory_entries=5,
            max_clipboard_chars=256,
        ),
        backend,
    )

    active = await kernel.execute(_call("win32_active_window"), _context())
    windows = await kernel.execute(
        _call("win32_visible_windows", {"limit": 100, "include_untitled": True}),
        _context(),
    )
    processes = await kernel.execute(
        _call("win32_processes", {"limit": 100, "include_paths": False}),
        _context(),
    )
    directory = await kernel.execute(
        _call(
            "win32_list_directory",
            {"path": "E:\\research", "pattern": "*.pdf", "limit": 100},
        ),
        _context(),
    )
    clipboard = await kernel.execute(
        _call("win32_clipboard_text", {"max_chars": 10_000}),
        _context(),
    )

    assert json.loads(active.output)["process_name"] == "editor.exe"
    assert active.metadata["host_observation"] is True
    assert active.metadata["observation_kind"] == "active_window"
    assert json.loads(directory.output)["entries"][0]["name"] == "paper.pdf"
    assert windows.ok and processes.ok and clipboard.ok
    assert ("visible_windows", (), {"limit": 3, "include_untitled": True}) in backend.calls
    assert ("processes", (), {"limit": 4, "include_paths": False}) in backend.calls
    assert (
        "directory",
        ("E:\\research",),
        {"pattern": "*.pdf", "limit": 5, "include_hidden": False},
    ) in backend.calls
    assert ("clipboard", (), {"max_chars": 256}) in backend.calls


def test_clipboard_capability_can_be_removed_from_registry() -> None:
    kernel = _kernel(Win32Config(clipboard_enabled=False), _FakeWin32Backend())

    assert "win32_clipboard_text" not in kernel.registry.names
