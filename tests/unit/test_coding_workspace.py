"""Coding workspace service boundaries and file safety."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from ssa.services.coding_workspace import (
    CodingWorkspaceBinaryError,
    CodingWorkspaceConflictError,
    CodingWorkspaceError,
    git_workspace_diff,
    git_workspace_status,
    list_workspace_tree,
    read_workspace_file,
    run_workspace_command,
    write_workspace_file,
)


def test_tree_uses_workspace_relative_paths_and_ignores_generated_directories(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "node_modules" / "pkg").mkdir(parents=True)
    (workspace / "src" / "app.ts").write_text("export const zero = true;\n", encoding="utf-8")
    (workspace / "node_modules" / "pkg" / "index.js").write_text("ignored", encoding="utf-8")

    tree = list_workspace_tree(str(workspace), max_depth=4)

    assert [entry["path"] for entry in tree["entries"]] == ["src", "src/app.ts"]
    assert tree["truncated"] is False


def test_tree_breadth_first_keeps_later_root_directories_visible(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    for name in ("alpha", "zero-web"):
        (workspace / name).mkdir(parents=True)
    for index in range(20):
        (workspace / "alpha" / f"file-{index}.txt").write_text("x", encoding="utf-8")

    tree = list_workspace_tree(str(workspace), max_depth=3, max_entries=3)

    assert [entry["path"] for entry in tree["entries"][:2]] == ["alpha", "zero-web"]


def test_file_read_write_detects_external_change_and_workspace_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "notes.md"
    target.write_text("alpha\n", encoding="utf-8")
    opened = read_workspace_file(str(workspace), "notes.md")

    saved = write_workspace_file(
        str(workspace),
        "notes.md",
        "beta\n",
        expected_mtime_ns=opened["mtime_ns"],
    )

    assert target.read_text(encoding="utf-8") == "beta\n"
    assert saved["mtime_ns"] != opened["mtime_ns"]
    with pytest.raises(CodingWorkspaceConflictError):
        write_workspace_file(
            str(workspace),
            "notes.md",
            "stale",
            expected_mtime_ns=opened["mtime_ns"],
        )
    with pytest.raises(CodingWorkspaceError, match="outside"):
        read_workspace_file(str(workspace), "../outside.txt")


def test_binary_file_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "asset.bin").write_bytes(b"zero\x00binary")

    with pytest.raises(CodingWorkspaceBinaryError):
        read_workspace_file(str(workspace), "asset.bin")


@pytest.mark.asyncio
async def test_workspace_command_runs_inside_selected_root(tmp_path: Path) -> None:
    result = await run_workspace_command(str(tmp_path), "Get-Location", timeout_seconds=10)

    assert result["exit_code"] == 0
    assert str(tmp_path).lower() in result["stdout"].lower()
    assert result["timed_out"] is False


@pytest.mark.asyncio
async def test_git_status_and_diff_are_structured(tmp_path: Path) -> None:
    try:
        subprocess.run(
            ["git", "init"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        pytest.skip("git is not installed")
    target = tmp_path / "tracked.txt"
    target.write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Zero Test",
            "-c",
            "user.email=zero@example.test",
            "commit",
            "-m",
            "initial",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    target.write_text("one\ntwo\n", encoding="utf-8")

    status, diff = await asyncio.gather(
        git_workspace_status(str(tmp_path)),
        git_workspace_diff(str(tmp_path), "tracked.txt"),
    )

    assert status["repository"] is True
    assert status["clean"] is False
    assert status["changes"][0]["path"] == "tracked.txt"
    assert "+two" in diff["diff"]
