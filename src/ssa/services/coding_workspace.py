"""Bounded filesystem and Git operations for the Coding workspace UI."""

from __future__ import annotations

import asyncio
import os
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_IGNORED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "release",
}
_MAX_FILE_BYTES = 2 * 1024 * 1024
_MAX_COMMAND_OUTPUT = 256 * 1024
_MAX_DIFF_CHARS = 320_000


class CodingWorkspaceError(ValueError):
    """Base error surfaced as a client-visible workspace request failure."""


class CodingWorkspaceNotFoundError(CodingWorkspaceError):
    pass


class CodingWorkspaceBinaryError(CodingWorkspaceError):
    pass


class CodingWorkspaceTooLargeError(CodingWorkspaceError):
    pass


class CodingWorkspaceConflictError(CodingWorkspaceError):
    pass


def resolve_workspace_root(raw_root: str) -> Path:
    root = Path(raw_root).expanduser().resolve(strict=False)
    if not root.exists():
        raise CodingWorkspaceNotFoundError(f"workspace does not exist: {root}")
    if not root.is_dir():
        raise CodingWorkspaceError(f"workspace is not a directory: {root}")
    return root


def resolve_workspace_path(raw_root: str, raw_path: str = ".") -> tuple[Path, Path]:
    root = resolve_workspace_root(raw_root)
    candidate = Path(raw_path).expanduser()
    resolved = (
        candidate.resolve(strict=False)
        if candidate.is_absolute()
        else (root / candidate).resolve(strict=False)
    )
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CodingWorkspaceError(f"path is outside the coding workspace: {resolved}") from exc
    return root, resolved


def list_workspace_tree(
    workspace_root: str,
    *,
    path: str = ".",
    max_depth: int = 5,
    max_entries: int = 1_500,
    include_hidden: bool = False,
) -> dict[str, Any]:
    root, start = resolve_workspace_path(workspace_root, path)
    if not start.exists():
        raise CodingWorkspaceNotFoundError(f"directory does not exist: {start}")
    if not start.is_dir():
        raise CodingWorkspaceError(f"path is not a directory: {start}")

    entries: list[dict[str, object]] = []
    queue: list[tuple[Path, int]] = [(start, 1)]
    while queue and len(entries) < max_entries:
        directory, depth = queue.pop(0)
        if depth > max_depth:
            continue
        try:
            children = sorted(
                directory.iterdir(),
                key=lambda item: (not item.is_dir(), item.name.lower()),
            )
        except OSError:
            continue
        for child in children:
            if len(entries) >= max_entries:
                break
            if not include_hidden and child.name.startswith("."):
                continue
            if child.is_symlink():
                continue
            is_directory = child.is_dir()
            if is_directory and child.name in _IGNORED_DIRECTORIES:
                continue
            record: dict[str, object] = {
                "name": child.name,
                "path": child.relative_to(root).as_posix(),
                "kind": "directory" if is_directory else "file",
            }
            if not is_directory:
                with suppress(OSError):
                    record["bytes"] = child.stat().st_size
            entries.append(record)
            if is_directory:
                queue.append((child, depth + 1))
    return {
        "root": str(root),
        "path": start.relative_to(root).as_posix() or ".",
        "entries": entries,
        "truncated": len(entries) >= max_entries,
    }


def read_workspace_file(
    workspace_root: str,
    path: str,
    *,
    max_bytes: int = _MAX_FILE_BYTES,
) -> dict[str, Any]:
    root, target = resolve_workspace_path(workspace_root, path)
    if not target.exists():
        raise CodingWorkspaceNotFoundError(f"file does not exist: {target}")
    if not target.is_file():
        raise CodingWorkspaceError(f"path is not a file: {target}")
    stat = target.stat()
    if stat.st_size > max_bytes:
        raise CodingWorkspaceTooLargeError(
            f"file is too large for the editor: {stat.st_size} bytes (limit {max_bytes})"
        )
    payload = target.read_bytes()
    if b"\x00" in payload[:8_192]:
        raise CodingWorkspaceBinaryError(f"binary file preview is not supported: {target.name}")
    try:
        content = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CodingWorkspaceBinaryError(
            f"file is not valid UTF-8 text: {target.name}"
        ) from exc
    return {
        "root": str(root),
        "path": target.relative_to(root).as_posix(),
        "content": content,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "line_count": content.count("\n") + 1,
        "encoding": "utf-8",
    }


def write_workspace_file(
    workspace_root: str,
    path: str,
    content: str,
    *,
    expected_mtime_ns: int | None,
) -> dict[str, Any]:
    root, target = resolve_workspace_path(workspace_root, path)
    if not target.exists():
        raise CodingWorkspaceNotFoundError(f"file does not exist: {target}")
    if not target.is_file():
        raise CodingWorkspaceError(f"path is not a file: {target}")
    current = target.stat()
    if expected_mtime_ns is not None and current.st_mtime_ns != expected_mtime_ns:
        raise CodingWorkspaceConflictError(
            "file changed on disk; reload it before saving your edits"
        )
    payload = content.encode("utf-8")
    if len(payload) > _MAX_FILE_BYTES:
        raise CodingWorkspaceTooLargeError(
            f"file is too large for the editor: {len(payload)} bytes (limit {_MAX_FILE_BYTES})"
        )
    temporary = target.with_name(f".{target.name}.zero-save-{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.chmod(temporary, current.st_mode)
        os.replace(temporary, target)
    finally:
        with suppress(OSError):
            temporary.unlink()
    stat = target.stat()
    return {
        "root": str(root),
        "path": target.relative_to(root).as_posix(),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "line_count": content.count("\n") + 1,
        "encoding": "utf-8",
    }


async def search_workspace(
    workspace_root: str,
    query: str,
    *,
    path: str = ".",
    max_results: int = 120,
) -> dict[str, Any]:
    root, start = resolve_workspace_path(workspace_root, path)
    if not start.exists():
        raise CodingWorkspaceNotFoundError(f"search path does not exist: {start}")
    command = [
        "rg",
        "--json",
        "--line-number",
        "--color",
        "never",
        "--smart-case",
        "--hidden",
    ]
    for ignored in sorted(_IGNORED_DIRECTORIES):
        command.extend(["--glob", f"!{ignored}/**"])
    command.extend(["--", query, str(start)])
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except FileNotFoundError as exc:
        raise CodingWorkspaceError("ripgrep executable was not found") from exc
    stdout, stderr = await process.communicate()
    if process.returncode not in {0, 1}:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise CodingWorkspaceError(detail or "workspace search failed")
    matches: list[dict[str, object]] = []
    import json

    for raw_line in stdout.decode("utf-8", errors="replace").splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue
        data = event.get("data", {})
        path_text = data.get("path", {}).get("text", "")
        try:
            relative = Path(path_text).resolve().relative_to(root).as_posix()
        except (OSError, ValueError):
            continue
        matches.append(
            {
                "path": relative,
                "line": int(data.get("line_number") or 1),
                "text": str(data.get("lines", {}).get("text", "")).rstrip("\r\n")[:500],
            }
        )
        if len(matches) >= max_results:
            break
    return {"root": str(root), "matches": matches, "count": len(matches)}


async def git_workspace_status(workspace_root: str) -> dict[str, Any]:
    root = resolve_workspace_root(workspace_root)
    process = await _run_process(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--short",
            "--branch",
            "--untracked-files=all",
            "--",
            ".",
        ]
    )
    if process.exit_code != 0:
        if "not a git repository" in process.stderr.lower():
            return {
                "root": str(root),
                "repository": False,
                "branch": "",
                "clean": True,
                "changes": [],
            }
        raise CodingWorkspaceError(process.stderr or "git status failed")
    lines = process.stdout.splitlines()
    branch = lines[0][3:] if lines and lines[0].startswith("## ") else ""
    changes = []
    for line in lines[1:] if branch else lines:
        if len(line) < 3:
            continue
        status = line[:2]
        change_path = line[3:]
        changes.append(
            {
                "status": status,
                "path": change_path.split(" -> ")[-1],
                "label": _git_status_label(status),
            }
        )
    total_changes = len(changes)
    visible_changes = changes[:600]
    return {
        "root": str(root),
        "repository": True,
        "branch": branch,
        "clean": not changes,
        "changes": visible_changes,
        "total_changes": total_changes,
        "truncated": total_changes > len(visible_changes),
    }


async def git_workspace_diff(workspace_root: str, path: str) -> dict[str, Any]:
    root, target = resolve_workspace_path(workspace_root, path)
    relative = target.relative_to(root).as_posix()
    unstaged = await _run_process(
        ["git", "-C", str(root), "diff", "--no-ext-diff", "--relative", "--", relative]
    )
    staged = await _run_process(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--cached",
            "--no-ext-diff",
            "--relative",
            "--",
            relative,
        ]
    )
    for result in (unstaged, staged):
        if result.exit_code != 0:
            raise CodingWorkspaceError(result.stderr or "git diff failed")
    sections = []
    if staged.stdout:
        sections.append("# STAGED\n" + staged.stdout)
    if unstaged.stdout:
        sections.append("# WORKTREE\n" + unstaged.stdout)
    diff = "\n\n".join(sections)
    truncated = len(diff) > _MAX_DIFF_CHARS
    if truncated:
        diff = diff[:_MAX_DIFF_CHARS] + "\n\n... diff truncated ..."
    return {"root": str(root), "path": relative, "diff": diff, "truncated": truncated}


async def run_workspace_command(
    workspace_root: str,
    command: str,
    *,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    root = resolve_workspace_root(workspace_root)
    if not command.strip():
        raise CodingWorkspaceError("command must not be empty")
    if os.name == "nt":
        argv = [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "$OutputEncoding = [Console]::OutputEncoding = "
            "[Text.UTF8Encoding]::new($false); " + command,
        ]
    else:
        argv = ["/bin/sh", "-lc", command]
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except FileNotFoundError as exc:
        raise CodingWorkspaceError("command shell executable was not found") from exc
    timed_out = False
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_seconds)
    except TimeoutError:
        timed_out = True
        process.kill()
        stdout, stderr = await process.communicate()
    output = stdout.decode("utf-8-sig", errors="replace")
    error = stderr.decode("utf-8-sig", errors="replace")
    if len(output) > _MAX_COMMAND_OUTPUT:
        output = output[:_MAX_COMMAND_OUTPUT] + "\n... stdout truncated ..."
    if len(error) > _MAX_COMMAND_OUTPUT:
        error = error[:_MAX_COMMAND_OUTPUT] + "\n... stderr truncated ..."
    return {
        "root": str(root),
        "command": command,
        "exit_code": process.returncode,
        "stdout": output,
        "stderr": error,
        "timed_out": timed_out,
    }


@dataclass(frozen=True)
class _ProcessResult:
    exit_code: int
    stdout: str
    stderr: str


async def _run_process(argv: list[str]) -> _ProcessResult:
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except FileNotFoundError as exc:
        raise CodingWorkspaceError(f"executable was not found: {argv[0]}") from exc
    stdout, stderr = await process.communicate()
    return _ProcessResult(
        exit_code=process.returncode or 0,
        stdout=stdout.decode("utf-8", errors="replace").strip(),
        stderr=stderr.decode("utf-8", errors="replace").strip(),
    )


def _git_status_label(status: str) -> str:
    if status == "??":
        return "未跟踪"
    if "U" in status:
        return "冲突"
    if "D" in status:
        return "删除"
    if "R" in status:
        return "重命名"
    if "A" in status:
        return "新增"
    if "M" in status:
        return "修改"
    return status.strip() or "变更"


__all__ = [
    "CodingWorkspaceBinaryError",
    "CodingWorkspaceConflictError",
    "CodingWorkspaceError",
    "CodingWorkspaceNotFoundError",
    "CodingWorkspaceTooLargeError",
    "git_workspace_diff",
    "git_workspace_status",
    "list_workspace_tree",
    "read_workspace_file",
    "resolve_workspace_path",
    "resolve_workspace_root",
    "run_workspace_command",
    "search_workspace",
    "write_workspace_file",
]
