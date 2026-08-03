"""Structured workspace tools used by the interactive coding mode."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from contextlib import suppress
from pathlib import Path
from typing import Any

from ssa.tools.models import (
    CodingGitStatusArguments,
    CodingProjectTreeArguments,
    CodingSearchArguments,
    ToolAutonomyContext,
    ToolOutcome,
)
from ssa.tools.registry import ToolCapability, ToolRegistry

_IGNORED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "dist",
    "node_modules",
    "release",
}


class CodingToolExecutor:
    def register_into(self, registry: ToolRegistry) -> None:
        registry.register(
            ToolCapability(
                name="coding_project_tree",
                description=(
                    "Inspect a bounded project tree in the active coding workspace. Use this "
                    "before editing when the repository structure is not already known."
                ),
                arguments_model=CodingProjectTreeArguments,
                handler=self.project_tree,
            )
        )
        registry.register(
            ToolCapability(
                name="coding_search",
                description=(
                    "Search source text with ripgrep in the active coding workspace and return "
                    "structured file, line, and matching text records."
                ),
                arguments_model=CodingSearchArguments,
                handler=self.search,
            )
        )
        registry.register(
            ToolCapability(
                name="coding_git_status",
                description=(
                    "Read the current Git branch and porcelain status for a repository inside "
                    "the active coding workspace."
                ),
                arguments_model=CodingGitStatusArguments,
                handler=self.git_status,
            )
        )

    async def project_tree(
        self,
        raw_arguments: dict[str, Any],
        context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = CodingProjectTreeArguments.model_validate(raw_arguments)
        root = _workspace_path(context, arguments.path)
        if not root.is_dir():
            return ToolOutcome(ok=False, error=f"directory does not exist: {root}")
        entries: list[dict[str, object]] = []

        def walk(directory: Path, depth: int) -> None:
            if depth > arguments.max_depth or len(entries) >= arguments.max_entries:
                return
            try:
                children = sorted(
                    directory.iterdir(),
                    key=lambda item: (not item.is_dir(), item.name.lower()),
                )
            except OSError:
                return
            for child in children:
                if len(entries) >= arguments.max_entries:
                    return
                if not arguments.include_hidden and child.name.startswith("."):
                    continue
                if child.is_dir() and child.name in _IGNORED_DIRECTORIES:
                    continue
                record: dict[str, object] = {
                    "path": child.relative_to(root).as_posix(),
                    "kind": "directory" if child.is_dir() else "file",
                }
                if child.is_file():
                    with suppress(OSError):
                        record["bytes"] = child.stat().st_size
                entries.append(record)
                if child.is_dir():
                    walk(child, depth + 1)

        walk(root, 1)
        return ToolOutcome(
            ok=True,
            output=json.dumps(
                {
                    "root": str(root),
                    "entries": entries,
                    "truncated": len(entries) >= arguments.max_entries,
                },
                ensure_ascii=False,
            ),
            metadata={"workspace_root": str(_workspace_root(context)), "path": str(root)},
        )

    async def search(
        self,
        raw_arguments: dict[str, Any],
        context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = CodingSearchArguments.model_validate(raw_arguments)
        root = _workspace_path(context, arguments.path)
        command = [
            "rg",
            "--json",
            "--line-number",
            "--color",
            "never",
            "--max-count",
            str(arguments.max_results),
        ]
        if arguments.case_sensitive:
            command.append("--case-sensitive")
        else:
            command.append("--smart-case")
        if arguments.glob:
            command.extend(["--glob", arguments.glob])
        command.extend(["--", arguments.query, str(root)])
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except FileNotFoundError:
            return ToolOutcome(ok=False, error="ripgrep executable was not found")
        stdout, stderr = await process.communicate()
        if process.returncode not in {0, 1}:
            return ToolOutcome(
                ok=False,
                error=stderr.decode("utf-8", errors="replace").strip() or "ripgrep failed",
                exit_code=process.returncode,
            )
        matches: list[dict[str, object]] = []
        for raw_line in stdout.decode("utf-8", errors="replace").splitlines():
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "match":
                continue
            data = event.get("data", {})
            path_text = data.get("path", {}).get("text", "")
            line_text = data.get("lines", {}).get("text", "").rstrip("\r\n")
            try:
                relative = Path(path_text).resolve().relative_to(root).as_posix()
            except (OSError, ValueError):
                relative = path_text
            matches.append(
                {
                    "path": relative,
                    "line": data.get("line_number"),
                    "text": line_text[:500],
                }
            )
            if len(matches) >= arguments.max_results:
                break
        return ToolOutcome(
            ok=True,
            output=json.dumps(
                {"root": str(root), "matches": matches, "count": len(matches)},
                ensure_ascii=False,
            ),
            exit_code=process.returncode,
            metadata={"workspace_root": str(_workspace_root(context)), "path": str(root)},
        )

    async def git_status(
        self,
        raw_arguments: dict[str, Any],
        context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = CodingGitStatusArguments.model_validate(raw_arguments)
        root = _workspace_path(context, arguments.path)
        command = [
            "git",
            "-C",
            str(root),
            "status",
            "--short",
            "--branch",
            f"--untracked-files={'all' if arguments.include_untracked else 'no'}",
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except FileNotFoundError:
            return ToolOutcome(ok=False, error="git executable was not found")
        stdout, stderr = await process.communicate()
        output = stdout.decode("utf-8", errors="replace")
        error = stderr.decode("utf-8", errors="replace").strip()
        if process.returncode != 0:
            return ToolOutcome(ok=False, error=error or "git status failed")
        lines = output.splitlines()
        return ToolOutcome(
            ok=True,
            output=json.dumps(
                {
                    "root": str(root),
                    "branch": lines[0][3:] if lines and lines[0].startswith("## ") else "",
                    "changes": lines[1:] if lines and lines[0].startswith("## ") else lines,
                    "clean": len(lines) <= 1,
                },
                ensure_ascii=False,
            ),
            exit_code=0,
            metadata={"workspace_root": str(_workspace_root(context)), "path": str(root)},
        )


def _workspace_root(context: ToolAutonomyContext) -> Path:
    root = context.workspace_root or os.getcwd()
    return Path(root).expanduser().resolve(strict=False)


def _workspace_path(context: ToolAutonomyContext, raw_path: str) -> Path:
    root = _workspace_root(context)
    candidate = Path(raw_path).expanduser()
    resolved = (
        candidate.resolve(strict=False)
        if candidate.is_absolute()
        else (root / candidate).resolve(strict=False)
    )
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path is outside the coding workspace: {resolved}") from exc
    return resolved


__all__ = ["CodingToolExecutor"]
