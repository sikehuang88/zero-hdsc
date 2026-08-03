"""HTTP contract coverage for the local Coding workspace routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from ssa.interfaces.coding_workspace_api import CodingFileWriteRequest, coding_workspace_router


@pytest.mark.asyncio
async def test_coding_workspace_http_contract_reads_writes_and_blocks_escape(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "main.ts"
    target.write_bytes(b"export const value = 1;\n")
    routes = {route.name: route.endpoint for route in coding_workspace_router().routes}

    tree = await routes["tree"](
        workspace_root=str(workspace),
        path=".",
        max_depth=5,
        max_entries=1_500,
    )
    opened = await routes["read_file"](
        workspace_root=str(workspace),
        path="main.ts",
    )
    saved = await routes["write_file"](
        CodingFileWriteRequest(
            workspace_root=str(workspace),
            path="main.ts",
            content="export const value = 2;\n",
            expected_mtime_ns=opened["mtime_ns"],
        )
    )
    assert tree["entries"][0]["path"] == "main.ts"
    assert opened["content"] == "export const value = 1;\n"
    assert saved["path"] == "main.ts"
    assert target.read_text(encoding="utf-8") == "export const value = 2;\n"

    with pytest.raises(HTTPException, match="outside"):
        await routes["read_file"](
            workspace_root=str(workspace),
            path="../outside.txt",
        )
