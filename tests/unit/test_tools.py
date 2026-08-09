"""Unit coverage for the isolated global tool kernel."""

from __future__ import annotations

import json
import os
from importlib.util import find_spec
from pathlib import Path

import pytest
from pydantic import BaseModel

from ssa.adapters.llm import FunctionCall, ToolCall
from ssa.config import FirecrawlConfig, MultimodalConfig, ToolConfig, Win32Config
from ssa.tools.executors import GlobalToolExecutor, build_default_tool_kernel
from ssa.tools.mineradio import MineradioExecutor
from ssa.tools.models import ToolAutonomyContext
from ssa.tools.soda_music import _query_match_score, soda_music_enabled


class _NoArguments(BaseModel):
    pass


def _context() -> ToolAutonomyContext:
    return ToolAutonomyContext(
        correlation_id="tool-correlation",
        conversation_id="tool-conversation",
        energy=0.7,
        valence=0.2,
        arousal=0.4,
        trust=0.6,
        tension=0.1,
        situation_mode="clarify",
        situation_confidence=0.8,
    )


def _call(call_id: str, name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(
        id=call_id,
        function=FunctionCall(name=name, arguments=json.dumps(arguments)),
    )


def test_default_registry_exposes_stable_global_capabilities() -> None:
    kernel = build_default_tool_kernel(ToolConfig())

    expected = (
        "coding_git_status",
        "coding_project_tree",
        "coding_search",
        "generate_image",
        "mineradio_control",
        "mineradio_search_play",
        "mineradio_status",
        "powershell",
        "read_file",
        "truth_geocode",
        "truth_holidays",
        "truth_music",
        "truth_news",
        "truth_radio",
        "truth_service_status",
        "truth_sun_times",
        "truth_translate",
        "truth_weather",
        "web_close",
        "web_open",
        "web_read",
        "web_search",
        "web_tabs",
        "write_file",
    )
    if os.name == "nt" and soda_music_enabled() and find_spec("websockets") is not None:
        expected = expected[:9] + (
            "soda_music_control",
            "soda_music_login_status",
            "soda_music_now_playing",
            "soda_music_search_play",
        ) + expected[9:]
    assert kernel.registry.names == expected
    definitions = kernel.definitions()
    assert [item["function"]["name"] for item in definitions] == list(expected)
    assert all(item["function"]["parameters"]["type"] == "object" for item in definitions)
    assert all(
        item["function"]["parameters"]["additionalProperties"] is False for item in definitions
    )


@pytest.mark.parametrize(
    ("query", "state", "expected"),
    [
        (
            "周杰伦的晴天",
            {"found": True, "title": "晴天", "artist": "周杰伦", "album": "叶惠美"},
            2,
        ),
        (
            "周杰伦的晴天",
            {
                "found": True,
                "title": "晴天 (钢琴版) [原唱: 周杰伦]",
                "artist": "Bryan Chi",
                "album": "钢琴曲",
            },
            1,
        ),
        (
            "周杰伦 七里香",
            {"found": True, "title": "七里香", "artist": "周杰伦", "album": "七里香"},
            2,
        ),
        (
            "周杰伦的晴天",
            {"found": True, "title": "渐泠", "artist": "熹微", "album": ""},
            0,
        ),
    ],
)
def test_soda_music_query_match_score(
    query: str,
    state: dict[str, object],
    expected: int,
) -> None:
    assert _query_match_score(query, state) == expected


def test_mineradio_normalizes_provider_ids_without_cross_provider_fallback() -> None:
    kugou = MineradioExecutor._normalize_track(
        "kugou",
        {
            "id": "track-id",
            "name": "晴天",
            "artist": "周杰伦",
            "hash": "HASH",
            "albumId": "966846",
            "albumAudioId": "32100650",
        },
    )
    qq = MineradioExecutor._normalize_track(
        "qq",
        {"id": "001", "name": "晴天", "artist": "周杰伦"},
    )

    assert kugou["provider_ids"]["hash"] == "HASH"
    assert kugou["provider_ids"]["album_id"] == "966846"
    assert kugou["provider_ids"]["album_audio_id"] == "32100650"
    assert qq["provider_ids"]["hash"] == ""


def test_mineradio_public_playback_exposes_only_local_audio_proxy() -> None:
    playback = MineradioExecutor._public_playback(
        {
            "provider": "netease",
            "playable": True,
            "url": "https://media.example/audio.mp3?token=secret value",
            "quality": "higher",
        }
    )

    assert playback == {
        "playable": True,
        "provider": "netease",
        "trial": False,
        "reason": "",
        "message": "",
        "audio_path": (
            "/mineradio/api/audio?url="
            "https%3A%2F%2Fmedia.example%2Faudio.mp3%3Ftoken%3Dsecret%20value"
        ),
        "quality": "higher",
    }


@pytest.mark.asyncio
async def test_coding_tools_inspect_search_and_bound_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    source = workspace / "src"
    source.mkdir(parents=True)
    (source / "app.py").write_text("def launch():\n    return 'zero'\n", encoding="utf-8")
    context = _context().model_copy(
        update={"interaction_mode": "coding", "workspace_root": str(workspace)}
    )
    kernel = build_default_tool_kernel(ToolConfig())

    tree = await kernel.execute(
        _call("tree-1", "coding_project_tree", {"max_depth": 3}),
        context,
    )
    search = await kernel.execute(
        _call("search-1", "coding_search", {"query": "launch", "path": "src"}),
        context,
    )
    escaped = await kernel.execute(
        _call("tree-2", "coding_project_tree", {"path": ".."}),
        context,
    )

    assert tree.ok is True
    assert '"path": "src/app.py"' in tree.output
    assert search.ok is True
    assert '"path": "app.py"' in search.output
    assert '"line": 1' in search.output
    assert escaped.ok is False
    assert "outside the coding workspace" in (escaped.error or "")


def test_gpt_bridge_registers_only_when_key_is_configured() -> None:
    without_key = build_default_tool_kernel(
        ToolConfig(),
        multimodal_config=MultimodalConfig(),
    )
    with_key = build_default_tool_kernel(
        ToolConfig(),
        multimodal_config=MultimodalConfig(),
        multimodal_api_key="sk-test-only",
    )

    assert "ask_gpt" not in without_key.registry.names
    assert "ask_gpt" in with_key.registry.names
    definition = next(
        item for item in with_key.definitions() if item["function"]["name"] == "ask_gpt"
    )
    properties = definition["function"]["parameters"]["properties"]
    assert set(properties) == {"prompt", "instructions", "paths"}


def test_firecrawl_registers_only_when_key_is_configured() -> None:
    without_key = build_default_tool_kernel(
        ToolConfig(),
        firecrawl_config=FirecrawlConfig(),
    )
    with_key = build_default_tool_kernel(
        ToolConfig(),
        firecrawl_config=FirecrawlConfig(),
        firecrawl_api_key="fc-test-only",
    )

    assert "firecrawl_search" not in without_key.registry.names
    assert "firecrawl_scrape" not in without_key.registry.names
    assert {"firecrawl_search", "firecrawl_scrape"}.issubset(with_key.registry.names)


@pytest.mark.skipif(os.name != "nt", reason="native registry is Windows-only")
def test_win32_registry_is_added_only_when_explicitly_configured() -> None:
    without_win32 = build_default_tool_kernel(ToolConfig())
    with_win32 = build_default_tool_kernel(
        ToolConfig(),
        win32_config=Win32Config(),
    )

    assert not any(name.startswith("win32_") for name in without_win32.registry.names)
    assert [name for name in with_win32.registry.names if name.startswith("win32_")] == [
        "win32_active_window",
        "win32_clipboard_text",
        "win32_drives",
        "win32_list_directory",
        "win32_processes",
        "win32_visible_windows",
    ]


@pytest.mark.asyncio
async def test_global_file_tools_write_append_and_read(tmp_path: Path) -> None:
    kernel = build_default_tool_kernel(ToolConfig())
    target = tmp_path / "outside-workspace-root" / "state.txt"

    created = await kernel.execute(
        _call(
            "write-1",
            "write_file",
            {"path": str(target), "content": "alpha", "mode": "overwrite"},
        ),
        _context(),
    )
    appended = await kernel.execute(
        _call(
            "write-2",
            "write_file",
            {"path": str(target), "content": "-beta", "mode": "append"},
        ),
        _context(),
    )
    read = await kernel.execute(
        _call("read-1", "read_file", {"path": str(target)}),
        _context(),
    )

    assert created.ok is True
    assert appended.ok is True
    assert read.ok is True
    assert read.output == "alpha-beta"
    assert Path(str(read.metadata["path"])).is_absolute()


@pytest.mark.asyncio
async def test_powershell_executes_and_captures_output(tmp_path: Path) -> None:
    kernel = build_default_tool_kernel(ToolConfig())

    result = await kernel.execute(
        _call(
            "shell-1",
            "powershell",
            {
                "command": "Write-Output 'kernel-ok'",
                "working_directory": str(tmp_path),
                "timeout_seconds": 30,
            },
        ),
        _context(),
    )

    assert result.ok is True
    assert result.exit_code == 0
    assert "kernel-ok" in result.output
    assert result.elevated is False


@pytest.mark.asyncio
async def test_elevation_can_be_disabled_without_launching_uac() -> None:
    kernel = build_default_tool_kernel(ToolConfig(allow_elevation=False))

    result = await kernel.execute(
        _call(
            "shell-elevated",
            "powershell",
            {"command": "Write-Output elevated", "elevated": True},
        ),
        _context(),
    )

    assert result.ok is False
    assert result.elevated is False
    assert result.error == "elevated PowerShell is disabled"


def test_elevation_launcher_uses_windows_runas(tmp_path: Path) -> None:
    runner = tmp_path / "runner.ps1"

    command = GlobalToolExecutor._elevation_launcher_command("powershell.exe", runner)

    assert "Start-Process" in command
    assert "-Verb RunAs" in command
    assert str(runner) in command


@pytest.mark.asyncio
async def test_unknown_tool_and_invalid_arguments_are_isolated() -> None:
    kernel = build_default_tool_kernel(ToolConfig())

    unknown = await kernel.execute(_call("bad-1", "missing", {}), _context())
    invalid = await kernel.execute(
        ToolCall(
            id="bad-2",
            function=FunctionCall(name="read_file", arguments="[]"),
        ),
        _context(),
    )

    assert unknown.ok is False
    assert "unknown tool" in (unknown.error or "")
    assert invalid.ok is False
    assert "JSON object" in (invalid.error or "")


@pytest.mark.asyncio
async def test_tool_kernel_reports_original_output_size_and_truncation() -> None:
    from ssa.tools.models import ToolOutcome
    from ssa.tools.registry import ToolCapability, ToolRegistry

    async def long_output(
        _arguments: dict[str, object],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        return ToolOutcome(ok=True, output="abcdefghij")

    registry = ToolRegistry()
    registry.register(
        ToolCapability(
            name="long_output",
            description="return bounded text",
            arguments_model=_NoArguments,
            handler=long_output,
        )
    )
    from ssa.tools.kernel import ToolKernel

    result = await ToolKernel(registry, max_output_chars=5).execute(
        _call("long-1", "long_output", {}),
        _context(),
    )

    assert result.ok is True
    assert result.output.startswith("abcde")
    assert result.output_chars == 10
    assert result.truncated is True
