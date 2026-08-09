"""Tests for the dependency-free core CLI."""

import os
from pathlib import Path

import pytest

from ssa.config import DatabaseConfig, Settings
from ssa.interfaces.cli import main


def test_cli_without_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "doctor" in output
    assert "hdsc" in output


def test_cli_help_exits_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    assert "init-db" in capsys.readouterr().out


def test_cli_init_and_doctor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "cli.db"

    assert main(["init-db", "--database", str(database_path)]) == 0
    assert database_path.exists()
    assert "schema v25" in capsys.readouterr().out

    settings = Settings(database=DatabaseConfig(path=str(database_path)))
    monkeypatch.setattr(
        "ssa.interfaces.cli._settings",
        lambda _environment, _database_path: settings,
    )
    monkeypatch.setattr("ssa.interfaces.cli.find_spec", lambda _name: None)

    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert "Schema:      v25" in output
    assert "sqlite-vec:  loaded" in output
    assert "LLM main: deepseek/deepseek-v4-flash" in output
    assert "LLM reasoning: deepseek/deepseek-v4-pro" in output
    assert "LLM endpoint: https://api.deepseek.com" in output
    assert "LLM thinking: disabled (effort=high)" in output
    assert "LiteLLM:     missing (add runtime extra)" in output
    assert "Multimodal:  gpt-5.6-luna" in output
    assert "MM endpoint: https://sky1818.com/v1" in output
    assert "MM secret:   not set" in output
    expected_win32 = "enabled" if os.name == "nt" else "disabled"
    assert f"Win32 tools: {expected_win32} (clipboard=on)" in output
    assert "Emotion lib:" in output
    assert "Soda music:" in output


def test_cli_launches_one_shot_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def launch(
        settings: Settings,
        conversation_id: str,
        *,
        once: bool,
        poll_seconds: float,
    ) -> int:
        captured.update(
            settings=settings,
            conversation_id=conversation_id,
            once=once,
            poll_seconds=poll_seconds,
        )
        return 0

    monkeypatch.setattr("ssa.interfaces.cli._launch_worker", launch)

    assert main(["worker", "--conversation", "background", "--once", "--poll-seconds", "0.5"]) == 0
    assert captured["conversation_id"] == "background"
    assert captured["once"] is True
    assert captured["poll_seconds"] == 0.5
