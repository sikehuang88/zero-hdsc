"""Unit coverage for the slash-command capability kernel."""

from __future__ import annotations

import pytest

from ssa.commands import CommandCapability, CommandRegistry, build_default_command_registry


def test_default_registry_resolves_commands_aliases_and_arguments() -> None:
    registry = build_default_command_registry()

    activity = registry.parse("/activity")
    attach = registry.parse('/attach "E:\\research\\paper.pdf"')
    exit_alias = registry.parse("/exit")

    assert activity is not None
    assert activity.capability.action == "tab.open"
    assert activity.capability.payload == "activity-tab"
    assert attach is not None
    assert attach.capability.requires_argument is True
    assert attach.argument == '"E:\\research\\paper.pdf"'
    assert exit_alias is not None
    assert exit_alias.capability.name == "quit"
    assert registry.parse("not-a-command") is None
    assert registry.parse("/missing") is None


def test_prefix_suggestions_include_usage_and_filter_after_arguments() -> None:
    registry = build_default_command_registry()

    all_commands = registry.suggest("/")
    activity = registry.suggest("/act")
    alias = registry.suggest("/comm")

    assert len(all_commands) == 8
    assert [item.name for item in activity] == ["activity"]
    assert [item.name for item in alias] == ["help"]
    assert registry.suggest("/attach E:\\file.png") == ()
    assert registry.get("/exit") == registry.get("quit")
    attach = registry.get("attach")
    assert attach is not None
    assert attach.usage == "/attach PATH"


def test_registry_rejects_duplicate_tokens_and_invalid_names() -> None:
    registry = CommandRegistry()
    registry.register(CommandCapability("help", "Show help", "help.show", aliases=("h",)))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(CommandCapability("history", "History", "history.show", aliases=("h",)))
    with pytest.raises(ValueError, match="lowercase"):
        CommandCapability("Bad_Name", "Bad", "bad.action")
