"""Stable slash-command capabilities shared by interactive interfaces."""

from __future__ import annotations

import re
from dataclasses import dataclass

_COMMAND_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")


@dataclass(frozen=True)
class CommandCapability:
    name: str
    description: str
    action: str
    argument_hint: str = ""
    aliases: tuple[str, ...] = ()
    payload: str = ""

    def __post_init__(self) -> None:
        values = (self.name, *self.aliases)
        if any(_COMMAND_NAME.fullmatch(value) is None for value in values):
            raise ValueError("command names and aliases must use lowercase letters, digits, or hyphens")
        if not self.description.strip() or not self.action.strip():
            raise ValueError("command description and action must not be empty")
        if self.argument_hint and not self.argument_hint.strip():
            raise ValueError("command argument hint must not be blank")

    @property
    def requires_argument(self) -> bool:
        return bool(self.argument_hint)

    @property
    def usage(self) -> str:
        suffix = f" {self.argument_hint}" if self.argument_hint else ""
        return f"/{self.name}{suffix}"


@dataclass(frozen=True)
class ParsedCommand:
    capability: CommandCapability
    argument: str = ""


class CommandRegistry:
    """Registration, alias resolution, parsing, and prefix completion."""

    def __init__(self) -> None:
        self._commands: dict[str, CommandCapability] = {}
        self._tokens: dict[str, str] = {}

    def register(self, capability: CommandCapability) -> None:
        tokens = (capability.name, *capability.aliases)
        duplicate = next((token for token in tokens if token in self._tokens), None)
        if duplicate is not None:
            raise ValueError(f"slash-command token {duplicate!r} is already registered")
        self._commands[capability.name] = capability
        for token in tokens:
            self._tokens[token] = capability.name

    def get(self, name: str) -> CommandCapability | None:
        canonical = self._tokens.get(name.casefold().removeprefix("/"))
        return self._commands.get(canonical) if canonical is not None else None

    def parse(self, raw: str) -> ParsedCommand | None:
        value = raw.strip()
        if not value.startswith("/"):
            return None
        token, separator, argument = value[1:].partition(" ")
        capability = self.get(token)
        if capability is None:
            return None
        return ParsedCommand(
            capability=capability,
            argument=argument.strip() if separator else "",
        )

    def suggest(self, raw: str, *, limit: int = 8) -> tuple[CommandCapability, ...]:
        if limit < 1:
            return ()
        value = raw.lstrip()
        if not value.startswith("/"):
            return ()
        fragment = value[1:].casefold()
        if " " in fragment:
            return ()
        matches = [
            capability
            for capability in self._commands.values()
            if capability.name.startswith(fragment)
            or any(alias.startswith(fragment) for alias in capability.aliases)
        ]
        matches.sort(key=lambda item: (not item.name.startswith(fragment), item.name))
        return tuple(matches[:limit])

    @property
    def capabilities(self) -> tuple[CommandCapability, ...]:
        return tuple(self._commands[name] for name in sorted(self._commands))


def build_default_command_registry() -> CommandRegistry:
    registry = CommandRegistry()
    for capability in (
        CommandCapability(
            "attach",
            "Queue a file for the next multimodal message",
            "attachment.add",
            "PATH",
        ),
        CommandCapability(
            "attachments",
            "Show files queued for the next message",
            "attachment.list",
        ),
        CommandCapability(
            "detach",
            "Remove a queued file by number",
            "attachment.remove",
            "N",
        ),
        CommandCapability(
            "clear-attachments",
            "Clear the queued attachment list",
            "attachment.clear",
        ),
        CommandCapability("space", "Open trace-space inspection", "tab.open", payload="space-tab"),
        CommandCapability("state", "Open organism state", "tab.open", payload="state-tab"),
        CommandCapability(
            "relation",
            "Open relationship state",
            "tab.open",
            payload="relation-tab",
        ),
        CommandCapability("memory", "Open persistent memories", "tab.open", payload="memory-tab"),
        CommandCapability("identity", "Open evidence-backed identity", "tab.open", payload="identity-tab"),
        CommandCapability(
            "activity",
            "Open offline and learning activity",
            "tab.open",
            payload="activity-tab",
        ),
        CommandCapability("refresh", "Refresh the current dashboard", "dashboard.refresh"),
        CommandCapability("clear", "Clear visible conversation messages", "conversation.clear"),
        CommandCapability(
            "help",
            "Show all registered slash commands",
            "commands.help",
            aliases=("commands",),
        ),
        CommandCapability("quit", "Close the terminal interface", "app.quit", aliases=("exit",)),
    ):
        registry.register(capability)
    return registry


__all__ = [
    "CommandCapability",
    "CommandRegistry",
    "ParsedCommand",
    "build_default_command_registry",
]
