"""Capability registry isolated from interface and conversation layers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ssa.tools.models import ToolAutonomyContext, ToolOutcome

ToolHandler = Callable[[dict[str, Any], ToolAutonomyContext], Awaitable[ToolOutcome]]


@dataclass(frozen=True)
class ToolCapability:
    name: str
    description: str
    arguments_model: type[BaseModel]
    handler: ToolHandler

    def to_llm_definition(self) -> dict[str, Any]:
        parameters = self.arguments_model.model_json_schema()
        parameters.setdefault("additionalProperties", False)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


class ToolRegistry:
    """Mutable only during kernel construction, stable during model calls."""

    def __init__(self) -> None:
        self._capabilities: dict[str, ToolCapability] = {}

    def register(self, capability: ToolCapability) -> None:
        if capability.name in self._capabilities:
            raise ValueError(f"tool capability {capability.name!r} is already registered")
        self._capabilities[capability.name] = capability

    def get(self, name: str) -> ToolCapability | None:
        return self._capabilities.get(name)

    def definitions(self) -> list[dict[str, Any]]:
        return [self._capabilities[name].to_llm_definition() for name in sorted(self._capabilities)]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._capabilities))


__all__ = ["ToolCapability", "ToolHandler", "ToolRegistry"]
