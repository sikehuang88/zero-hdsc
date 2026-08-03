"""Dispatch and error isolation for registered model tools."""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import ValidationError

from ssa.adapters.llm import ToolCall
from ssa.tools.models import ToolAutonomyContext, ToolExecutionResult
from ssa.tools.registry import ToolRegistry


class ToolKernel:
    def __init__(self, registry: ToolRegistry, *, max_output_chars: int = 32_000) -> None:
        if max_output_chars < 1:
            raise ValueError("max_output_chars must be positive")
        self._registry = registry
        self._max_output_chars = max_output_chars

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    def definitions(self) -> list[dict[str, Any]]:
        return self._registry.definitions()

    async def execute(
        self,
        call: ToolCall,
        context: ToolAutonomyContext,
    ) -> ToolExecutionResult:
        started = time.monotonic()
        capability = self._registry.get(call.function.name)
        if capability is None:
            return self._failure(call, started, f"unknown tool {call.function.name!r}")

        try:
            decoded = json.loads(call.function.arguments)
            if not isinstance(decoded, dict):
                raise ValueError("tool arguments must be a JSON object")
            validated = capability.arguments_model.model_validate(decoded)
            outcome = await capability.handler(validated.model_dump(exclude_unset=True), context)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            return self._failure(call, started, str(exc))
        except Exception as exc:
            return self._failure(call, started, f"tool execution failed: {exc}")

        raw_output = outcome.output
        truncated = len(raw_output) > self._max_output_chars
        output = raw_output
        if truncated:
            output = raw_output[: self._max_output_chars] + "\n...[truncated]"
        return ToolExecutionResult(
            **outcome.model_dump(exclude={"output"}),
            output=output,
            call_id=call.id,
            tool_name=call.function.name,
            duration_ms=int((time.monotonic() - started) * 1000),
            output_chars=len(raw_output),
            truncated=truncated,
        )

    @staticmethod
    def _failure(call: ToolCall, started: float, error: str) -> ToolExecutionResult:
        return ToolExecutionResult(
            call_id=call.id,
            tool_name=call.function.name,
            ok=False,
            error=error,
            duration_ms=int((time.monotonic() - started) * 1000),
            output_chars=0,
            truncated=False,
        )


__all__ = ["ToolKernel"]
