"""Bounded GPT delegation capability for the DeepSeek tool loop."""

from __future__ import annotations

from typing import Any

from ssa.adapters.multimodal import (
    GPTResponder,
    MagicAIMultimodalAdapter,
    MultimodalError,
)
from ssa.config import MultimodalConfig
from ssa.tools.models import AskGPTArguments, ToolAutonomyContext, ToolOutcome
from ssa.tools.registry import ToolCapability, ToolRegistry


class GPTBridgeExecutor:
    """Delegate a focused subproblem to GPT and return evidence to DeepSeek."""

    def __init__(
        self,
        config: MultimodalConfig,
        api_key: str,
        *,
        responder: GPTResponder | None = None,
    ) -> None:
        if not api_key.strip() and responder is None:
            raise ValueError("GPT bridge requires a multimodal API key")
        self._responder = responder or MagicAIMultimodalAdapter(config, api_key)

    def register_into(self, registry: ToolRegistry) -> None:
        registry.register(
            ToolCapability(
                name="ask_gpt",
                description=(
                    "Delegate a focused question, independent second opinion, complex analysis, "
                    "or image/PDF/file interpretation to GPT. Optionally provide up to eight "
                    "absolute host file paths. The GPT result is model inference for this turn; "
                    "evaluate it before composing the final answer."
                ),
                arguments_model=AskGPTArguments,
                handler=self.ask_gpt,
            )
        )

    async def ask_gpt(
        self,
        raw_arguments: dict[str, Any],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = AskGPTArguments.model_validate(raw_arguments)
        try:
            response = await self._responder.respond(
                arguments.prompt,
                instructions=arguments.instructions,
                paths=tuple(arguments.paths),
            )
        except MultimodalError as exc:
            return ToolOutcome(ok=False, error=str(exc))
        return ToolOutcome(
            ok=True,
            output=response.text,
            metadata={
                "provider": "magicai",
                "model": response.model,
                "response_id": response.response_id,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "model_inference": True,
                "files": [
                    {
                        "name": item.name,
                        "mime_type": item.mime_type,
                        "sha256": item.sha256,
                        "input_type": item.input_type,
                        "truncated": item.truncated,
                    }
                    for item in response.files
                ],
            },
        )


__all__ = ["GPTBridgeExecutor"]
