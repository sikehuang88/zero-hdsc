"""Unit coverage for DeepSeek-to-GPT tool delegation."""

from __future__ import annotations

import json

import pytest

from ssa.adapters.llm import FunctionCall, ToolCall
from ssa.adapters.multimodal import MultimodalAnalysis, MultimodalError, MultimodalFileInfo
from ssa.config import MultimodalConfig
from ssa.tools.gpt_bridge import GPTBridgeExecutor
from ssa.tools.kernel import ToolKernel
from ssa.tools.models import ToolAutonomyContext
from ssa.tools.registry import ToolRegistry


class _FakeResponder:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []

    async def respond(
        self,
        prompt: str,
        *,
        instructions: str = "",
        paths: tuple[str, ...] = (),
    ) -> MultimodalAnalysis:
        self.calls.append((prompt, instructions, paths))
        if self.fail:
            raise MultimodalError("delegation unavailable")
        return MultimodalAnalysis(
            text="Independent GPT result",
            model="gpt-5.6-luna",
            response_id="resp-bridge-1",
            input_tokens=40,
            output_tokens=10,
            files=[
                MultimodalFileInfo(
                    path=paths[0],
                    name="figure.png",
                    mime_type="image/png",
                    size_bytes=256,
                    sha256="b" * 64,
                    input_type="input_image",
                )
            ]
            if paths
            else [],
        )


def _context() -> ToolAutonomyContext:
    return ToolAutonomyContext(
        correlation_id="bridge-correlation",
        conversation_id="bridge-conversation",
        energy=0.7,
        valence=0.1,
        arousal=0.3,
        trust=0.8,
        tension=0.1,
        situation_mode="respond",
        situation_confidence=0.9,
    )


def _kernel(responder: _FakeResponder) -> ToolKernel:
    registry = ToolRegistry()
    GPTBridgeExecutor(
        MultimodalConfig(),
        "",
        responder=responder,
    ).register_into(registry)
    return ToolKernel(registry)


@pytest.mark.asyncio
async def test_ask_gpt_returns_bounded_model_inference_metadata() -> None:
    responder = _FakeResponder()
    kernel = _kernel(responder)
    path = "E:\\research\\figure.png"
    call = ToolCall(
        id="call-gpt-1",
        function=FunctionCall(
            name="ask_gpt",
            arguments=json.dumps(
                {
                    "prompt": "Check this derivation",
                    "instructions": "Be skeptical",
                    "paths": [path],
                }
            ),
        ),
    )

    result = await kernel.execute(call, _context())

    assert result.ok is True
    assert result.output == "Independent GPT result"
    assert result.metadata["model_inference"] is True
    assert result.metadata["model"] == "gpt-5.6-luna"
    assert result.metadata["response_id"] == "resp-bridge-1"
    assert result.metadata["files"][0]["sha256"] == "b" * 64
    assert responder.calls == [("Check this derivation", "Be skeptical", (path,))]


@pytest.mark.asyncio
async def test_ask_gpt_provider_failure_isolated_as_tool_failure() -> None:
    result = await _kernel(_FakeResponder(fail=True)).execute(
        ToolCall(
            id="call-gpt-fail",
            function=FunctionCall(
                name="ask_gpt",
                arguments=json.dumps({"prompt": "Check independently"}),
            ),
        ),
        _context(),
    )

    assert result.ok is False
    assert result.error == "delegation unavailable"
