"""Unit coverage for frontend-routed image generation."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from ssa.adapters.llm import FunctionCall, ToolCall
from ssa.tools.image_generation import GeneratedImageStore, ImageGenerationExecutor
from ssa.tools.kernel import ToolKernel
from ssa.tools.models import ImageGenerationRoute, ToolAutonomyContext
from ssa.tools.registry import ToolRegistry

_PNG = b"\x89PNG\r\n\x1a\n" + b"zero-image-fixture"


class _FakeTransport:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.posts: list[tuple[str, dict[str, Any], dict[str, str], int]] = []
        self.downloads: list[str] = []

    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        *,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        self.posts.append((url, payload, headers, timeout_seconds))
        return self.response

    async def get_bytes(
        self,
        url: str,
        *,
        timeout_seconds: int,
        max_bytes: int,
    ) -> tuple[bytes, str]:
        del timeout_seconds, max_bytes
        self.downloads.append(url)
        return _PNG, "image/png"


def _context(protocol: str, *, base_url: str = "https://images.example/v1") -> ToolAutonomyContext:
    return ToolAutonomyContext(
        correlation_id="image-correlation",
        conversation_id="image-conversation",
        energy=0.7,
        valence=0.2,
        arousal=0.4,
        trust=0.8,
        tension=0.1,
        situation_mode="respond",
        situation_confidence=0.9,
        image_generation_route=ImageGenerationRoute(
            protocol=protocol,
            base_url=base_url,
            api_key="image-key",
            model="gpt-image-2",
        ),
    )


def _kernel(tmp_path: Path, transport: _FakeTransport) -> ToolKernel:
    registry = ToolRegistry()
    ImageGenerationExecutor(
        transport=transport,
        store=GeneratedImageStore(tmp_path / "images"),
    ).register_into(registry)
    return ToolKernel(registry)


def _call() -> ToolCall:
    return ToolCall(
        id="call-image-1",
        function=FunctionCall(
            name="generate_image",
            arguments=json.dumps(
                {
                    "prompt": "  cinematic moonlit garden  ",
                    "size": "1536x1024",
                    "quality": "high",
                }
            ),
        ),
    )


@pytest.mark.parametrize(
    ("protocol", "response", "endpoint", "auth_header"),
    [
        (
            "openai-responses",
            {
                "output": [
                    {
                        "type": "image_generation_call",
                        "result": base64.b64encode(_PNG).decode("ascii"),
                    }
                ]
            },
            "https://images.example/v1/responses",
            "Authorization",
        ),
        (
            "openai-chat",
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/png;base64,"
                                        + base64.b64encode(_PNG).decode("ascii")
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            "https://images.example/v1/chat/completions",
            "Authorization",
        ),
        (
            "anthropic-messages",
            {
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(_PNG).decode("ascii"),
                        },
                    }
                ]
            },
            "https://images.example/v1/messages",
            "x-api-key",
        ),
    ],
)
@pytest.mark.asyncio
async def test_generate_image_supports_frontend_protocols(
    tmp_path: Path,
    protocol: str,
    response: dict[str, Any],
    endpoint: str,
    auth_header: str,
) -> None:
    transport = _FakeTransport(response)
    result = await _kernel(tmp_path, transport).execute(_call(), _context(protocol))

    assert result.ok is True
    assert result.tool_name == "generate_image"
    assert result.metadata["image_generation"] is True
    assert result.metadata["asset_url"].startswith("/api/generated-images/")
    assert result.metadata["mime_type"] == "image/png"
    assert result.metadata["prompt"] == "cinematic moonlit garden"
    assert len(list((tmp_path / "images").glob("*.png"))) == 1
    url, payload, headers, timeout = transport.posts[0]
    assert url == endpoint
    assert payload["model"] == "gpt-image-2"
    assert auth_header in headers
    assert "image-key" in headers[auth_header]
    assert timeout == 180


@pytest.mark.asyncio
async def test_generate_image_downloads_remote_provider_asset(tmp_path: Path) -> None:
    transport = _FakeTransport(
        {"data": [{"type": "image", "url": "https://cdn.example/render.png"}]}
    )

    result = await _kernel(tmp_path, transport).execute(_call(), _context("openai-chat"))

    assert result.ok is True
    assert transport.downloads == ["https://cdn.example/render.png"]


@pytest.mark.asyncio
async def test_generate_image_reports_missing_frontend_route(tmp_path: Path) -> None:
    transport = _FakeTransport({})
    context = _context("openai-responses").model_copy(
        update={"image_generation_route": None}
    )

    result = await _kernel(tmp_path, transport).execute(_call(), context)

    assert result.ok is False
    assert result.error == "image generation provider is not configured"
    assert transport.posts == []
