"""Contract tests for the OpenAI-compatible MagicAI Responses adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ssa.adapters.multimodal import MagicAIMultimodalAdapter, MultimodalError
from ssa.config import MultimodalConfig


class _FakeTransport:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def post_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: int,
        max_response_bytes: int,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
                "max_response_bytes": max_response_bytes,
            }
        )
        return self.response


@pytest.mark.asyncio
async def test_builds_image_pdf_and_text_responses_content(tmp_path: Path) -> None:
    image = tmp_path / "screen.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nimage")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\npaper")
    note = tmp_path / "notes.md"
    note.write_text("**important**", encoding="utf-8")
    transport = _FakeTransport(
        {
            "id": "resp-mm-1",
            "model": "gpt-5.6-luna",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Three files analyzed."}],
                }
            ],
            "usage": {"input_tokens": 120, "output_tokens": 12},
        }
    )
    adapter = MagicAIMultimodalAdapter(
        MultimodalConfig(timeout_seconds=33),
        "sk-test-only",
        transport=transport,
    )

    result = await adapter.analyze(
        (str(image), str(pdf), str(note)),
        prompt="Compare the evidence.",
    )

    assert result.text == "Three files analyzed."
    assert result.response_id == "resp-mm-1"
    assert result.input_tokens == 120
    assert [item.input_type for item in result.files] == [
        "input_image",
        "input_file",
        "input_text",
    ]
    call = transport.calls[0]
    assert call["url"] == "https://sky1818.com/v1/responses"
    assert call["timeout_seconds"] == 33
    assert call["headers"]["Authorization"] == "Bearer sk-test-only"
    payload = call["payload"]
    assert payload["model"] == "gpt-5.6-luna"
    assert payload["reasoning"] == {"effort": "medium"}
    assert payload["store"] is False
    parts = payload["input"][0]["content"]
    assert [part["type"] for part in parts] == [
        "input_text",
        "input_text",
        "input_image",
        "input_file",
        "input_text",
    ]
    assert parts[2]["image_url"].startswith("data:image/png;base64,")
    assert parts[3]["file_data"].startswith("data:application/pdf;base64,")
    assert "notes.md" in parts[4]["text"]
    persisted = result.model_dump_json()
    assert "base64" not in persisted
    assert "sk-test-only" not in persisted


@pytest.mark.asyncio
async def test_rejects_unsupported_and_oversized_files(tmp_path: Path) -> None:
    unsupported = tmp_path / "archive.bin"
    unsupported.write_bytes(b"binary")
    oversized = tmp_path / "large.png"
    oversized.write_bytes(b"x" * 1_025)
    transport = _FakeTransport({"output_text": "unused"})
    adapter = MagicAIMultimodalAdapter(
        MultimodalConfig(max_file_bytes=1_024, max_total_bytes=1_024),
        "sk-test-only",
        transport=transport,
    )

    with pytest.raises(MultimodalError, match="unsupported attachment type"):
        await adapter.analyze((str(unsupported),), prompt="read")
    with pytest.raises(MultimodalError, match="limit is 1024"):
        await adapter.analyze((str(oversized),), prompt="read")
    assert transport.calls == []


@pytest.mark.asyncio
async def test_accepts_top_level_output_text(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpeg")
    transport = _FakeTransport({"output_text": "A photo.", "model": "gpt-5.6-luna"})
    adapter = MagicAIMultimodalAdapter(
        MultimodalConfig(),
        "sk-test-only",
        transport=transport,
    )

    result = await adapter.analyze((str(image),), prompt="describe")

    assert result.text == "A photo."


@pytest.mark.asyncio
async def test_text_response_supports_gpt_tool_delegation_without_files() -> None:
    transport = _FakeTransport(
        {
            "id": "resp-text-1",
            "output_text": "Independent answer.",
            "model": "gpt-5.6-luna",
        }
    )
    adapter = MagicAIMultimodalAdapter(
        MultimodalConfig(),
        "sk-test-only",
        transport=transport,
    )

    result = await adapter.respond(
        "Review this claim.",
        instructions="Act as an independent critic.",
    )

    assert result.text == "Independent answer."
    assert result.files == []
    payload = transport.calls[0]["payload"]
    assert payload["instructions"] == "Act as an independent critic."
    assert payload["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Review this claim."}],
        }
    ]
