"""Unit coverage for Zero's Firecrawl web research tools."""

from __future__ import annotations

import json
from typing import Any

import pytest

from ssa.adapters.llm import FunctionCall, ToolCall
from ssa.config import FirecrawlConfig
from ssa.tools.firecrawl import FirecrawlExecutor, FirecrawlResponse
from ssa.tools.kernel import ToolKernel
from ssa.tools.models import ToolAutonomyContext
from ssa.tools.registry import ToolRegistry


class _FakeTransport:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def post_json(
        self,
        url: str,
        *,
        body: dict[str, Any],
        headers: dict[str, str],
        timeout_seconds: int,
        max_response_bytes: int,
    ) -> FirecrawlResponse:
        self.calls.append(
            {
                "url": url,
                "body": body,
                "headers": headers,
                "timeout_seconds": timeout_seconds,
                "max_response_bytes": max_response_bytes,
            }
        )
        return FirecrawlResponse(
            data=self.responses.pop(0),
            final_url=url,
            status_code=200,
        )


def _context() -> ToolAutonomyContext:
    return ToolAutonomyContext(
        correlation_id="firecrawl-correlation",
        conversation_id="firecrawl-conversation",
        energy=0.7,
        valence=0.2,
        arousal=0.4,
        trust=0.8,
        tension=0.1,
        situation_mode="explore",
        situation_confidence=0.9,
    )


def _kernel(
    transport: _FakeTransport,
    *,
    config: FirecrawlConfig | None = None,
    api_key: str = "fc-test-only",
) -> ToolKernel:
    registry = ToolRegistry()
    FirecrawlExecutor(
        config or FirecrawlConfig(),
        api_key,
        transport=transport,
    ).register_into(registry)
    return ToolKernel(registry, max_output_chars=64_000)


def _call(name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(
        id=f"call-{name}",
        function=FunctionCall(name=name, arguments=json.dumps(arguments)),
    )


@pytest.mark.asyncio
async def test_firecrawl_search_builds_v2_request_and_preserves_sources() -> None:
    transport = _FakeTransport(
        [
            {
                "success": True,
                "id": "search-123",
                "creditsUsed": 2,
                "data": {
                    "web": [
                        {
                            "title": "Official documentation",
                            "url": "https://example.test/docs",
                            "description": "Primary source",
                            "highlights": ["Zero live web research"],
                            "markdown": "# Docs\n" + "x" * 1_000,
                            "ignoredSecret": "never-forward-this",
                        }
                    ],
                    "news": [
                        {
                            "title": "Release news",
                            "url": "https://example.test/news",
                            "publishedDate": "2026-08-02",
                        }
                    ],
                },
            }
        ]
    )
    kernel = _kernel(
        transport,
        config=FirecrawlConfig(max_result_content_chars=500),
    )

    result = await kernel.execute(
        _call(
            "firecrawl_search",
            {
                "query": "Zero web research",
                "limit": 6,
                "sources": ["web", "news"],
                "categories": ["github"],
                "recency": "w",
                "location": "Shenzhen, China",
                "country": "cn",
                "scrape_results": True,
            },
        ),
        _context(),
    )

    assert result.ok is True
    assert transport.calls == [
        {
            "url": "https://api.firecrawl.dev/v2/search",
            "body": {
                "query": "Zero web research",
                "limit": 6,
                "highlights": True,
                "sources": [{"type": "web"}, {"type": "news"}],
                "country": "CN",
                "categories": [{"type": "github"}],
                "tbs": "qdr:w",
                "location": "Shenzhen, China",
                "scrapeOptions": {
                    "formats": [{"type": "markdown"}],
                    "onlyMainContent": True,
                },
            },
            "headers": {
                "authorization": "Bearer fc-test-only",
                "content-type": "application/json; charset=utf-8",
                "user-agent": "Zero/0.5 firecrawl",
            },
            "timeout_seconds": 60,
            "max_response_bytes": 4_194_304,
        }
    ]
    payload = json.loads(result.output)
    assert payload["search_id"] == "search-123"
    assert payload["results"]["web"][0]["url"] == "https://example.test/docs"
    assert payload["results"]["web"][0]["highlights"] == ["Zero live web research"]
    assert payload["results"]["web"][0]["markdown"].endswith("...[truncated]")
    assert "ignoredSecret" not in result.output
    assert "fc-test-only" not in result.output
    assert result.metadata["provider"] == "firecrawl"
    assert result.metadata["record_count"] == 2


@pytest.mark.asyncio
async def test_firecrawl_scrape_supports_focused_question_and_bounds_content() -> None:
    transport = _FakeTransport(
        [
            {
                "success": True,
                "data": {
                    "markdown": "m" * 2_000,
                    "answer": "The release date is August 2, 2026.",
                    "metadata": {"title": "Release notes", "scrapeId": "scrape-7"},
                },
            }
        ]
    )

    result = await _kernel(transport).execute(
        _call(
            "firecrawl_scrape",
            {
                "url": "https://example.test/releases",
                "question": "What is the release date?",
                "wait_for_ms": 500,
                "max_age_ms": 60_000,
                "max_chars": 1_000,
            },
        ),
        _context(),
    )

    assert result.ok is True
    assert transport.calls[0]["body"] == {
        "url": "https://example.test/releases",
        "formats": [
            "markdown",
            {"type": "query", "prompt": "What is the release date?"},
        ],
        "onlyMainContent": True,
        "waitFor": 500,
        "maxAge": 60_000,
    }
    payload = json.loads(result.output)
    assert payload["answer"] == "The release date is August 2, 2026."
    assert payload["markdown"].endswith("...[truncated]")
    assert result.metadata["source_url"] == "https://example.test/releases"


@pytest.mark.asyncio
async def test_firecrawl_missing_key_fails_without_contacting_transport() -> None:
    transport = _FakeTransport([])

    result = await _kernel(transport, api_key="").execute(
        _call("firecrawl_search", {"query": "latest Zero release"}),
        _context(),
    )

    assert result.ok is False
    assert "HDSC_FIRECRAWL_API_KEY" in (result.error or "")
    assert transport.calls == []
