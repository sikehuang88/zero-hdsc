"""Contract coverage for the isolated, keyless external-truth registry."""

from __future__ import annotations

import json
from typing import Any

import pytest

from ssa.adapters.llm import FunctionCall, ToolCall
from ssa.config import ExternalTruthConfig
from ssa.tools.external_truth import ExternalTruthExecutor, JsonResponse
from ssa.tools.kernel import ToolKernel
from ssa.tools.models import ToolAutonomyContext
from ssa.tools.registry import ToolRegistry


class _FakeTransport:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [{}])
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []
        self.final_url: str | None = None

    async def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout_seconds: int,
        max_response_bytes: int,
    ) -> JsonResponse:
        assert timeout_seconds == 3
        assert max_response_bytes == 64_000
        self.urls.append(url)
        self.headers.append(headers)
        data = self.responses[min(len(self.urls) - 1, len(self.responses) - 1)]
        return JsonResponse(data=data, final_url=self.final_url or url, status_code=200)


def _context() -> ToolAutonomyContext:
    return ToolAutonomyContext(
        correlation_id="truth-correlation",
        conversation_id="truth-conversation",
        energy=0.7,
        valence=0.1,
        arousal=0.3,
        trust=0.7,
        tension=0.1,
        situation_mode="respond",
        situation_confidence=0.8,
    )


def _call(name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(
        id=f"call-{name}",
        function=FunctionCall(name=name, arguments=json.dumps(arguments)),
    )


def _kernel(transport: _FakeTransport) -> ToolKernel:
    registry = ToolRegistry()
    ExternalTruthExecutor(
        ExternalTruthConfig(
            request_timeout_seconds=3,
            max_response_bytes=64_000,
            cache_ttl_seconds=300,
            min_request_interval_ms=0,
        ),
        transport=transport,
    ).register_into(registry)
    return ToolKernel(registry)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments", "host", "provider"),
    [
        (
            "truth_weather",
            {"latitude": 39.9, "longitude": 116.4},
            "api.open-meteo.com",
            "open-meteo",
        ),
        (
            "truth_sun_times",
            {"latitude": 39.9, "longitude": 116.4, "date": "2030-03-05"},
            "api.sunrise-sunset.org",
            "sunrise-sunset",
        ),
        (
            "truth_holidays",
            {"country_code": "cn", "year": 2030},
            "date.nager.at",
            "nager-date",
        ),
        (
            "truth_geocode",
            {"operation": "search", "query": "Beijing"},
            "nominatim.openstreetmap.org",
            "nominatim",
        ),
        (
            "truth_news",
            {"query": "AI", "limit": 2},
            "noozra.com",
            "noozra",
        ),
        (
            "truth_radio",
            {"query": "ambient", "limit": 2},
            "de1.api.radio-browser.info",
            "radio-browser",
        ),
        (
            "truth_music",
            {"entity": "artist", "query": "Coldplay"},
            "musicbrainz.org",
            "musicbrainz",
        ),
        (
            "truth_translate",
            {"text": "hello", "source_language": "en", "target_language": "zh-CN"},
            "api.mymemory.translated.net",
            "mymemory",
        ),
        (
            "truth_service_status",
            {"provider": "github"},
            "www.githubstatus.com",
            "github-status",
        ),
    ],
)
async def test_each_truth_tool_returns_provenance_envelope(
    name: str,
    arguments: dict[str, object],
    host: str,
    provider: str,
) -> None:
    transport = _FakeTransport([{"articles": [], "components": [], "incidents": []}])
    result = await _kernel(transport).execute(_call(name, arguments), _context())

    assert result.ok is True
    envelope = json.loads(result.output)
    assert envelope["contract"] == "external-observation-v1"
    assert envelope["truth_status"] == "provider_reported"
    assert envelope["provider"] == provider
    assert envelope["cached"] is False
    assert transport.urls[0].startswith(f"https://{host}/")
    assert transport.headers[0]["Accept"] == "application/json"
    assert "HDSC" in transport.headers[0]["User-Agent"]
    assert result.metadata["source_kind"] == "world_observed"
    assert result.metadata["external_truth"] is True


@pytest.mark.asyncio
async def test_identical_truth_request_uses_cache() -> None:
    transport = _FakeTransport([{"current": {"temperature_2m": 20.0}}])
    kernel = _kernel(transport)
    call = _call("truth_weather", {"latitude": 1.0, "longitude": 2.0})

    first = await kernel.execute(call, _context())
    second = await kernel.execute(call, _context())

    assert first.ok is True
    assert second.ok is True
    assert len(transport.urls) == 1
    assert json.loads(first.output)["cached"] is False
    assert json.loads(second.output)["cached"] is True


@pytest.mark.asyncio
async def test_invalid_arguments_fail_before_network_access() -> None:
    transport = _FakeTransport()
    result = await _kernel(transport).execute(
        _call("truth_geocode", {"operation": "reverse", "latitude": 20.0}),
        _context(),
    )

    assert result.ok is False
    assert "longitude" in (result.error or "")
    assert transport.urls == []


@pytest.mark.asyncio
async def test_redirect_outside_allowlist_is_isolated() -> None:
    transport = _FakeTransport()
    transport.final_url = "https://example.invalid/captured"

    result = await _kernel(transport).execute(
        _call("truth_service_status", {"provider": "github"}),
        _context(),
    )

    assert result.ok is False
    assert "allowlist" in (result.error or "")
