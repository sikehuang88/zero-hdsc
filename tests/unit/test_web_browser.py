"""Unit coverage for Zero's intent-triggered browser sub-agent."""

from __future__ import annotations

import json
from typing import Any

import pytest

from ssa.adapters.llm import FunctionCall, ToolCall
from ssa.tools.kernel import ToolKernel
from ssa.tools.models import ToolAutonomyContext
from ssa.tools.registry import ToolRegistry
from ssa.tools.web_browser import BrowserBridgeError, BrowserSubAgent


class _FakeBridge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def command(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((action, payload))
        if action == "search":
            return {
                "ok": True,
                "agent": "browser_subagent",
                "tab_id": "main",
                "query": payload["query"],
                "results": [
                    {
                        "title": "WebView2 data folders",
                        "url": "https://example.test/webview2",
                        "snippet": "Persistent profile documentation",
                    }
                ],
            }
        return {"ok": True, "tab_id": "main", "action": action, **payload}


class _OfflineBridge:
    async def command(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        del action, payload
        raise BrowserBridgeError("embedded browser bridge is offline")


def _context() -> ToolAutonomyContext:
    return ToolAutonomyContext(
        correlation_id="browser-correlation",
        conversation_id="browser-conversation",
        energy=0.7,
        valence=0.2,
        arousal=0.4,
        trust=0.8,
        tension=0.1,
        situation_mode="explore",
        situation_confidence=0.9,
    )


def _kernel(bridge: _FakeBridge | _OfflineBridge) -> ToolKernel:
    registry = ToolRegistry()
    BrowserSubAgent(bridge).register_into(registry)
    return ToolKernel(registry)


def _call(name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(
        id=f"call-{name}",
        function=FunctionCall(name=name, arguments=json.dumps(arguments)),
    )


@pytest.mark.asyncio
async def test_web_search_runs_bounded_browser_subagent() -> None:
    bridge = _FakeBridge()
    result = await _kernel(bridge).execute(
        _call("web_search", {"query": "WebView2 persistent cookies", "max_results": 4}),
        _context(),
    )

    assert result.ok is True
    assert bridge.calls == [
        ("search", {"query": "WebView2 persistent cookies", "max_results": 4})
    ]
    payload = json.loads(result.output)
    assert payload["agent"] == "browser_subagent"
    assert payload["results"][0]["url"] == "https://example.test/webview2"
    assert result.metadata["external_truth"] is True
    assert result.metadata["provider"] == "zero-embedded-chromium"


@pytest.mark.asyncio
async def test_web_open_accepts_only_absolute_http_urls() -> None:
    bridge = _FakeBridge()
    kernel = _kernel(bridge)

    rejected = await kernel.execute(
        _call("web_open", {"url": "javascript:alert(1)"}),
        _context(),
    )
    opened = await kernel.execute(
        _call("web_open", {"url": "https://example.test/article"}),
        _context(),
    )

    assert rejected.ok is False
    assert "absolute HTTP(S) URL" in (rejected.error or "")
    assert opened.ok is True
    assert bridge.calls == [("open", {"url": "https://example.test/article"})]


@pytest.mark.asyncio
async def test_browser_bridge_offline_error_is_isolated() -> None:
    result = await _kernel(_OfflineBridge()).execute(
        _call("web_tabs", {}),
        _context(),
    )

    assert result.ok is False
    assert result.error == "embedded browser bridge is offline"
    assert result.metadata == {"agent": "browser_subagent"}
