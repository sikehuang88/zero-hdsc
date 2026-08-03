"""Bounded browser sub-agent backed by Zero's local Electron browser bridge."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from ssa.tools.models import (
    ToolAutonomyContext,
    ToolOutcome,
    WebCloseArguments,
    WebOpenArguments,
    WebReadArguments,
    WebSearchArguments,
    WebTabsArguments,
)
from ssa.tools.registry import ToolCapability, ToolRegistry

_DEFAULT_BRIDGE_URL = "http://127.0.0.1:18788"
_DEFAULT_BRIDGE_TOKEN = "zero-local-browser"
_MAX_BRIDGE_RESPONSE_BYTES = 4_194_304


class BrowserBridgeError(RuntimeError):
    pass


class BrowserBridge(Protocol):
    async def command(self, action: str, payload: dict[str, Any]) -> dict[str, Any]: ...


class LocalBrowserBridge:
    """JSON transport to the Electron-owned persistent Chromium session."""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        *,
        timeout_seconds: int = 45,
    ) -> None:
        self._base_url = (
            base_url or os.getenv("ZERO_BROWSER_BRIDGE_URL") or _DEFAULT_BRIDGE_URL
        ).rstrip("/")
        self._token = token or os.getenv("ZERO_BROWSER_BRIDGE_TOKEN") or _DEFAULT_BRIDGE_TOKEN
        self._timeout_seconds = timeout_seconds

    async def command(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._command_sync, action, payload)

    def _command_sync(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self._base_url}/browser/{action}",
            data=body,
            method="POST",
            headers={
                "content-type": "application/json; charset=utf-8",
                "x-zero-browser-token": self._token,
            },
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                raw = response.read(_MAX_BRIDGE_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            detail = exc.read(8_192).decode("utf-8", errors="replace").strip()
            raise BrowserBridgeError(
                f"embedded browser bridge returned HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise BrowserBridgeError(f"embedded browser bridge is offline: {exc}") from exc
        if len(raw) > _MAX_BRIDGE_RESPONSE_BYTES:
            raise BrowserBridgeError("embedded browser bridge response exceeded 4 MiB")
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BrowserBridgeError("embedded browser bridge returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise BrowserBridgeError("embedded browser bridge response must be a JSON object")
        return decoded


class BrowserSubAgent:
    """Intent-triggered browser worker exposed through bounded model tools."""

    def __init__(self, bridge: BrowserBridge | None = None) -> None:
        self._bridge = bridge or LocalBrowserBridge()

    def register_into(self, registry: ToolRegistry) -> None:
        registry.register(
            ToolCapability(
                name="web_search",
                description=(
                    "Start Zero's embedded browser on demand and search the live web. Returns "
                    "structured titles, URLs, snippets, and the visible search page state. Use "
                    "this whenever the owner explicitly asks to search or look something up online."
                ),
                arguments_model=WebSearchArguments,
                handler=self.search,
            )
        )
        registry.register(
            ToolCapability(
                name="web_open",
                description=(
                    "Open an HTTP or HTTPS URL in Zero's persistent embedded browser. Existing "
                    "site cookies and login state remain inside the browser profile."
                ),
                arguments_model=WebOpenArguments,
                handler=self.open,
            )
        )
        registry.register(
            ToolCapability(
                name="web_read",
                description=(
                    "Read the current embedded-browser page as sanitized title, URL, visible text, "
                    "and links. Password fields, cookies, and session tokens are excluded."
                ),
                arguments_model=WebReadArguments,
                handler=self.read,
            )
        )
        registry.register(
            ToolCapability(
                name="web_tabs",
                description="List the tabs currently owned by Zero's embedded browser worker.",
                arguments_model=WebTabsArguments,
                handler=self.tabs,
            )
        )
        registry.register(
            ToolCapability(
                name="web_close",
                description=(
                    "Close or hide an embedded browser tab after the browsing task is complete. "
                    "Persistent cookies and site storage remain in the browser profile."
                ),
                arguments_model=WebCloseArguments,
                handler=self.close,
            )
        )

    async def search(
        self,
        raw_arguments: dict[str, Any],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = WebSearchArguments.model_validate(raw_arguments)
        return await self._execute(
            "search",
            {"query": arguments.query, "max_results": arguments.max_results},
            external_truth=True,
        )

    async def open(
        self,
        raw_arguments: dict[str, Any],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = WebOpenArguments.model_validate(raw_arguments)
        parsed = urlsplit(arguments.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ToolOutcome(ok=False, error="web_open requires an absolute HTTP(S) URL")
        return await self._execute("open", {"url": arguments.url}, external_truth=True)

    async def read(
        self,
        raw_arguments: dict[str, Any],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = WebReadArguments.model_validate(raw_arguments)
        return await self._execute(
            "read",
            arguments.model_dump(exclude_none=True),
            external_truth=True,
        )

    async def tabs(
        self,
        raw_arguments: dict[str, Any],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        WebTabsArguments.model_validate(raw_arguments)
        return await self._execute("tabs", {}, external_truth=False)

    async def close(
        self,
        raw_arguments: dict[str, Any],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = WebCloseArguments.model_validate(raw_arguments)
        return await self._execute(
            "close",
            arguments.model_dump(exclude_none=True),
            external_truth=False,
        )

    async def _execute(
        self,
        action: str,
        payload: dict[str, Any],
        *,
        external_truth: bool,
    ) -> ToolOutcome:
        try:
            result = await self._bridge.command(action, payload)
        except BrowserBridgeError as exc:
            return ToolOutcome(ok=False, error=str(exc), metadata={"agent": "browser_subagent"})
        if result.get("ok") is False:
            error = str(result.get("error") or "embedded browser command failed")
            return ToolOutcome(ok=False, error=error, metadata={"agent": "browser_subagent"})
        metadata: dict[str, Any] = {
            "agent": "browser_subagent",
            "provider": "zero-embedded-chromium",
            "host_observation": True,
            "observation_kind": f"web_{action}",
            "observed_at_ms": int(time.time() * 1_000),
        }
        if external_truth:
            metadata.update(
                {
                    "external_truth": True,
                    "truth_status": "observed",
                    "retrieved_at_ms": metadata["observed_at_ms"],
                    "cached": False,
                }
            )
        return ToolOutcome(
            ok=True,
            output=json.dumps(result, ensure_ascii=False),
            metadata=metadata,
        )


__all__ = [
    "BrowserBridge",
    "BrowserBridgeError",
    "BrowserSubAgent",
    "LocalBrowserBridge",
]
