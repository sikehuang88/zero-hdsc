"""Firecrawl-backed live web search and page extraction for Zero."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from ssa.config import FirecrawlConfig
from ssa.tools.models import (
    FirecrawlScrapeArguments,
    FirecrawlSearchArguments,
    ToolAutonomyContext,
    ToolOutcome,
)
from ssa.tools.registry import ToolCapability, ToolRegistry


@dataclass(frozen=True)
class FirecrawlResponse:
    data: dict[str, Any]
    final_url: str
    status_code: int


class FirecrawlTransport(Protocol):
    async def post_json(
        self,
        url: str,
        *,
        body: dict[str, Any],
        headers: dict[str, str],
        timeout_seconds: int,
        max_response_bytes: int,
    ) -> FirecrawlResponse: ...


class UrllibFirecrawlTransport:
    async def post_json(
        self,
        url: str,
        *,
        body: dict[str, Any],
        headers: dict[str, str],
        timeout_seconds: int,
        max_response_bytes: int,
    ) -> FirecrawlResponse:
        return await asyncio.to_thread(
            self._post_json_sync,
            url,
            body,
            headers,
            timeout_seconds,
            max_response_bytes,
        )

    @staticmethod
    def _post_json_sync(
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
        timeout_seconds: int,
        max_response_bytes: int,
    ) -> FirecrawlResponse:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read(max_response_bytes + 1)
                if len(raw) > max_response_bytes:
                    raise ValueError("Firecrawl response exceeded configured byte limit")
                decoded = json.loads(raw.decode("utf-8", errors="strict"))
                if not isinstance(decoded, dict):
                    raise ValueError("Firecrawl response must be a JSON object")
                return FirecrawlResponse(
                    data=decoded,
                    final_url=response.geturl(),
                    status_code=int(response.status),
                )
        except urllib.error.HTTPError as exc:
            detail = exc.read(8_192).decode("utf-8", errors="replace").strip()
            try:
                parsed = json.loads(detail)
                if isinstance(parsed, dict):
                    detail = str(parsed.get("error") or parsed.get("message") or detail)
            except json.JSONDecodeError:
                pass
            suffix = f": {detail}" if detail else ""
            raise ValueError(f"Firecrawl returned HTTP {exc.code}{suffix}") from exc
        except urllib.error.URLError as exc:
            raise ValueError(f"Firecrawl request failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError("Firecrawl returned invalid JSON") from exc


class FirecrawlExecutor:
    """Authenticated Firecrawl v2 client exposed as bounded model tools."""

    def __init__(
        self,
        config: FirecrawlConfig,
        api_key: str,
        *,
        transport: FirecrawlTransport | None = None,
    ) -> None:
        self._config = config
        self._api_key = api_key.strip()
        self._transport = transport or UrllibFirecrawlTransport()

    def register_into(self, registry: ToolRegistry) -> None:
        registry.register(
            ToolCapability(
                name="firecrawl_search",
                description=(
                    "Search the live web through Firecrawl. Returns current web, news, or image "
                    "results with source URLs and optional extracted page content. Use this first "
                    "for discovery, recent facts, research, comparisons, and fact-checking."
                ),
                arguments_model=FirecrawlSearchArguments,
                handler=self.search,
            )
        )
        registry.register(
            ToolCapability(
                name="firecrawl_scrape",
                description=(
                    "Extract clean content from one known public HTTP(S) page through Firecrawl. "
                    "Use after search to read the strongest sources or when the owner supplies a URL."
                ),
                arguments_model=FirecrawlScrapeArguments,
                handler=self.scrape,
            )
        )

    async def search(
        self,
        raw: dict[str, Any],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = FirecrawlSearchArguments.model_validate(raw)
        body: dict[str, Any] = {
            "query": arguments.query,
            "limit": arguments.limit,
            "highlights": True,
            "sources": [{"type": source} for source in arguments.sources],
            "country": arguments.country,
        }
        if arguments.categories:
            body["categories"] = [{"type": category} for category in arguments.categories]
        if arguments.recency is not None:
            body["tbs"] = f"qdr:{arguments.recency}"
        if arguments.location is not None:
            body["location"] = arguments.location
        if arguments.scrape_results:
            body["scrapeOptions"] = {
                "formats": [{"type": "markdown"}],
                "onlyMainContent": True,
            }

        response = await self._post("search", body)
        envelope = response.data
        if envelope.get("success") is False:
            return ToolOutcome(ok=False, error=_firecrawl_error(envelope))
        data = envelope.get("data")
        if not isinstance(data, dict):
            data = {}
        normalized = {
            key: [self._normalize_search_item(item) for item in items if isinstance(item, dict)]
            for key in ("web", "news", "images")
            if isinstance((items := data.get(key)), list)
        }
        output = {
            "query": arguments.query,
            "results": normalized,
            "warning": envelope.get("warning"),
            "search_id": envelope.get("id"),
            "credits_used": envelope.get("creditsUsed"),
        }
        return ToolOutcome(
            ok=True,
            output=json.dumps(output, ensure_ascii=False, indent=2),
            metadata=self._metadata(
                response,
                observation_kind="web_search",
                record_count=sum(len(items) for items in normalized.values()),
                search_id=envelope.get("id"),
            ),
        )

    async def scrape(
        self,
        raw: dict[str, Any],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = FirecrawlScrapeArguments.model_validate(raw)
        formats: list[Any] = ["markdown"]
        if arguments.question is not None:
            formats.append({"type": "query", "prompt": arguments.question})
        body: dict[str, Any] = {
            "url": arguments.url,
            "formats": formats,
            "onlyMainContent": arguments.only_main_content,
        }
        if arguments.wait_for_ms is not None:
            body["waitFor"] = arguments.wait_for_ms
        if arguments.max_age_ms is not None:
            body["maxAge"] = arguments.max_age_ms

        response = await self._post("scrape", body)
        envelope = response.data
        if envelope.get("success") is False:
            return ToolOutcome(ok=False, error=_firecrawl_error(envelope))
        data = envelope.get("data", envelope)
        if not isinstance(data, dict):
            return ToolOutcome(ok=False, error="Firecrawl scrape returned no page object")
        normalized = dict(data)
        markdown = normalized.get("markdown")
        if isinstance(markdown, str):
            normalized["markdown"] = _truncate(markdown, arguments.max_chars)
        html = normalized.get("html")
        if isinstance(html, str):
            normalized["html"] = _truncate(html, arguments.max_chars)
        return ToolOutcome(
            ok=True,
            output=json.dumps(normalized, ensure_ascii=False, indent=2),
            metadata=self._metadata(
                response,
                observation_kind="web_page",
                record_count=1,
                source_url=arguments.url,
            ),
        )

    async def _post(self, endpoint: str, body: dict[str, Any]) -> FirecrawlResponse:
        if not self._api_key:
            raise ValueError(
                "Firecrawl is not configured; set HDSC_FIRECRAWL_API_KEY or FIRECRAWL_API_KEY"
            )
        return await self._transport.post_json(
            f"{self._config.base_url}/{endpoint}",
            body=body,
            headers={
                "authorization": f"Bearer {self._api_key}",
                "content-type": "application/json; charset=utf-8",
                "user-agent": "Zero/0.5 firecrawl",
            },
            timeout_seconds=self._config.request_timeout_seconds,
            max_response_bytes=self._config.max_response_bytes,
        )

    def _normalize_search_item(self, item: dict[str, Any]) -> dict[str, Any]:
        normalized = {
            key: value
            for key, value in item.items()
            if key
            in {
                "title",
                "url",
                "description",
                "snippet",
                "publishedDate",
                "category",
                "imageUrl",
                "width",
                "height",
                "position",
                "highlights",
            }
        }
        for key in ("markdown", "html"):
            value = item.get(key)
            if isinstance(value, str):
                normalized[key] = _truncate(value, self._config.max_result_content_chars)
        return normalized

    @staticmethod
    def _metadata(
        response: FirecrawlResponse,
        *,
        observation_kind: str,
        record_count: int,
        **extra: Any,
    ) -> dict[str, Any]:
        observed_at = datetime.now(UTC)
        return {
            "external_truth": True,
            "truth_status": "provider_reported",
            "provider": "firecrawl",
            "provider_url": response.final_url,
            "observed_at": observed_at.isoformat(),
            "observed_at_ms": int(observed_at.timestamp() * 1_000),
            "observation_kind": observation_kind,
            "record_count": record_count,
            **{key: value for key, value in extra.items() if value is not None},
        }


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[truncated]"


def _firecrawl_error(payload: dict[str, Any]) -> str:
    return str(payload.get("error") or payload.get("message") or "Firecrawl request failed")


__all__ = [
    "FirecrawlExecutor",
    "FirecrawlResponse",
    "FirecrawlTransport",
    "UrllibFirecrawlTransport",
]
