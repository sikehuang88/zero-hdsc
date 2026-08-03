"""Isolated, keyless external-observation capabilities for the tool kernel."""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal, Protocol
from urllib.parse import urlencode, urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator

from ssa.config import ExternalTruthConfig
from ssa.tools.models import ToolAutonomyContext, ToolOutcome
from ssa.tools.registry import ToolCapability, ToolRegistry

_ALLOWED_HOSTS = {
    "api.open-meteo.com",
    "api.sunrise-sunset.org",
    "date.nager.at",
    "nominatim.openstreetmap.org",
    "noozra.com",
    "de1.api.radio-browser.info",
    "musicbrainz.org",
    "api.mymemory.translated.net",
    "www.githubstatus.com",
    "www.cloudflarestatus.com",
    "discordstatus.com",
    "status.openai.com",
}


class WeatherTruthArguments(BaseModel):
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    forecast_days: int = Field(default=2, ge=1, le=7)


class SunTimesTruthArguments(BaseModel):
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    date: str = "today"

    @field_validator("date")
    @classmethod
    def _date_is_iso_or_today(cls, value: str) -> str:
        if value == "today":
            return value
        date.fromisoformat(value)
        return value


class HolidaysTruthArguments(BaseModel):
    country_code: str
    year: int = Field(default_factory=lambda: datetime.now(UTC).year, ge=1970, le=2100)

    @field_validator("country_code")
    @classmethod
    def _country_code(cls, value: str) -> str:
        normalized = value.strip().upper()
        if len(normalized) != 2 or not normalized.isalpha():
            raise ValueError("country_code must be a two-letter ISO code")
        return normalized


class GeocodeTruthArguments(BaseModel):
    operation: Literal["search", "reverse"] = "search"
    query: str | None = Field(default=None, max_length=200)
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    limit: int = Field(default=3, ge=1, le=5)

    @model_validator(mode="after")
    def _operation_has_inputs(self) -> GeocodeTruthArguments:
        if self.operation == "search" and not (self.query and self.query.strip()):
            raise ValueError("search requires query")
        if self.operation == "reverse" and (self.latitude is None or self.longitude is None):
            raise ValueError("reverse requires latitude and longitude")
        return self


class NewsTruthArguments(BaseModel):
    query: str | None = Field(default=None, max_length=200)
    category: Literal[
        "ai",
        "business",
        "culture",
        "entertainment",
        "finance",
        "general",
        "health",
        "lifestyle",
        "opinion",
        "politics",
        "science",
        "sports",
        "tech",
        "weather",
        "world",
    ] = "general"
    limit: int = Field(default=5, ge=1, le=10)


class RadioTruthArguments(BaseModel):
    query: str = Field(max_length=100)
    country_code: str | None = None
    tag: str | None = Field(default=None, max_length=50)
    limit: int = Field(default=5, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def _query_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value.strip()

    @field_validator("country_code")
    @classmethod
    def _optional_country_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if len(normalized) != 2 or not normalized.isalpha():
            raise ValueError("country_code must be a two-letter ISO code")
        return normalized


class MusicTruthArguments(BaseModel):
    entity: Literal["artist", "recording", "release", "release-group"] = "artist"
    query: str = Field(max_length=200)
    limit: int = Field(default=5, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def _music_query_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value.strip()


class TranslateTruthArguments(BaseModel):
    text: str = Field(max_length=2_000)
    source_language: str = Field(max_length=20)
    target_language: str = Field(max_length=20)

    @field_validator("text", "source_language", "target_language")
    @classmethod
    def _translation_text_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value.strip()


class ServiceStatusTruthArguments(BaseModel):
    provider: Literal["github", "cloudflare", "discord", "openai"]


@dataclass(frozen=True)
class JsonResponse:
    data: Any
    final_url: str
    status_code: int


class JsonTransport(Protocol):
    async def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout_seconds: int,
        max_response_bytes: int,
    ) -> JsonResponse: ...


class UrllibJsonTransport:
    async def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout_seconds: int,
        max_response_bytes: int,
    ) -> JsonResponse:
        return await asyncio.to_thread(
            self._get_json_sync,
            url,
            headers,
            timeout_seconds,
            max_response_bytes,
        )

    @staticmethod
    def _get_json_sync(
        url: str,
        headers: dict[str, str],
        timeout_seconds: int,
        max_response_bytes: int,
    ) -> JsonResponse:
        last_network_error: urllib.error.URLError | None = None
        for attempt in range(2):
            request = urllib.request.Request(url, headers=headers, method="GET")
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    payload = response.read(max_response_bytes + 1)
                    if len(payload) > max_response_bytes:
                        raise ValueError("external truth response exceeded byte limit")
                    charset = response.headers.get_content_charset() or "utf-8"
                    data = json.loads(payload.decode(charset, errors="strict"))
                    return JsonResponse(
                        data=data,
                        final_url=response.geturl(),
                        status_code=int(response.status),
                    )
            except urllib.error.HTTPError as exc:
                raise ValueError(f"external provider returned HTTP {exc.code}") from exc
            except urllib.error.URLError as exc:
                last_network_error = exc
                if attempt == 0:
                    time.sleep(0.25)
            except json.JSONDecodeError as exc:
                raise ValueError("external provider returned invalid JSON") from exc
        assert last_network_error is not None
        raise ValueError(
            f"external provider request failed: {last_network_error.reason}"
        ) from last_network_error


@dataclass(frozen=True)
class _CacheEntry:
    stored_at: float
    response: JsonResponse


class ExternalTruthExecutor:
    """Fixed-provider network client with caching, rate limits, and provenance."""

    def __init__(
        self,
        config: ExternalTruthConfig,
        *,
        transport: JsonTransport | None = None,
    ) -> None:
        self._config = config
        self._transport = transport or UrllibJsonTransport()
        self._cache: dict[str, _CacheEntry] = {}
        self._last_request: dict[str, float] = {}
        self._lock = asyncio.Lock()

    def register_into(self, registry: ToolRegistry) -> None:
        capabilities = {
            "weather": ToolCapability(
                name="truth_weather",
                description=(
                    "Read current conditions and a short forecast from Open-Meteo. Returns "
                    "provider-reported external observation data with source provenance."
                ),
                arguments_model=WeatherTruthArguments,
                handler=self.weather,
            ),
            "sun_times": ToolCapability(
                name="truth_sun_times",
                description=(
                    "Read sunrise, sunset, and twilight times for coordinates from Sunrise-Sunset."
                ),
                arguments_model=SunTimesTruthArguments,
                handler=self.sun_times,
            ),
            "holidays": ToolCapability(
                name="truth_holidays",
                description="Read public holidays for an ISO country and year from Nager.Date.",
                arguments_model=HolidaysTruthArguments,
                handler=self.holidays,
            ),
            "geocode": ToolCapability(
                name="truth_geocode",
                description=(
                    "Forward or reverse geocode with OpenStreetMap Nominatim. Use only when "
                    "location context materially helps the owner's request."
                ),
                arguments_model=GeocodeTruthArguments,
                handler=self.geocode,
            ),
            "news": ToolCapability(
                name="truth_news",
                description="Read or search current curated headlines from Noozra.",
                arguments_model=NewsTruthArguments,
                handler=self.news,
            ),
            "radio": ToolCapability(
                name="truth_radio",
                description="Find playable internet radio stations through Radio Browser.",
                arguments_model=RadioTruthArguments,
                handler=self.radio,
            ),
            "music": ToolCapability(
                name="truth_music",
                description="Search canonical music metadata through MusicBrainz.",
                arguments_model=MusicTruthArguments,
                handler=self.music,
            ),
            "translate": ToolCapability(
                name="truth_translate",
                description="Translate short text through the keyless MyMemory translation API.",
                arguments_model=TranslateTruthArguments,
                handler=self.translate,
            ),
            "service_status": ToolCapability(
                name="truth_service_status",
                description=(
                    "Read current public incident status for GitHub, Cloudflare, Discord, or "
                    "OpenAI from their official status pages."
                ),
                arguments_model=ServiceStatusTruthArguments,
                handler=self.service_status,
            ),
        }
        for source in self._config.enabled_sources:
            registry.register(capabilities[source])

    async def weather(self, raw: dict[str, Any], _context: ToolAutonomyContext) -> ToolOutcome:
        args = WeatherTruthArguments.model_validate(raw)
        query = urlencode(
            {
                "latitude": args.latitude,
                "longitude": args.longitude,
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,"
                    "precipitation,rain,weather_code,cloud_cover,wind_speed_10m"
                ),
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,sunrise,sunset",
                "forecast_days": args.forecast_days,
                "timezone": "auto",
            }
        )
        return await self._observe("open-meteo", f"https://api.open-meteo.com/v1/forecast?{query}")

    async def sun_times(self, raw: dict[str, Any], _context: ToolAutonomyContext) -> ToolOutcome:
        args = SunTimesTruthArguments.model_validate(raw)
        query = urlencode(
            {
                "lat": args.latitude,
                "lng": args.longitude,
                "date": args.date,
                "formatted": 0,
            }
        )
        return await self._observe("sunrise-sunset", f"https://api.sunrise-sunset.org/json?{query}")

    async def holidays(self, raw: dict[str, Any], _context: ToolAutonomyContext) -> ToolOutcome:
        args = HolidaysTruthArguments.model_validate(raw)
        url = f"https://date.nager.at/api/v3/PublicHolidays/{args.year}/{args.country_code}"
        return await self._observe("nager-date", url)

    async def geocode(self, raw: dict[str, Any], _context: ToolAutonomyContext) -> ToolOutcome:
        args = GeocodeTruthArguments.model_validate(raw)
        if args.operation == "search":
            query = urlencode(
                {
                    "q": args.query,
                    "format": "jsonv2",
                    "addressdetails": 1,
                    "limit": args.limit,
                }
            )
            url = f"https://nominatim.openstreetmap.org/search?{query}"
        else:
            query = urlencode(
                {
                    "lat": args.latitude,
                    "lon": args.longitude,
                    "format": "jsonv2",
                    "addressdetails": 1,
                }
            )
            url = f"https://nominatim.openstreetmap.org/reverse?{query}"
        return await self._observe("nominatim", url)

    async def news(self, raw: dict[str, Any], _context: ToolAutonomyContext) -> ToolOutcome:
        args = NewsTruthArguments.model_validate(raw)
        if args.query and args.query.strip():
            query = urlencode({"q": args.query.strip(), "limit": args.limit})
            url = f"https://noozra.com/api/search?{query}"
        else:
            query = urlencode({"category": args.category, "limit": args.limit})
            url = f"https://noozra.com/api/articles?{query}"
        return await self._observe("noozra", url, normalizer=_normalize_news)

    async def radio(self, raw: dict[str, Any], _context: ToolAutonomyContext) -> ToolOutcome:
        args = RadioTruthArguments.model_validate(raw)
        params: dict[str, object] = {
            "name": args.query,
            "limit": args.limit,
            "hidebroken": "true",
            "order": "votes",
            "reverse": "true",
        }
        if args.country_code:
            params["countrycode"] = args.country_code
        if args.tag:
            params["tag"] = args.tag
        url = "https://de1.api.radio-browser.info/json/stations/search?" + urlencode(params)
        return await self._observe("radio-browser", url, normalizer=_normalize_radio)

    async def music(self, raw: dict[str, Any], _context: ToolAutonomyContext) -> ToolOutcome:
        args = MusicTruthArguments.model_validate(raw)
        query = urlencode({"query": args.query, "fmt": "json", "limit": args.limit})
        url = f"https://musicbrainz.org/ws/2/{args.entity}/?{query}"
        return await self._observe("musicbrainz", url)

    async def translate(self, raw: dict[str, Any], _context: ToolAutonomyContext) -> ToolOutcome:
        args = TranslateTruthArguments.model_validate(raw)
        query = urlencode(
            {
                "q": args.text,
                "langpair": f"{args.source_language}|{args.target_language}",
            }
        )
        return await self._observe(
            "mymemory",
            f"https://api.mymemory.translated.net/get?{query}",
            normalizer=_normalize_translation,
        )

    async def service_status(
        self, raw: dict[str, Any], _context: ToolAutonomyContext
    ) -> ToolOutcome:
        args = ServiceStatusTruthArguments.model_validate(raw)
        urls = {
            "github": "https://www.githubstatus.com/api/v2/summary.json",
            "cloudflare": "https://www.cloudflarestatus.com/api/v2/summary.json",
            "discord": "https://discordstatus.com/api/v2/summary.json",
            "openai": "https://status.openai.com/api/v2/summary.json",
        }
        return await self._observe(
            f"{args.provider}-status",
            urls[args.provider],
            normalizer=_normalize_status,
        )

    async def _observe(
        self,
        provider: str,
        url: str,
        *,
        normalizer: Normalizer | None = None,
    ) -> ToolOutcome:
        response, cached = await self._request(provider, url)
        data = normalizer(response.data) if normalizer is not None else response.data
        retrieved_at_ms = time.time_ns() // 1_000_000
        envelope = {
            "contract": "external-observation-v1",
            "truth_status": "provider_reported",
            "provider": provider,
            "source_url": response.final_url,
            "retrieved_at_ms": retrieved_at_ms,
            "cached": cached,
            "data": data,
        }
        return ToolOutcome(
            ok=True,
            output=json.dumps(envelope, ensure_ascii=False, sort_keys=True),
            metadata={
                "external_truth": True,
                "source_kind": "world_observed",
                "truth_status": "provider_reported",
                "provider": provider,
                "source_url": response.final_url,
                "status_code": response.status_code,
                "retrieved_at_ms": retrieved_at_ms,
                "cached": cached,
            },
        )

    async def _request(self, provider: str, url: str) -> tuple[JsonResponse, bool]:
        _validate_provider_url(url)
        async with self._lock:
            now = time.monotonic()
            cached = self._cache.get(url)
            if cached is not None and now - cached.stored_at <= self._config.cache_ttl_seconds:
                return cached.response, True
            last_request = self._last_request.get(provider)
            min_interval = self._config.min_request_interval_ms / 1_000.0
            if last_request is not None:
                delay = min_interval - (now - last_request)
                if delay > 0:
                    await asyncio.sleep(delay)
            response = await self._transport.get_json(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": self._config.user_agent,
                },
                timeout_seconds=self._config.request_timeout_seconds,
                max_response_bytes=self._config.max_response_bytes,
            )
            _validate_provider_url(response.final_url)
            self._last_request[provider] = time.monotonic()
            if self._config.cache_ttl_seconds > 0:
                self._cache[url] = _CacheEntry(time.monotonic(), response)
            return response, False


Normalizer = Callable[[Any], Any]


def _validate_provider_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_HOSTS:
        raise ValueError("external truth provider URL is outside the fixed HTTPS allowlist")


def _normalize_news(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    articles = data.get("articles", [])
    if not isinstance(articles, list):
        return data
    return {
        "query": data.get("query"),
        "category": data.get("category"),
        "articles": [
            {
                key: item.get(key)
                for key in ("headline", "url", "source", "published_at", "summary", "category")
                if item.get(key) is not None
            }
            for item in articles
            if isinstance(item, dict)
        ],
    }


def _normalize_radio(data: Any) -> Any:
    if not isinstance(data, list):
        return data
    fields = (
        "stationuuid",
        "name",
        "url_resolved",
        "homepage",
        "tags",
        "country",
        "countrycode",
        "language",
        "codec",
        "bitrate",
        "votes",
    )
    return [{key: item.get(key) for key in fields} for item in data if isinstance(item, dict)]


def _normalize_translation(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    response_data = data.get("responseData")
    return {
        "responseData": response_data,
        "responseStatus": data.get("responseStatus"),
        "quotaFinished": data.get("quotaFinished"),
    }


def _normalize_status(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    components = data.get("components", [])
    incidents = data.get("incidents", [])
    return {
        "page": data.get("page"),
        "status": data.get("status"),
        "degraded_components": [
            {
                "name": item.get("name"),
                "status": item.get("status"),
                "updated_at": item.get("updated_at"),
            }
            for item in components
            if isinstance(item, dict) and item.get("status") != "operational"
        ],
        "active_incidents": [
            {
                "name": item.get("name"),
                "status": item.get("status"),
                "impact": item.get("impact"),
                "updated_at": item.get("updated_at"),
                "shortlink": item.get("shortlink"),
            }
            for item in incidents
            if isinstance(item, dict) and item.get("status") != "resolved"
        ],
    }


__all__ = [
    "ExternalTruthExecutor",
    "GeocodeTruthArguments",
    "HolidaysTruthArguments",
    "JsonResponse",
    "JsonTransport",
    "MusicTruthArguments",
    "NewsTruthArguments",
    "RadioTruthArguments",
    "ServiceStatusTruthArguments",
    "SunTimesTruthArguments",
    "TranslateTruthArguments",
    "UrllibJsonTransport",
    "WeatherTruthArguments",
]
