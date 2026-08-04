"""Cached public GitHub activity used by the community recruitment page."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_REPOSITORIES = (
    ("zero-desktop", "https://github.com/sikehuang88/zero-desktop"),
    ("zero-hdsc", "https://github.com/sikehuang88/zero-hdsc"),
)

_FALLBACK: dict[str, dict[str, Any]] = {
    "zero-desktop": {
        "description": "ZERO desktop client and interactive digital-life workspace.",
        "language": "TypeScript",
        "stars": 0,
        "forks": 0,
        "open_items": 0,
        "updated_at": None,
    },
    "zero-hdsc": {
        "description": "The memory, voice, tools and agent runtime behind ZERO.",
        "language": "Python",
        "stars": 0,
        "forks": 0,
        "open_items": 0,
        "updated_at": None,
    },
}


@dataclass(frozen=True)
class _CacheEntry:
    expires_at: float
    payload: dict[str, Any]


class CommunityGithubService:
    """Fetch repository summaries without making the page depend on GitHub."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 2.5,
        cache_ttl_seconds: float = 900.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._cache_ttl_seconds = cache_ttl_seconds
        self._clock = clock
        self._cache: _CacheEntry | None = None
        self._lock = threading.Lock()

    def get_activity(self) -> dict[str, Any]:
        now = self._clock()
        cached = self._cache
        if cached is not None and cached.expires_at > now:
            return cached.payload

        with self._lock:
            now = self._clock()
            cached = self._cache
            if cached is not None and cached.expires_at > now:
                return cached.payload

            repositories = [self._fetch_repository(name, url) for name, url in _REPOSITORIES]
            payload = {
                "generated_at": datetime.now(UTC).isoformat(),
                "live": all(repository["live"] for repository in repositories),
                "repositories": repositories,
            }
            self._cache = _CacheEntry(now + self._cache_ttl_seconds, payload)
            return payload

    def _fetch_repository(self, name: str, html_url: str) -> dict[str, Any]:
        api_url = f"https://api.github.com/repos/sikehuang88/{name}"
        try:
            request = Request(
                api_url,
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "zero-community-page/1.0",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            with urlopen(request, timeout=self._timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("GitHub repository response was not an object")
            return {
                "name": name,
                "url": str(data.get("html_url") or html_url),
                "description": str(data.get("description") or _FALLBACK[name]["description"]),
                "language": str(data.get("language") or _FALLBACK[name]["language"]),
                "stars": _non_negative_int(data.get("stargazers_count")),
                "forks": _non_negative_int(data.get("forks_count")),
                "open_items": _non_negative_int(data.get("open_issues_count")),
                "updated_at": data.get("pushed_at") or data.get("updated_at"),
                "live": True,
            }
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            return {"name": name, "url": html_url, **_FALLBACK[name], "live": False}


def _non_negative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    return 0


__all__ = ["CommunityGithubService"]
