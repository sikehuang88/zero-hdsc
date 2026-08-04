from __future__ import annotations

import json
from typing import Any
from urllib.error import URLError

from ssa.services.community_github import CommunityGithubService


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_github_activity_is_cached(monkeypatch: Any) -> None:
    calls: list[str] = []

    def fake_urlopen(request: Any, *, timeout: float) -> _Response:
        calls.append(request.full_url)
        name = request.full_url.rsplit("/", 1)[-1]
        return _Response(
            {
                "html_url": f"https://github.com/sikehuang88/{name}",
                "description": f"Live {name}",
                "language": "TypeScript" if name == "zero-desktop" else "Python",
                "stargazers_count": 12,
                "forks_count": 3,
                "open_issues_count": 5,
                "pushed_at": "2026-08-03T10:00:00Z",
            }
        )

    monkeypatch.setattr("ssa.services.community_github.urlopen", fake_urlopen)
    service = CommunityGithubService(cache_ttl_seconds=60)

    first = service.get_activity()
    second = service.get_activity()

    assert first is second
    assert len(calls) == 2
    assert first["live"] is True
    assert first["repositories"][0]["stars"] == 12


def test_github_activity_falls_back_per_repository(monkeypatch: Any) -> None:
    def fake_urlopen(request: Any, *, timeout: float) -> _Response:
        if request.full_url.endswith("zero-desktop"):
            raise URLError("offline")
        return _Response(
            {
                "description": "Runtime",
                "language": "Python",
                "stargazers_count": 7,
                "forks_count": 2,
                "open_issues_count": 4,
            }
        )

    monkeypatch.setattr("ssa.services.community_github.urlopen", fake_urlopen)

    payload = CommunityGithubService().get_activity()

    desktop, runtime = payload["repositories"]
    assert payload["live"] is False
    assert desktop["name"] == "zero-desktop"
    assert desktop["live"] is False
    assert runtime["name"] == "zero-hdsc"
    assert runtime["live"] is True
    assert runtime["stars"] == 7
