"""Community recruitment persistence and HTTP contract coverage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ssa.config import DatabaseConfig
from ssa.interfaces.community_api import CommunityApplicationRequest, community_router
from ssa.storage.database import Database


class _GithubStub:
    def get_activity(self) -> dict[str, Any]:
        return {"generated_at": "2026-08-03T00:00:00Z", "live": False, "repositories": []}


def test_community_application_is_persisted_and_deduplicated(tmp_path: Path) -> None:
    database_path = tmp_path / "community.db"
    database = Database(DatabaseConfig(path=str(database_path)))
    database.initialize()
    try:
        routes = {route.name: route.endpoint for route in community_router(str(database_path)).routes}
        request = CommunityApplicationRequest(
            track="developer",
            display_name="Lin",
            email="LIN@example.com",
            focus="web-3d",
            profile_url="https://github.com/example",
            notes="Three.js and accessibility",
            consent=True,
        )

        first = routes["apply"](request)
        duplicate = routes["apply"](request)
        status = routes["status"]()

        assert first["application_code"].startswith("ZERO-DEV-")
        assert first["queue_position"] == 1
        assert first["created"] is True
        assert duplicate["application_code"] == first["application_code"]
        assert duplicate["created"] is False
        assert status["developer"] == {"reserved": 1, "capacity": 80}
        assert status["tester"] == {"reserved": 0, "capacity": 500}
    finally:
        database.close()


def test_tester_application_requires_platform() -> None:
    try:
        CommunityApplicationRequest(
            track="tester",
            display_name="Qing",
            email="qing@example.com",
            focus="daily-companion",
            consent=True,
        )
    except ValueError as exc:
        assert "platform" in str(exc)
    else:
        raise AssertionError("tester request without platform was accepted")


def test_community_github_endpoint_uses_activity_service(tmp_path: Path) -> None:
    routes = {
        route.name: route.endpoint
        for route in community_router(
            str(tmp_path / "community.db"),
            github_service=_GithubStub(),
        ).routes
    }

    payload = routes["github_activity"]()

    assert payload["live"] is False
    assert payload["repositories"] == []
