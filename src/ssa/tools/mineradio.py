"""Mineradio local-service tools for ZERO's native music mode."""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from ssa.tools.models import (
    MineradioControlArguments,
    MineradioNoArguments,
    MineradioSearchPlayArguments,
    ToolAutonomyContext,
    ToolOutcome,
)
from ssa.tools.registry import ToolCapability, ToolRegistry

_DEFAULT_BASE_URL = "http://127.0.0.1:18789"
_PROVIDERS = ("netease", "qq", "kugou", "qishui")
_SEARCH_PATHS = {
    "netease": "/api/search",
    "qq": "/api/qq/search",
    "kugou": "/api/kugou/search",
    "qishui": "/api/qishui/search",
}
_COMPACT_RE = re.compile(r"[^0-9a-z\u4e00-\u9fff]+", re.IGNORECASE)
_LYRIC_PATHS = {
    "netease": "/api/lyric",
    "qq": "/api/qq/lyric",
    "kugou": "/api/kugou/lyric",
    "qishui": "/api/qishui/lyric",
}
# Only a leading [mm:ss], [mm:ss.xx] or [mm:ss.xxx] stamp counts. This deliberately rejects
# NetEase's enhanced-format JSON lines ({"t":0,"c":[...]}) and the [ti:]/[ar:]/[al:]/[by:]/
# [offset:]/[id:]/[hash:] metadata headers emitted by QQ and Kugou.
_LRC_STAMP_RE = re.compile(r"\[(\d{1,3}):([0-5]?\d)(?:[.:](\d{1,3}))?\]")
_MAX_LYRIC_LINES = 600
_MAX_PLAIN_CHARS = 6_000


class MineradioExecutor:
    """Search and resolve tracks through the independently running Mineradio service."""

    def __init__(self, *, base_url: str | None = None, timeout_seconds: float = 15.0) -> None:
        self._base_url = (base_url or os.environ.get("HDSC_MINERADIO_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
        self._timeout_seconds = timeout_seconds

    def register_into(self, registry: ToolRegistry) -> None:
        registry.register(
            ToolCapability(
                name="mineradio_status",
                description="Check whether ZERO's Mineradio music service is available.",
                arguments_model=MineradioNoArguments,
                handler=self.status,
            )
        )
        registry.register(
            ToolCapability(
                name="mineradio_control",
                description=(
                    "Control ZERO's native music mode with play, pause, toggle, next, or previous. "
                    "The desktop client applies the verified command to its local player."
                ),
                arguments_model=MineradioControlArguments,
                handler=self.control,
            )
        )
        registry.register(
            ToolCapability(
                name="mineradio_search_play",
                description=(
                    "Search Mineradio across supported local music providers, select the strongest "
                    "match, verify whether an audio source is available, and open it in ZERO's music mode."
                ),
                arguments_model=MineradioSearchPlayArguments,
                handler=self.search_play,
            )
        )

    async def status(
        self,
        _raw_arguments: dict[str, Any],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        try:
            version = await self._get_json("/api/app/version")
        except RuntimeError as exc:
            return ToolOutcome(ok=False, error=str(exc))
        payload = {
            "schema": "zero.mineradio.status.v1",
            "summary": f"Mineradio {version.get('version', '')} is ready".strip(),
            "ready": True,
            "version": version.get("version"),
        }
        return ToolOutcome(ok=True, output=json.dumps(payload, ensure_ascii=False))

    async def control(
        self,
        raw_arguments: dict[str, Any],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = MineradioControlArguments.model_validate(raw_arguments)
        payload = {
            "schema": "zero.mineradio.control.v1",
            "summary": f"Music command ready: {arguments.action}",
            "action": arguments.action,
        }
        return ToolOutcome(ok=True, output=json.dumps(payload, ensure_ascii=False))

    async def search_play(
        self,
        raw_arguments: dict[str, Any],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = MineradioSearchPlayArguments.model_validate(raw_arguments)
        providers = _PROVIDERS if arguments.provider == "auto" else (arguments.provider,)
        search_results = await asyncio.gather(
            *(self._search_provider(provider, arguments.query) for provider in providers),
            return_exceptions=True,
        )
        tracks: list[dict[str, Any]] = []
        errors: list[str] = []
        for provider, result in zip(providers, search_results, strict=True):
            if isinstance(result, BaseException):
                errors.append(f"{provider}: {result}")
                continue
            tracks.extend(result)
        if not tracks:
            detail = "; ".join(errors[:3]) or "no matching tracks"
            return ToolOutcome(ok=False, error=f"Mineradio search returned no tracks: {detail}")

        ranked = sorted(
            tracks,
            key=lambda track: self._match_score(arguments.query, track),
            reverse=True,
        )
        candidates = ranked[:8]
        resolved = await asyncio.gather(
            *(self._resolve_playback(track, arguments.quality) for track in candidates[:6]),
            return_exceptions=True,
        )
        selected = candidates[0]
        playback: dict[str, Any] = {
            "playable": False,
            "reason": "source_unavailable",
            "message": "No provider returned a playable audio source.",
        }
        for track, playback_result in zip(candidates[:6], resolved, strict=True):
            if isinstance(playback_result, BaseException):
                continue
            if playback_result.get("playable") is True and playback_result.get("url"):
                selected = track
                playback = self._public_playback(playback_result)
                break
            if track is selected:
                playback = self._public_playback(playback_result)

        payload = {
            "schema": "zero.mineradio.play.v1",
            "summary": (
                f"Ready to play {selected['name']} - {selected['artist']}"
                if playback.get("playable")
                else f"Found {selected['name']} - {selected['artist']}, but playback needs attention"
            ),
            "query": arguments.query,
            "track": selected,
            "playback": playback,
            "queue": candidates,
            "lyrics": await self._fetch_lyrics(selected),
        }
        return ToolOutcome(ok=True, output=json.dumps(payload, ensure_ascii=False))

    async def _search_provider(self, provider: str, query: str) -> list[dict[str, Any]]:
        payload = await self._get_json(
            _SEARCH_PATHS[provider],
            {"keywords": query, "limit": 6, "offset": 0},
        )
        songs = payload.get("songs")
        if not isinstance(songs, list):
            return []
        return [self._normalize_track(provider, song) for song in songs if isinstance(song, dict)]

    async def _resolve_playback(self, track: dict[str, Any], quality: str) -> dict[str, Any]:
        provider = str(track["provider"])
        raw_ids = track.get("provider_ids")
        ids: dict[str, Any] = raw_ids if isinstance(raw_ids, dict) else {}
        common = {"quality": quality, "fee": track.get("fee", 0)}
        if provider == "netease":
            return await self._get_json(
                "/api/song/url",
                {
                    **common,
                    "id": track["id"],
                    "name": track["name"],
                    "artist": track["artist"],
                    "album": track["album"],
                    "duration": track["duration_ms"],
                },
            )
        if provider == "qq":
            return await self._get_json(
                "/api/qq/song/url",
                {**common, "mid": ids.get("mid", track["id"]), "mediaMid": ids.get("media_mid", "")},
            )
        if provider == "kugou":
            return await self._get_json(
                "/api/kugou/song/url",
                {
                    **common,
                    "hash": ids.get("hash", track["id"]),
                    "albumId": ids.get("album_id", ""),
                    "albumAudioId": ids.get("album_audio_id", ""),
                },
            )
        return await self._get_json(
            "/api/qishui/song/url",
            {**common, "id": ids.get("provider_song_id", track["id"])},
        )

    async def _fetch_lyrics(self, track: dict[str, Any]) -> dict[str, Any]:
        """Best-effort lyric lookup.

        This never raises: any transport, decoding, or shape failure degrades to
        ``available=False`` so a missing lyric can never fail the whole search_play call.
        """
        provider = str(track.get("provider") or "")
        try:
            payload = await self._request_lyrics(provider, track)
        except Exception:  # lyrics are optional decoration; never fail search_play over them
            return self._empty_lyrics(provider)
        return self._public_lyrics(provider, payload)

    async def _request_lyrics(self, provider: str, track: dict[str, Any]) -> dict[str, Any]:
        if provider not in _LYRIC_PATHS:
            raise RuntimeError(f"Mineradio has no lyric route for provider {provider!r}")
        raw_ids = track.get("provider_ids")
        ids: dict[str, Any] = raw_ids if isinstance(raw_ids, dict) else {}
        if provider == "netease":
            return await self._get_json(_LYRIC_PATHS["netease"], {"id": track.get("id", "")})
        if provider == "qq":
            return await self._get_json(
                _LYRIC_PATHS["qq"],
                {
                    "songmid": ids.get("mid") or track.get("id", ""),
                    "id": ids.get("qq_id", ""),
                },
            )
        if provider == "kugou":
            return await self._get_json(
                _LYRIC_PATHS["kugou"],
                {
                    "hash": ids.get("hash") or track.get("id", ""),
                    "albumAudioId": ids.get("album_audio_id", ""),
                    "duration": track.get("duration_ms", 0),
                },
            )
        return await self._get_json(
            _LYRIC_PATHS["qishui"],
            {"id": ids.get("provider_song_id") or track.get("id", "")},
        )

    async def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self._get_json_sync, path, params)

    def _get_json_sync(self, path: str, params: dict[str, Any] | None) -> dict[str, Any]:
        query = urlencode({key: value for key, value in (params or {}).items() if value not in {None, ""}})
        url = f"{self._base_url}{path}{'?' + query if query else ''}"
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "ZERO/0.1"})
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Mineradio request failed ({path}): {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Mineradio returned invalid JSON for {path}")
        return payload

    @staticmethod
    def _normalize_track(provider: str, song: dict[str, Any]) -> dict[str, Any]:
        duration = int(float(song.get("duration") or song.get("duration_ms") or 0))
        if provider == "qishui" and 0 < duration < 10_000:
            duration *= 1_000
        track_id = str(song.get("id") or song.get("mid") or song.get("hash") or "")
        return {
            "provider": provider,
            "id": track_id,
            "name": str(song.get("name") or "Unknown track"),
            "artist": str(song.get("artist") or "Unknown artist"),
            "album": str(song.get("album") or ""),
            "cover": str(song.get("cover") or ""),
            "duration_ms": max(0, duration),
            "fee": int(song.get("fee") or 0),
            "provider_ids": {
                "mid": str(song.get("mid") or song.get("songmid") or ""),
                "media_mid": str(song.get("mediaMid") or song.get("media_mid") or ""),
                "qq_id": str(song.get("qqId") or ""),
                "hash": str(
                    song.get("hash")
                    or song.get("fileHash")
                    or (track_id if provider == "kugou" else "")
                ),
                "album_id": str(song.get("albumId") or song.get("album_id") or ""),
                "album_audio_id": str(song.get("albumAudioId") or song.get("album_audio_id") or song.get("mixSongId") or ""),
                "provider_song_id": str(song.get("providerSongId") or track_id),
            },
        }

    @staticmethod
    def _match_score(query: str, track: dict[str, Any]) -> int:
        normalized = _COMPACT_RE.sub("", query.casefold())
        title = _COMPACT_RE.sub("", str(track.get("name", "")).casefold())
        artist = _COMPACT_RE.sub("", str(track.get("artist", "")).casefold())
        score = 0
        if title and title in normalized:
            score += 8
        if artist and artist in normalized:
            score += 6
        if normalized and normalized in f"{title}{artist}":
            score += 10
        score += min(4, sum(1 for char in set(normalized) if char in title))
        return score

    @classmethod
    def _public_lyrics(cls, provider: str, payload: dict[str, Any]) -> dict[str, Any]:
        lines = cls._parse_lrc(payload.get("lyric"))
        if not lines:
            return cls._empty_lyrics(provider)
        # NetEase/QQ/Qishui name the translated track "tlyric"; Kugou names it "trans".
        translation = cls._parse_lrc(payload.get("tlyric") or payload.get("trans"))
        return {
            "available": True,
            "format": "lrc",
            "source": provider,
            "lines": lines,
            "translation": translation,
            "plain": cls._plain_lyrics(lines),
        }

    @staticmethod
    def _empty_lyrics(provider: str) -> dict[str, Any]:
        return {
            "available": False,
            "format": "lrc",
            "source": provider,
            "lines": [],
            "translation": [],
            "plain": "",
        }

    @staticmethod
    def _parse_lrc(raw: Any) -> list[dict[str, Any]]:
        """Parse standard LRC text into time-ordered lines.

        Rows without a leading ``[mm:ss.xx]`` stamp are dropped, which removes NetEase's
        enhanced-format JSON rows and the ``[ti:]``/``[ar:]``/``[al:]``/``[by:]``/``[offset:]``
        headers used by QQ and Kugou. One row may carry several stamps; each becomes its own line.
        """
        if not isinstance(raw, str) or not raw.strip():
            return []
        collected: list[tuple[int, int, str]] = []
        rows = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        for order, row in enumerate(rows):
            line = row.strip()
            if not line:
                continue
            stamps: list[int] = []
            cursor = 0
            while (match := _LRC_STAMP_RE.match(line, cursor)) is not None:
                minutes, seconds, fraction = match.groups()
                milliseconds = int((fraction or "0").ljust(3, "0")[:3])
                stamps.append(int(minutes) * 60_000 + int(seconds) * 1_000 + milliseconds)
                cursor = match.end()
                while cursor < len(line) and line[cursor] == " ":
                    cursor += 1
            text = line[cursor:].strip()
            if not stamps or not text:
                continue
            collected.extend((stamp, order, text) for stamp in stamps)
        collected.sort(key=lambda item: (item[0], item[1]))
        return [
            {"time_ms": time_ms, "text": text}
            for time_ms, _, text in collected[:_MAX_LYRIC_LINES]
        ]

    @staticmethod
    def _plain_lyrics(lines: list[dict[str, Any]]) -> str:
        chunks: list[str] = []
        budget = _MAX_PLAIN_CHARS
        for line in lines:
            text = str(line["text"])
            if len(text) + 1 > budget:
                break
            chunks.append(text)
            budget -= len(text) + 1
        return "\n".join(chunks)

    @staticmethod
    def _public_playback(payload: dict[str, Any]) -> dict[str, Any]:
        raw_restriction = payload.get("restriction")
        restriction: dict[str, Any] = raw_restriction if isinstance(raw_restriction, dict) else {}
        source_url = str(payload.get("url") or "")
        return {
            "playable": payload.get("playable") is True and bool(source_url),
            "provider": str(payload.get("provider") or ""),
            "trial": payload.get("trial") is True,
            "reason": str(payload.get("reason") or restriction.get("category") or ""),
            "message": str(payload.get("message") or restriction.get("message") or ""),
            "audio_path": f"/mineradio/api/audio?url={quote(source_url, safe='')}" if source_url else "",
            "quality": str(payload.get("quality") or payload.get("effectiveQuality") or ""),
        }


__all__ = ["MineradioExecutor"]
