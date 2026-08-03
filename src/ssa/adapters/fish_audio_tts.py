"""Fish Audio live WebSocket TTS adapter."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

DEFAULT_FISH_AUDIO_TTS_URL = "wss://api.fish.audio/v1/tts/live"
DEFAULT_FISH_AUDIO_API_URL = "https://api.fish.audio"
FISH_AUDIO_LIVE_MODELS = frozenset({"s1", "s2-pro"})
FISH_AUDIO_HTTP_MODELS = frozenset({"s2.1-pro", "s2.1-pro-free"})
FISH_AUDIO_MODELS = FISH_AUDIO_LIVE_MODELS | FISH_AUDIO_HTTP_MODELS


class FishAudioTTSError(RuntimeError):
    """Authentication, protocol, or synthesis failure."""


@dataclass(frozen=True)
class FishAudioVoice:
    id: str
    title: str
    description: str = ""
    state: str = ""
    visibility: str = ""
    languages: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    cover_image: str | None = None


@dataclass(frozen=True)
class FishAudioVoicePage:
    total: int
    items: tuple[FishAudioVoice, ...]
    has_more: bool = False


@dataclass(frozen=True)
class FishAudioEmotionMarkup:
    """Model-specific Fish Audio emotion tags placed at sentence boundaries."""

    s1_tags: tuple[str, ...] = ()
    s2_tags: tuple[str, ...] = ()

    def apply(self, text: str, model: str) -> str:
        tags = self.s1_tags if model == "s1" else self.s2_tags
        if not tags:
            return text
        wrapper = "({})" if model == "s1" else "[{}]"
        prefix = "".join(wrapper.format(tag) for tag in tags)
        return f"{prefix} {text}"


@dataclass(frozen=True)
class FishAudioTextSegment:
    text: str
    emotion_markup: FishAudioEmotionMarkup | None = None


_FISH_S1_RECIPE_TAGS: dict[str, tuple[str, ...]] = {
    "tender_ache": ("sad",),
    "jealous_soft": ("angry",),
    "hurt_composed": ("sad",),
    "relieved_tears": ("happy",),
    "playful_tease": ("happy",),
    "anxious_care": ("sad",),
    "warm_pride": ("happy",),
    "shy_longing": ("sad",),
    "calm": (),
}

_FISH_S2_RECIPE_DESCRIPTIONS: dict[str, str] = {
    "tender_ache": "sad",
    "jealous_soft": "jealous",
    "hurt_composed": "sad",
    "relieved_tears": "relieved",
    "playful_tease": "happy",
    "anxious_care": "worried",
    "warm_pride": "happy",
    "shy_longing": "shy",
    "venomous_sister": "sarcastic",
    "calm": "calm",
}


def fish_audio_emotion_markup(
    *,
    recipe: str,
    intensity: float,
    breathiness: float = 0.2,
    tremor: float = 0.04,
    energy: float = 0.5,
    pause_before_ms: int = 0,
    ending: str = "natural_fall",
    has_emphasis: bool = False,
) -> FishAudioEmotionMarkup:
    """Translate Zero's layered voice style into official Fish tag syntax."""
    description = _FISH_S2_RECIPE_DESCRIPTIONS.get(
        recipe,
        _FISH_S2_RECIPE_DESCRIPTIONS["calm"],
    )
    intensity_hint = "softly" if intensity <= 0.55 else "naturally"
    s2_tags = [f"{description} {intensity_hint}"]
    # Sound-effect tags such as whisper/sigh/laugh can introduce breath bursts,
    # vocalizations, or identity drift. Keep Fish emotion control semantic here.
    # S1 emotion tokens at character zero can deform the voice onset. Preserve
    # its reference voice with clean text and reserve semantic tags for S2 Pro.
    return FishAudioEmotionMarkup((), tuple(s2_tags))


async def list_fish_audio_voices(
    *,
    api_key: str,
    self_only: bool = True,
    title: str | None = None,
    page_size: int = 30,
    page_number: int = 1,
    api_url: str = DEFAULT_FISH_AUDIO_API_URL,
    timeout_seconds: float = 20.0,
) -> FishAudioVoicePage:
    """List Fish Audio voice models through the documented ``GET /model`` API."""
    normalized_key = api_key.strip()
    if not normalized_key:
        raise ValueError("Fish Audio API key must not be empty")
    page_size = max(1, min(100, page_size))
    page_number = max(1, page_number)
    from urllib.parse import urlencode

    query: dict[str, str | int] = {
        "self": str(self_only).lower(),
        "page_size": page_size,
        "page_number": page_number,
        "sort_by": "created_at" if self_only else "task_count",
    }
    if title and title.strip():
        query["title"] = title.strip()
    url = f"{api_url.rstrip('/')}/model?{urlencode(query)}"

    def fetch() -> dict[str, Any]:
        request = Request(
            url,
            headers={
                "Authorization": f"Bearer {normalized_key}",
                "Accept": "application/json",
                "User-Agent": "Zero/1.0 FishAudioVoiceSync",
            },
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise FishAudioTTSError(
                f"Fish Audio voice list HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise FishAudioTTSError(f"Fish Audio voice list failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise FishAudioTTSError("Fish Audio voice list returned an invalid response")
        return payload

    payload = await asyncio.to_thread(fetch)
    voices: list[FishAudioVoice] = []
    raw_items = payload.get("items")
    if isinstance(raw_items, list):
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            voice_id = raw.get("id") or raw.get("_id")
            title_value = raw.get("title")
            if not isinstance(voice_id, str) or not voice_id.strip():
                continue
            languages_value = raw.get("languages", raw.get("language", []))
            tags_value = raw.get("tags", [])
            languages = (
                tuple(str(value) for value in languages_value if value)
                if isinstance(languages_value, list)
                else (str(languages_value),) if languages_value else ()
            )
            tags = (
                tuple(str(value) for value in tags_value if value)
                if isinstance(tags_value, list)
                else ()
            )
            cover = raw.get("cover_image") or raw.get("cover_image_url") or raw.get("cover")
            voices.append(
                FishAudioVoice(
                    id=voice_id.strip(),
                    title=title_value.strip() if isinstance(title_value, str) and title_value.strip() else voice_id.strip(),
                    description=str(raw.get("description") or ""),
                    state=str(raw.get("state") or ""),
                    visibility=str(raw.get("visibility") or ""),
                    languages=languages,
                    tags=tags,
                    cover_image=cover if isinstance(cover, str) and cover else None,
                )
            )
    total = payload.get("total")
    return FishAudioVoicePage(
        total=total if isinstance(total, int) else len(voices),
        items=tuple(voices),
        has_more=payload.get("has_more") is True,
    )


@dataclass(frozen=True)
class FishAudioTTSStreamEvent:
    audio: bytes = b""
    finished: bool = False
    subtitle: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    log_id: str | None = None


class FishAudioTTSAdapter:
    def __init__(
        self,
        *,
        api_key: str,
        websocket_url: str = DEFAULT_FISH_AUDIO_TTS_URL,
        model: str = "s1",
        reference_id: str,
        audio_format: str = "mp3",
        sample_rate: int | None = 44_100,
        timeout_seconds: float = 60.0,
        max_audio_bytes: int = 10 * 1024 * 1024,
        context_texts: Sequence[str] = (),
    ) -> None:
        parsed = urlsplit(websocket_url)
        if parsed.scheme not in {"http", "https", "ws", "wss"} or not parsed.netloc:
            raise ValueError("Fish Audio base URL must be an HTTP(S) or WS(S) URL")
        if not api_key.strip():
            raise ValueError("Fish Audio API key must not be empty")
        if not model.strip() or not reference_id.strip():
            raise ValueError("Fish Audio model and reference_id must not be empty")
        if model.strip() not in FISH_AUDIO_MODELS:
            raise ValueError(
                "Fish Audio model must be s1, s2-pro, s2.1-pro, or s2.1-pro-free"
            )
        if audio_format not in {"mp3", "wav", "pcm", "opus"}:
            raise ValueError("Fish Audio audio format is invalid")
        if timeout_seconds <= 0 or max_audio_bytes <= 0:
            raise ValueError("Fish Audio numeric settings must be positive")
        self._api_key = api_key.strip()
        self._websocket_url = websocket_url
        self._model = model.strip()
        self._reference_id = reference_id.strip()
        self._audio_format = audio_format
        self._sample_rate = sample_rate
        self._timeout = timeout_seconds
        self._max_audio_bytes = max_audio_bytes
        self._websocket: Any | None = None

    @property
    def _uses_live_websocket(self) -> bool:
        return self._model in FISH_AUDIO_LIVE_MODELS

    def _http_tts_url(self) -> str:
        parsed = urlsplit(self._websocket_url)
        scheme = "https" if parsed.scheme in {"https", "wss"} else "http"
        path = parsed.path.rstrip("/")
        if path.endswith("/tts/live"):
            path = path[: -len("/live")]
        elif not path.endswith("/tts"):
            path = "/v1/tts"
        return f"{scheme}://{parsed.netloc}{path}"

    @property
    def connection_usage(self) -> dict[str, Any] | None:
        return None

    @property
    def log_id(self) -> str | None:
        return None

    async def __aenter__(self) -> FishAudioTTSAdapter:
        await self.open()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def open(self) -> None:
        if not self._uses_live_websocket:
            return
        if self._websocket is not None:
            return
        try:
            import msgpack
            import websockets

            self._msgpack = msgpack
            self._websocket = await asyncio.wait_for(
                websockets.connect(
                    self._websocket_url,
                    additional_headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "model": self._model,
                    },
                    max_size=self._max_audio_bytes,
                    open_timeout=self._timeout,
                    ping_interval=20,
                    ping_timeout=20,
                    proxy=None,
                ),
                timeout=self._timeout,
            )
        except ModuleNotFoundError as exc:
            raise FishAudioTTSError("Fish Audio requires the msgpack package") from exc
        except Exception as exc:
            await self.close()
            raise FishAudioTTSError(f"Fish Audio connection failed: {exc}") from exc

    async def close(self) -> None:
        websocket = self._websocket
        self._websocket = None
        if websocket is not None:
            with suppress(Exception):
                await websocket.close()

    async def stream_segments(
        self,
        texts: AsyncIterable[str],
        *,
        context_texts: Sequence[str] | None = None,
        prosody_speed: float = 1.0,
        prosody_volume: float = 0.0,
        temperature: float = 0.7,
        top_p: float = 0.7,
        emotion_markup: FishAudioEmotionMarkup | None = None,
    ) -> AsyncIterator[FishAudioTTSStreamEvent]:
        async def styled_texts() -> AsyncIterator[FishAudioTextSegment]:
            async for text in texts:
                yield FishAudioTextSegment(text=text, emotion_markup=emotion_markup)

        async for event in self.stream_styled_segments(
            styled_texts(),
            prosody_speed=prosody_speed,
            prosody_volume=prosody_volume,
            temperature=temperature,
            top_p=top_p,
        ):
            yield event

    async def stream_styled_segments(
        self,
        segments: AsyncIterable[FishAudioTextSegment],
        *,
        prosody_speed: float = 1.0,
        prosody_volume: float = 0.0,
        temperature: float = 0.7,
        top_p: float = 0.7,
    ) -> AsyncIterator[FishAudioTTSStreamEvent]:
        """Synthesize a whole reply in one connection while changing emotion per segment."""
        if self._uses_live_websocket and self._websocket is None:
            await self.open()
        request: dict[str, Any] = {
            "text": "",
            "reference_id": self._reference_id,
            "format": self._audio_format,
            "latency": "normal",
            "features": ["quality-guard"],
            "condition_on_previous_chunks": True,
            "normalize": True,
            "chunk_length": 300,
            "min_chunk_length": 50,
            "repetition_penalty": 1.35,
            "max_new_tokens": 1024,
            "early_stop_threshold": 1,
            "temperature": max(0.0, min(1.0, temperature)),
            "top_p": max(0.0, min(1.0, top_p)),
        }
        if self._audio_format == "mp3":
            request["mp3_bitrate"] = 192
        normalized_speed = max(0.5, min(2.0, prosody_speed))
        normalized_volume = max(-20.0, min(20.0, prosody_volume))
        if normalized_speed != 1.0 or normalized_volume != 0.0:
            request["prosody"] = {
                "speed": normalized_speed,
                "volume": normalized_volume,
            }
        if self._sample_rate is not None:
            request["sample_rate"] = self._sample_rate
        try:
            collected: list[FishAudioTextSegment] = []
            async for segment in segments:
                text = segment.text.strip()
                if text:
                    collected.append(FishAudioTextSegment(text, segment.emotion_markup))
            if not collected:
                raise FishAudioTTSError("Fish Audio TTS text must not be empty")
            collected = self._stabilize_short_reply(collected)
            combined_text = self._render_text(collected)
            if not self._uses_live_websocket:
                plain_text = "".join(segment.text for segment in collected)
                request["text"] = combined_text
                audio = await asyncio.to_thread(self._synthesize_http, request)
                if self._audio_is_suspiciously_long(audio, plain_text):
                    retry_request = {
                        **request,
                        "text": plain_text,
                        "condition_on_previous_chunks": False,
                        "repetition_penalty": 1.5,
                        "temperature": min(float(request["temperature"]), 0.55),
                        "top_p": min(float(request["top_p"]), 0.65),
                    }
                    audio = await asyncio.to_thread(self._synthesize_http, retry_request)
                    if self._audio_is_suspiciously_long(audio, plain_text):
                        raise FishAudioTTSError(
                            "Fish Audio returned implausibly long audio for a short reply"
                        )
                yield FishAudioTTSStreamEvent(audio=audio)
                yield FishAudioTTSStreamEvent(finished=True)
                return
            await self._send({"event": "start", "request": request})
            audio_bytes = 0
            await self._send({"event": "text", "text": combined_text})
            await self._send({"event": "flush"})
            await self._send({"event": "close"})
            while True:
                payload = await self._receive()
                event = payload.get("event") if isinstance(payload, dict) else None
                if event == "audio":
                    audio = payload.get("audio")
                    if isinstance(audio, str):
                        audio = audio.encode("latin1")
                    if not isinstance(audio, (bytes, bytearray)):
                        continue
                    audio_bytes += len(audio)
                    if audio_bytes > self._max_audio_bytes:
                        raise FishAudioTTSError("Fish Audio audio exceeded the configured size limit")
                    yield FishAudioTTSStreamEvent(audio=bytes(audio))
                elif event == "finish":
                    if payload.get("reason") not in {None, "stop"}:
                        raise FishAudioTTSError(f"Fish Audio synthesis failed: {payload}")
                    yield FishAudioTTSStreamEvent(finished=True)
                    await self.close()
                    return
                elif event == "error":
                    raise FishAudioTTSError(f"Fish Audio synthesis failed: {payload}")
        except asyncio.CancelledError:
            raise
        except FishAudioTTSError:
            raise
        except Exception as exc:
            raise FishAudioTTSError(f"Fish Audio synthesis failed: {exc}") from exc

    def _render_text(self, collected: Sequence[FishAudioTextSegment]) -> str:
        if self._model == "s1":
            return "".join(segment.text for segment in collected)
        rendered: list[str] = []
        previous_markup: FishAudioEmotionMarkup | None = None
        for segment in collected:
            text = segment.text
            markup = segment.emotion_markup
            if markup is not None and markup != previous_markup:
                text = markup.apply(text, self._model)
                previous_markup = markup
            rendered.append(text)
        return "".join(rendered)

    @staticmethod
    def _stabilize_short_reply(
        collected: Sequence[FishAudioTextSegment],
    ) -> list[FishAudioTextSegment]:
        """Keep short replies in one vocal condition to prevent tail regeneration."""
        total_text = "".join(segment.text for segment in collected)
        if len(total_text) > 36 or len(collected) <= 1:
            return list(collected)
        primary_markup = next(
            (segment.emotion_markup for segment in collected if segment.emotion_markup is not None),
            None,
        )
        return [FishAudioTextSegment(total_text, primary_markup)]

    @staticmethod
    def _audio_is_suspiciously_long(audio: bytes, rendered_text: str) -> bool:
        """Detect implausibly long 192 kbps MP3 output before it reaches playback."""
        spoken_chars = sum(character.isalnum() for character in rendered_text)
        expected_ceiling = max(180_000, spoken_chars * 10_000)
        return len(audio) > expected_ceiling

    def _synthesize_http(self, request_payload: dict[str, Any]) -> bytes:
        try:
            import msgpack
        except ModuleNotFoundError as exc:
            raise FishAudioTTSError("Fish Audio requires the msgpack package") from exc
        request = Request(
            self._http_tts_url(),
            data=msgpack.packb(request_payload, use_bin_type=True),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/msgpack",
                "Accept": "audio/mpeg, application/octet-stream",
                "model": self._model,
                "User-Agent": "Zero/1.0 FishAudioTTS",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                audio = response.read(self._max_audio_bytes + 1)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise FishAudioTTSError(
                f"Fish Audio TTS HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise FishAudioTTSError(f"Fish Audio HTTP TTS failed: {exc}") from exc
        if not audio:
            raise FishAudioTTSError("Fish Audio HTTP TTS returned empty audio")
        if len(audio) > self._max_audio_bytes:
            raise FishAudioTTSError("Fish Audio audio exceeded the configured size limit")
        return audio

    async def _send(self, payload: dict[str, Any]) -> None:
        assert self._websocket is not None
        await self._websocket.send(self._msgpack.packb(payload, use_bin_type=True))

    async def _receive(self) -> dict[str, Any]:
        assert self._websocket is not None
        try:
            raw = await asyncio.wait_for(self._websocket.recv(), timeout=self._timeout)
        except TimeoutError as exc:
            raise FishAudioTTSError("timed out waiting for Fish Audio TTS") from exc
        if isinstance(raw, str):
            raise FishAudioTTSError(f"Fish Audio returned an unexpected text frame: {raw[:160]}")
        payload = self._msgpack.unpackb(raw, raw=False)
        if not isinstance(payload, dict):
            raise FishAudioTTSError("Fish Audio returned an invalid MessagePack frame")
        return payload
