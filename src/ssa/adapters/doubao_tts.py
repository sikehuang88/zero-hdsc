"""Doubao Seed 2.0 bidirectional TTS adapter."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from ssa.adapters.doubao_tts_protocol import (
    EventType,
    Message,
    MsgType,
    client_event,
)

DEFAULT_DOUBAO_TTS_URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
DEFAULT_DOUBAO_SPEAKER = "zh_female_gaolengyujie_uranus_bigtts"


class DoubaoTTSError(RuntimeError):
    """Authentication, protocol, or synthesis failure."""


@dataclass(frozen=True)
class DoubaoTTSResult:
    audio: bytes
    usage: dict[str, Any] | None = None
    log_id: str | None = None


@dataclass(frozen=True)
class DoubaoTTSStreamEvent:
    audio: bytes = b""
    finished: bool = False
    subtitle: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    log_id: str | None = None


class DoubaoSeedTTSAdapter:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        app_id: str | None = None,
        access_token: str | None = None,
        websocket_url: str = DEFAULT_DOUBAO_TTS_URL,
        resource_id: str = "seed-tts-2.0",
        speaker: str = DEFAULT_DOUBAO_SPEAKER,
        audio_format: str = "mp3",
        sample_rate: int = 24_000,
        send_interval_ms: int = 5,
        timeout_seconds: float = 30.0,
        max_audio_bytes: int = 10 * 1024 * 1024,
        section_id: str | None = None,
        enable_subtitle: bool = True,
        context_texts: tuple[str, ...] = (),
    ) -> None:
        parsed = urlsplit(websocket_url)
        if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
            raise ValueError("Doubao TTS websocket_url must be a ws(s) URL")
        normalized_api_key = api_key.strip() if api_key else ""
        normalized_app_id = app_id.strip() if app_id else ""
        normalized_access_token = access_token.strip() if access_token else ""
        if not normalized_api_key and not (normalized_app_id and normalized_access_token):
            raise ValueError("Doubao TTS requires an API key or legacy app credentials")
        if not resource_id.strip() or not speaker.strip():
            raise ValueError("Doubao TTS resource_id and speaker must not be empty")
        if audio_format != "mp3":
            raise ValueError("the web voice bridge currently requires mp3 audio")
        if sample_rate <= 0 or max_audio_bytes <= 0 or timeout_seconds <= 0:
            raise ValueError("Doubao TTS numeric settings must be positive")
        self._api_key = normalized_api_key
        self._app_id = normalized_app_id
        self._access_token = normalized_access_token
        self._websocket_url = websocket_url
        self._resource_id = resource_id.strip()
        self._speaker = speaker.strip()
        self._audio_format = audio_format
        self._sample_rate = sample_rate
        self._send_interval = max(0, send_interval_ms) / 1000
        self._timeout = timeout_seconds
        self._max_audio_bytes = max_audio_bytes
        self._section_id = section_id.strip() if section_id else None
        self._enable_subtitle = enable_subtitle
        self._context_texts = tuple(
            text.strip() for text in context_texts if text.strip()
        )[:4]
        self._websocket: Any | None = None
        self._log_id: str | None = None
        self._connection_usage: dict[str, Any] | None = None

    @property
    def connection_usage(self) -> dict[str, Any] | None:
        return self._connection_usage

    @property
    def log_id(self) -> str | None:
        return self._log_id

    async def __aenter__(self) -> DoubaoSeedTTSAdapter:
        await self.open()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def open(self) -> None:
        if self._websocket is not None:
            return

        import websockets

        self._connection_usage = None
        headers = {
            "X-Api-Resource-Id": self._resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
            "X-Control-Require-Usage-Tokens-Return": "*",
        }
        if self._api_key:
            headers["X-Api-Key"] = self._api_key
        else:
            headers["X-Api-App-Id"] = self._app_id
            headers["X-Api-Access-Key"] = self._access_token
        try:
            websocket = await asyncio.wait_for(
                websockets.connect(
                    self._websocket_url,
                    additional_headers=headers,
                    max_size=self._max_audio_bytes,
                    open_timeout=self._timeout,
                    ping_interval=20,
                    ping_timeout=20,
                    proxy=None,
                ),
                timeout=self._timeout,
            )
            await self._send(websocket, client_event(EventType.START_CONNECTION))
            await self._wait_for_event(websocket, EventType.CONNECTION_STARTED)
        except Exception as exc:
            if "websocket" in locals():
                with suppress(Exception):
                    await websocket.close()
            raise DoubaoTTSError(f"Doubao TTS connection failed: {exc}") from exc

        response = getattr(websocket, "response", None)
        response_headers = getattr(response, "headers", None)
        self._log_id = response_headers.get("x-tt-logid") if response_headers is not None else None
        self._websocket = websocket

    async def close(self) -> None:
        websocket = self._websocket
        self._websocket = None
        if websocket is None:
            return
        with suppress(Exception):
            await self._send(websocket, client_event(EventType.FINISH_CONNECTION))
            finished = await self._wait_for_event(websocket, EventType.CONNECTION_FINISHED)
            payload = finished.json_payload()
            reported_usage = payload.get("usage") if payload else None
            if isinstance(reported_usage, dict):
                self._connection_usage = {
                    str(key): value for key, value in reported_usage.items()
                }
        with suppress(Exception):
            await websocket.close()
        self._log_id = None

    async def synthesize(
        self,
        text: str,
        *,
        context_texts: Sequence[str] | None = None,
        speech_rate: int | None = None,
        loudness_rate: int | None = None,
        emotion: str | None = None,
        emotion_scale: int | None = None,
        pitch: int | None = None,
    ) -> DoubaoTTSResult:
        owns_connection = self._websocket is None
        if owns_connection:
            await self.open()
        audio = bytearray()
        usage: dict[str, Any] | None = None
        log_id = self._log_id
        try:
            async for event in self.stream(
                text,
                context_texts=context_texts,
                speech_rate=speech_rate,
                loudness_rate=loudness_rate,
                emotion=emotion,
                emotion_scale=emotion_scale,
                pitch=pitch,
            ):
                if event.audio:
                    audio.extend(event.audio)
                if event.finished:
                    usage = event.usage
                    log_id = event.log_id
            return DoubaoTTSResult(audio=bytes(audio), usage=usage, log_id=log_id)
        finally:
            if owns_connection:
                await self.close()

    async def stream(
        self,
        text: str,
        *,
        context_texts: Sequence[str] | None = None,
        speech_rate: int | None = None,
        loudness_rate: int | None = None,
        emotion: str | None = None,
        emotion_scale: int | None = None,
        pitch: int | None = None,
    ) -> AsyncIterator[DoubaoTTSStreamEvent]:
        async def one_text() -> AsyncIterator[str]:
            yield text

        async for event in self.stream_segments(
            one_text(),
            context_texts=context_texts,
            speech_rate=speech_rate,
            loudness_rate=loudness_rate,
            emotion=emotion,
            emotion_scale=emotion_scale,
            pitch=pitch,
        ):
            yield event

    async def stream_segments(
        self,
        texts: AsyncIterable[str],
        *,
        context_texts: Sequence[str] | None = None,
        speech_rate: int | None = None,
        loudness_rate: int | None = None,
        emotion: str | None = None,
        emotion_scale: int | None = None,
        pitch: int | None = None,
    ) -> AsyncIterator[DoubaoTTSStreamEvent]:
        if self._websocket is None:
            raise DoubaoTTSError("Doubao TTS connection is not open")
        websocket = self._websocket

        send_task: asyncio.Task[None] | None = None
        usage: dict[str, Any] | None = None
        audio_bytes = 0
        session_id = str(uuid.uuid4())
        try:
            normalized_emotion = emotion.strip() if emotion else None
            self._validate_voice_control("speech_rate", speech_rate, -50, 100)
            self._validate_voice_control("loudness_rate", loudness_rate, -50, 100)
            self._validate_voice_control("emotion_scale", emotion_scale, 1, 5)
            self._validate_voice_control("pitch", pitch, -12, 12)
            if emotion_scale is not None and normalized_emotion is None:
                raise ValueError("Doubao TTS emotion_scale requires emotion")

            audio_params: dict[str, Any] = {
                "format": self._audio_format,
                "sample_rate": self._sample_rate,
                "enable_subtitle": self._enable_subtitle,
            }
            if speech_rate is not None:
                audio_params["speech_rate"] = speech_rate
            if loudness_rate is not None:
                audio_params["loudness_rate"] = loudness_rate
            if normalized_emotion is not None:
                audio_params["emotion"] = normalized_emotion
                if emotion_scale is not None:
                    audio_params["emotion_scale"] = emotion_scale

            additions: dict[str, Any] = {
                "disable_markdown_filter": True,
                "disable_emoji_filter": True,
            }
            if self._section_id is not None:
                additions["section_id"] = self._section_id
            active_context_texts = (
                self._context_texts
                if context_texts is None
                else tuple(text.strip() for text in context_texts if text.strip())[:4]
            )
            if active_context_texts:
                # Seed TTS 2.0 currently consumes the first context item only.
                additions["context_texts"] = ["。".join(active_context_texts)]
            if pitch is not None:
                additions["post_process"] = {"pitch": pitch}

            base_request = {
                "req_params": {
                    "speaker": self._speaker,
                    "audio_params": audio_params,
                    "additions": json.dumps(additions, ensure_ascii=False),
                }
            }
            start_request = {**base_request, "event": int(EventType.START_SESSION)}
            await self._send(
                websocket,
                client_event(
                    EventType.START_SESSION,
                    session_id=session_id,
                    payload=json.dumps(start_request, ensure_ascii=False).encode("utf-8"),
                ),
            )
            await self._wait_for_event(websocket, EventType.SESSION_STARTED)

            send_task = asyncio.create_task(
                self._stream_texts(websocket, session_id, texts)
            )
            while True:
                message = await self._receive(websocket)
                if message.type == MsgType.AUDIO_ONLY_SERVER:
                    audio_bytes += len(message.payload)
                    if audio_bytes > self._max_audio_bytes:
                        raise DoubaoTTSError("Doubao TTS audio exceeded the configured size limit")
                    if message.payload:
                        yield DoubaoTTSStreamEvent(audio=message.payload, log_id=self._log_id)
                    continue
                if message.type != MsgType.FULL_SERVER_RESPONSE:
                    raise DoubaoTTSError(f"unexpected Doubao TTS message type: {message.type.name}")
                payload = message.json_payload()
                if message.event == EventType.TTS_SUBTITLE and payload is not None:
                    yield DoubaoTTSStreamEvent(
                        subtitle=payload,
                        log_id=self._log_id,
                    )
                    continue
                reported_usage = payload.get("usage") if payload else None
                if isinstance(reported_usage, dict):
                    usage = {str(key): value for key, value in reported_usage.items()}
                if message.event == EventType.SESSION_FINISHED:
                    break

            await send_task
            send_task = None
            if audio_bytes == 0:
                raise DoubaoTTSError("Doubao TTS returned no audio data")
            yield DoubaoTTSStreamEvent(
                finished=True,
                usage=usage,
                log_id=self._log_id,
            )
        except asyncio.CancelledError:
            with suppress(Exception):
                await self._send(
                    websocket,
                    client_event(EventType.CANCEL_SESSION, session_id=session_id),
                )
            raise
        except DoubaoTTSError:
            raise
        except Exception as exc:
            raise DoubaoTTSError(f"Doubao TTS synthesis failed: {exc}") from exc
        finally:
            if send_task is not None:
                if not send_task.done():
                    send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    # The primary stream exception has already been surfaced. Retrieving the
                    # sender failure here prevents an orphaned task warning.
                    pass

    @staticmethod
    def _validate_voice_control(
        name: str,
        value: int | None,
        minimum: int,
        maximum: int,
    ) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Doubao TTS {name} must be an integer")
        if not minimum <= value <= maximum:
            raise ValueError(
                f"Doubao TTS {name} must be between {minimum} and {maximum}"
            )

    async def _stream_texts(
        self,
        websocket: Any,
        session_id: str,
        texts: AsyncIterable[str],
    ) -> None:
        total_chars = 0
        sent_text = False
        async for raw_text in texts:
            text = raw_text.strip()
            if not text:
                continue
            total_chars += len(text)
            if total_chars > 4_000:
                raise DoubaoTTSError("Doubao TTS session exceeds 4000 characters")
            sent_text = True
            for offset in range(0, len(text), 24):
                fragment = text[offset : offset + 24]
                request = {
                    "event": int(EventType.TASK_REQUEST),
                    "req_params": {"text": fragment},
                }
                await self._send(
                    websocket,
                    client_event(
                        EventType.TASK_REQUEST,
                        session_id=session_id,
                        payload=json.dumps(request, ensure_ascii=False).encode("utf-8"),
                    ),
                )
                if self._send_interval:
                    await asyncio.sleep(self._send_interval)
        if not sent_text:
            raise DoubaoTTSError("Doubao TTS text must not be empty")
        await self._send(
            websocket,
            client_event(EventType.FINISH_SESSION, session_id=session_id),
        )

    async def _send(self, websocket: Any, message: Message) -> None:
        await websocket.send(message.marshal())

    async def _receive(self, websocket: Any) -> Message:
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=self._timeout)
        except TimeoutError as exc:
            raise DoubaoTTSError("timed out waiting for Doubao TTS") from exc
        if isinstance(raw, str) or not raw:
            raise DoubaoTTSError("Doubao TTS returned an invalid WebSocket frame")
        message = Message.unmarshal(raw)
        if message.type == MsgType.ERROR:
            error_detail = message.payload.decode("utf-8", errors="replace")
            raise DoubaoTTSError(
                f"Doubao TTS error {message.error_code}: {error_detail}"
            )
        if message.event in {EventType.CONNECTION_FAILED, EventType.SESSION_FAILED}:
            failure_event = EventType(message.event)
            failure_detail = message.json_payload() or message.payload.decode(
                "utf-8",
                errors="replace",
            )
            raise DoubaoTTSError(
                f"Doubao TTS {failure_event.name}: {failure_detail}"
            )
        return message

    async def _wait_for_event(self, websocket: Any, event: EventType) -> Message:
        while True:
            message = await self._receive(websocket)
            if message.type == MsgType.FULL_SERVER_RESPONSE and message.event == event:
                return message
