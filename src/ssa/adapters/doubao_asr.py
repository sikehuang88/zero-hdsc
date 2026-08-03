"""Doubao big-model ASR adapter for PCM utterances."""

from __future__ import annotations

import asyncio
import gzip
import json
import uuid
from dataclasses import dataclass
from typing import Any


DEFAULT_DOUBAO_ASR_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"
DEFAULT_DOUBAO_ASR_RESOURCE_ID = "volc.bigasr.sauc.duration"


class DoubaoASRError(RuntimeError):
    """Authentication, protocol, or recognition failure."""


@dataclass(frozen=True)
class DoubaoASRResult:
    text: str
    log_id: str | None = None
    duration_ms: int | None = None


def _header(message_type: int, flags: int = 0) -> bytearray:
    return bytearray(((1 << 4) | 1, (message_type << 4) | flags, (1 << 4) | 1, 0))


def _frame(message_type: int, payload: bytes, flags: int = 0) -> bytes:
    compressed = gzip.compress(payload)
    message = _header(message_type, flags)
    message.extend(len(compressed).to_bytes(4, "big"))
    message.extend(compressed)
    return bytes(message)


def _parse_response(raw: bytes) -> dict[str, Any]:
    if len(raw) < 8:
        raise DoubaoASRError("Doubao ASR returned a truncated frame")
    header_size = (raw[0] & 0x0F) * 4
    message_type = raw[1] >> 4
    flags = raw[1] & 0x0F
    serialization = raw[2] >> 4
    compression = raw[2] & 0x0F
    payload = raw[header_size:]
    result: dict[str, Any] = {"message_type": message_type, "flags": flags}

    if message_type == 0b1001:
        offset = 0
        if flags in {0b0001, 0b0011}:
            if len(payload) < 4:
                raise DoubaoASRError("Doubao ASR response omitted its sequence")
            result["sequence"] = int.from_bytes(payload[:4], "big", signed=True)
            offset = 4
        if len(payload) < offset + 4:
            raise DoubaoASRError("Doubao ASR response omitted its payload size")
        size = int.from_bytes(payload[offset : offset + 4], "big", signed=False)
        body = payload[offset + 4 : offset + 4 + size]
    elif message_type == 0b1011:
        if len(payload) < 8:
            raise DoubaoASRError("Doubao ASR acknowledgement is truncated")
        result["sequence"] = int.from_bytes(payload[:4], "big", signed=True)
        size = int.from_bytes(payload[4:8], "big", signed=False)
        body = payload[8 : 8 + size]
    elif message_type == 0b1111:
        if len(payload) < 8:
            raise DoubaoASRError("Doubao ASR error frame is truncated")
        code = int.from_bytes(payload[:4], "big", signed=False)
        size = int.from_bytes(payload[4:8], "big", signed=False)
        body = payload[8 : 8 + size]
        detail = body.decode("utf-8", errors="replace")
        raise DoubaoASRError(f"Doubao ASR error {code}: {detail}")
    else:
        raise DoubaoASRError(f"unexpected Doubao ASR message type: {message_type}")

    if compression == 1 and body:
        body = gzip.decompress(body)
    if not body:
        return result
    if serialization == 1:
        decoded = json.loads(body.decode("utf-8"))
        if isinstance(decoded, dict):
            result["payload"] = decoded
    else:
        result["payload"] = body.decode("utf-8", errors="replace")
    return result


class DoubaoASRAdapter:
    def __init__(
        self,
        *,
        app_id: str,
        access_token: str,
        websocket_url: str = DEFAULT_DOUBAO_ASR_URL,
        resource_id: str = DEFAULT_DOUBAO_ASR_RESOURCE_ID,
        sample_rate: int = 16_000,
        timeout_seconds: float = 20.0,
        chunk_ms: int = 100,
    ) -> None:
        if not app_id.strip() or not access_token.strip():
            raise ValueError("Doubao ASR requires APP ID and Access Token")
        if sample_rate <= 0 or timeout_seconds <= 0 or chunk_ms <= 0:
            raise ValueError("Doubao ASR numeric settings must be positive")
        self._app_id = app_id.strip()
        self._access_token = access_token.strip()
        self._websocket_url = websocket_url
        self._resource_id = resource_id
        self._sample_rate = sample_rate
        self._timeout = timeout_seconds
        self._chunk_bytes = sample_rate * 2 * chunk_ms // 1000

    async def transcribe(self, pcm: bytes) -> DoubaoASRResult:
        if len(pcm) < self._sample_rate // 4:
            return DoubaoASRResult(text="")
        if len(pcm) > self._sample_rate * 2 * 120:
            raise DoubaoASRError("Doubao ASR utterance exceeds 120 seconds")

        import websockets

        headers = {
            "X-Api-App-Key": self._app_id,
            "X-Api-Access-Key": self._access_token,
            "X-Api-Resource-Id": self._resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }
        try:
            websocket = await asyncio.wait_for(
                websockets.connect(
                    self._websocket_url,
                    additional_headers=headers,
                    max_size=10 * 1024 * 1024,
                    open_timeout=self._timeout,
                    ping_interval=None,
                    proxy=None,
                ),
                timeout=self._timeout,
            )
        except Exception as exc:
            raise DoubaoASRError(f"Doubao ASR connection failed: {exc}") from exc

        response = getattr(websocket, "response", None)
        response_headers = getattr(response, "headers", None)
        log_id = response_headers.get("x-tt-logid") if response_headers is not None else None
        request = {
            "app": {"appid": self._app_id, "token": self._access_token},
            "user": {"uid": "zero-phone"},
            "request": {
                "reqid": str(uuid.uuid4()),
                "workflow": "audio_in,resample,partition,vad,fe,decode,itn,nlu_punctuate",
                "show_utterances": True,
                "result_type": "single",
                "sequence": 1,
                "end_window_size": 200,
            },
            "audio": {
                "format": "pcm",
                "codec": "pcm",
                "rate": self._sample_rate,
                "sample_rate": self._sample_rate,
                "bits": 16,
                "channel": 1,
            },
        }
        try:
            await websocket.send(
                _frame(0b0001, json.dumps(request, ensure_ascii=False).encode("utf-8"))
            )
            initial = _parse_response(await asyncio.wait_for(websocket.recv(), self._timeout))
            self._raise_for_payload(initial.get("payload"))

            for offset in range(0, len(pcm), self._chunk_bytes):
                chunk = pcm[offset : offset + self._chunk_bytes]
                last = offset + self._chunk_bytes >= len(pcm)
                await websocket.send(_frame(0b0010, chunk, 0b0010 if last else 0))

            latest_text = ""
            duration_ms: int | None = None
            while True:
                response_data = await asyncio.wait_for(websocket.recv(), self._timeout)
                parsed = _parse_response(response_data)
                payload = parsed.get("payload")
                self._raise_for_payload(payload)
                if not isinstance(payload, dict):
                    continue
                audio_info = payload.get("audio_info")
                if isinstance(audio_info, dict) and isinstance(audio_info.get("duration"), int):
                    duration_ms = audio_info["duration"]
                result = payload.get("result")
                if not isinstance(result, dict):
                    continue
                text = result.get("text")
                if isinstance(text, str) and text.strip():
                    latest_text = text.strip()
                utterances = result.get("utterances")
                if isinstance(utterances, list):
                    for utterance in reversed(utterances):
                        if not isinstance(utterance, dict):
                            continue
                        utterance_text = utterance.get("text")
                        if isinstance(utterance_text, str) and utterance_text.strip():
                            latest_text = utterance_text.strip()
                        if utterance.get("definite") is True:
                            return DoubaoASRResult(latest_text, log_id, duration_ms)
                if parsed.get("flags") in {0b0010, 0b0011} or parsed.get("sequence", 0) < 0:
                    return DoubaoASRResult(latest_text, log_id, duration_ms)
        except TimeoutError as exc:
            raise DoubaoASRError("timed out waiting for Doubao ASR") from exc
        finally:
            await websocket.close()

    @staticmethod
    def _raise_for_payload(payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        code = payload.get("code")
        if code in {None, 1000, 1013}:
            return
        detail = payload.get("message") or payload.get("error") or payload
        raise DoubaoASRError(f"Doubao ASR service error {code}: {detail}")
