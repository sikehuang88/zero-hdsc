"""Fish Audio protocol and voice-quality policy tests."""

# Chinese punctuation is intentional in the speech fixtures.
# ruff: noqa: RUF001

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import msgpack
import pytest

from ssa.adapters.fish_audio_tts import (
    FishAudioTextSegment,
    FishAudioTTSAdapter,
    FishAudioTTSError,
    fish_audio_emotion_markup,
    list_fish_audio_voices,
)


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.replies = [
            msgpack.packb({"event": "audio", "audio": b"mp3"}, use_bin_type=True),
            msgpack.packb({"event": "finish", "reason": "stop"}, use_bin_type=True),
        ]

    async def send(self, payload: bytes) -> None:
        self.sent.append(payload)

    async def recv(self) -> bytes:
        return self.replies.pop(0)

    async def close(self) -> None:
        return None


async def texts() -> AsyncIterator[str]:
    yield "你好"
    yield "，欢迎回来。"


@pytest.mark.asyncio
async def test_fish_audio_stream_uses_documented_msgpack_events() -> None:
    adapter = FishAudioTTSAdapter(
        api_key="key",
        model="s1",
        reference_id="voice-id",
    )
    websocket = FakeWebSocket()
    adapter._websocket = websocket
    adapter._msgpack = msgpack

    events = [event async for event in adapter.stream_segments(texts(), prosody_speed=0.9)]
    sent = [msgpack.unpackb(payload, raw=False) for payload in websocket.sent]

    assert [payload["event"] for payload in sent] == ["start", "text", "flush", "close"]
    assert sent[0]["request"]["reference_id"] == "voice-id"
    assert sent[0]["request"]["format"] == "mp3"
    assert sent[0]["request"]["chunk_length"] == 300
    assert sent[0]["request"]["min_chunk_length"] == 50
    assert sent[0]["request"]["latency"] == "normal"
    assert sent[0]["request"]["features"] == ["quality-guard"]
    assert sent[0]["request"]["mp3_bitrate"] == 192
    assert sent[0]["request"]["repetition_penalty"] == 1.35
    assert sent[0]["request"]["prosody"]["speed"] == 0.9
    assert events[0].audio == b"mp3"
    assert events[-1].finished is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "expected_prefix"),
    [
        ("s1", "你好，欢迎回来。"),
        (
            "s2-pro",
            "[sad naturally]",
        ),
    ],
)
async def test_fish_audio_emotion_markup_uses_model_specific_official_syntax(
    model: str,
    expected_prefix: str,
) -> None:
    adapter = FishAudioTTSAdapter(
        api_key="key",
        model=model,
        reference_id="voice-id",
    )
    websocket = FakeWebSocket()
    adapter._websocket = websocket
    adapter._msgpack = msgpack
    markup = fish_audio_emotion_markup(
        recipe="tender_ache",
        intensity=0.86,
        breathiness=0.68,
        tremor=0.24,
        energy=0.28,
        pause_before_ms=360,
        ending="soft_fall",
        has_emphasis=True,
    )

    _ = [event async for event in adapter.stream_segments(texts(), emotion_markup=markup)]
    sent = [msgpack.unpackb(payload, raw=False) for payload in websocket.sent]
    assert sent[1]["text"].startswith(expected_prefix)
    assert sent[1]["text"].count(expected_prefix) == 1


@pytest.mark.asyncio
async def test_fish_audio_omits_neutral_prosody_processing() -> None:
    adapter = FishAudioTTSAdapter(
        api_key="key",
        model="s1",
        reference_id="voice-id",
    )
    websocket = FakeWebSocket()
    adapter._websocket = websocket
    adapter._msgpack = msgpack

    _ = [event async for event in adapter.stream_segments(texts())]
    sent = [msgpack.unpackb(payload, raw=False) for payload in websocket.sent]

    assert "prosody" not in sent[0]["request"]


def test_fish_audio_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match=r"s2\.1-pro-free"):
        FishAudioTTSAdapter(
            api_key="key",
            model="unknown-model",
            reference_id="voice-id",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["s2.1-pro", "s2.1-pro-free"])
async def test_fish_audio_s21_uses_http_tts_endpoint(monkeypatch, model: str) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self, limit: int) -> bytes:
            captured["limit"] = limit
            return b"complete-mp3"

    def fake_urlopen(request, timeout: float):
        captured["url"] = request.full_url
        captured["model"] = request.get_header("Model")
        captured["content_type"] = request.get_header("Content-type")
        captured["payload"] = msgpack.unpackb(request.data, raw=False)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("ssa.adapters.fish_audio_tts.urlopen", fake_urlopen)
    adapter = FishAudioTTSAdapter(
        api_key="key",
        websocket_url="wss://api.fish.audio/v1/tts/live",
        model=model,
        reference_id="voice-id",
    )

    events = [event async for event in adapter.stream_segments(texts())]

    assert captured["url"] == "https://api.fish.audio/v1/tts"
    assert captured["model"] == model
    assert captured["content_type"] == "application/msgpack"
    assert captured["payload"]["text"] == "你好，欢迎回来。"
    assert [event.audio for event in events if event.audio] == [b"complete-mp3"]
    assert events[-1].finished is True


@pytest.mark.asyncio
async def test_fish_audio_keeps_multiple_emotions_in_one_websocket_session() -> None:
    adapter = FishAudioTTSAdapter(
        api_key="key",
        model="s2-pro",
        reference_id="female-voice-id",
    )
    websocket = FakeWebSocket()
    adapter._websocket = websocket
    adapter._msgpack = msgpack

    async def styled_segments() -> AsyncIterator[FishAudioTextSegment]:
        yield FishAudioTextSegment(
            "第一句话先把想念轻轻说出来，让情绪有足够的自然铺垫。",
            fish_audio_emotion_markup(recipe="shy_longing", intensity=0.6),
        )
        yield FishAudioTextSegment(
            "第二句话把克制的难过放在中间，但仍然保持同一个人说话。",
            fish_audio_emotion_markup(recipe="tender_ache", intensity=0.7),
        )
        yield FishAudioTextSegment(
            "第三句话平静落下，不把尾音拖长，也不重复前面的内容。",
            fish_audio_emotion_markup(recipe="calm", intensity=0.5),
        )

    _ = [event async for event in adapter.stream_styled_segments(styled_segments())]
    sent = [msgpack.unpackb(payload, raw=False) for payload in websocket.sent]

    assert [payload["event"] for payload in sent] == [
        "start",
        "text",
        "flush",
        "close",
    ]
    assert sum(payload["event"] == "start" for payload in sent) == 1
    assert sent[1]["text"].startswith("[shy naturally]")
    assert "[sad naturally]" in sent[1]["text"]
    assert "[calm softly]" in sent[1]["text"]


def test_short_reply_collapses_to_one_emotion_condition() -> None:
    adapter = FishAudioTTSAdapter(
        api_key="key",
        model="s2.1-pro-free",
        reference_id="female-voice-id",
    )
    first = fish_audio_emotion_markup(recipe="shy_longing", intensity=0.6)
    second = fish_audio_emotion_markup(recipe="tender_ache", intensity=0.7)

    stabilized = adapter._stabilize_short_reply(
        [
            FishAudioTextSegment("姐在呢。", first),
            FishAudioTextSegment("急什么？", second),
        ]
    )

    assert stabilized == [FishAudioTextSegment("姐在呢。急什么？", first)]


@pytest.mark.asyncio
async def test_s21_retries_suspiciously_long_short_reply_without_markup(monkeypatch) -> None:
    payloads: list[dict[str, object]] = []

    def synthesize(self, payload: dict[str, object]) -> bytes:
        payloads.append(dict(payload))
        return b"x" * (300_000 if len(payloads) == 1 else 90_000)

    monkeypatch.setattr(FishAudioTTSAdapter, "_synthesize_http", synthesize)
    adapter = FishAudioTTSAdapter(
        api_key="key",
        model="s2.1-pro-free",
        reference_id="female-voice-id",
    )

    async def short_segments() -> AsyncIterator[FishAudioTextSegment]:
        yield FishAudioTextSegment(
            "姐在呢。急什么？",
            fish_audio_emotion_markup(recipe="shy_longing", intensity=0.6),
        )

    events = [event async for event in adapter.stream_styled_segments(short_segments())]

    assert len(payloads) == 2
    assert str(payloads[0]["text"]).startswith("[shy naturally]")
    assert payloads[1]["text"] == "姐在呢。急什么？"
    assert payloads[1]["condition_on_previous_chunks"] is False
    assert payloads[1]["repetition_penalty"] == 1.5
    assert [event.audio for event in events if event.audio] == [b"x" * 90_000]


@pytest.mark.asyncio
async def test_s21_drops_audio_when_plain_retry_is_still_implausibly_long(monkeypatch) -> None:
    def synthesize(self, payload: dict[str, object]) -> bytes:
        return b"x" * 300_000

    monkeypatch.setattr(FishAudioTTSAdapter, "_synthesize_http", synthesize)
    adapter = FishAudioTTSAdapter(
        api_key="key",
        model="s2.1-pro-free",
        reference_id="female-voice-id",
    )

    async def short_segments() -> AsyncIterator[FishAudioTextSegment]:
        yield FishAudioTextSegment("姐在呢。急什么？")

    with pytest.raises(FishAudioTTSError, match="implausibly long"):
        _ = [event async for event in adapter.stream_styled_segments(short_segments())]


@pytest.mark.asyncio
async def test_s1_applies_only_one_emotion_tag_to_the_whole_reply() -> None:
    adapter = FishAudioTTSAdapter(
        api_key="key",
        model="s1",
        reference_id="female-voice-id",
    )
    websocket = FakeWebSocket()
    adapter._websocket = websocket
    adapter._msgpack = msgpack

    async def styled_segments() -> AsyncIterator[FishAudioTextSegment]:
        yield FishAudioTextSegment("第一句。", fish_audio_emotion_markup(recipe="hurt_composed", intensity=0.8))
        yield FishAudioTextSegment("第二句。", fish_audio_emotion_markup(recipe="playful_tease", intensity=0.8))
        yield FishAudioTextSegment("第三句。", fish_audio_emotion_markup(recipe="calm", intensity=0.5))

    _ = [event async for event in adapter.stream_styled_segments(styled_segments())]
    sent = [msgpack.unpackb(payload, raw=False) for payload in websocket.sent]

    assert [payload["event"] for payload in sent] == ["start", "text", "flush", "close"]
    assert sent[1]["text"] == "第一句。第二句。第三句。"


@pytest.mark.asyncio
async def test_fish_audio_voice_list_uses_official_model_endpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({
                "total": 1,
                "has_more": False,
                "items": [{
                    "_id": "voice-id",
                    "title": "Zero Voice",
                    "state": "ready",
                    "visibility": "private",
                    "language": ["zh"],
                    "tags": ["female", "warm"],
                }],
            }).encode()

    def fake_urlopen(request, timeout: float):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("ssa.adapters.fish_audio_tts.urlopen", fake_urlopen)
    page = await list_fish_audio_voices(
        api_key="fish-key",
        self_only=True,
        title="Zero",
    )

    assert "/model?" in str(captured["url"])
    assert "self=true" in str(captured["url"])
    assert "title=Zero" in str(captured["url"])
    assert captured["authorization"] == "Bearer fish-key"
    assert page.items[0].id == "voice-id"
    assert page.items[0].languages == ("zh",)
