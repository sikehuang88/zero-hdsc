"""Streaming presentation tags and per-session TTS expression routing."""

# Chinese punctuation is intentional in the streamed speech fixtures.
# ruff: noqa: RUF001

from __future__ import annotations

import json
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

import pytest

from ssa.adapters.doubao_tts import DoubaoSeedTTSAdapter
from ssa.adapters.doubao_tts_protocol import EventType, Message, MsgType
from ssa.interfaces.web_gateway import (
    PresentationTagStripper,
    SpeechSegmenter,
    VoiceExpressionCue,
    _voice_context_texts,
    _voice_style_schedule,
)


def test_presentation_tags_are_stripped_across_stream_chunks() -> None:
    stripper = PresentationTagStripper()
    clean_parts: list[str] = []
    face_cues: list[Any] = []
    voice_cues: list[VoiceExpressionCue] = []

    for delta in (
        "先听我说[fa",
        "ce:smile_soft|0.8]，我真的[voi",
        "ce:tender_ache|0.72|0.88]很想你。",
    ):
        clean, faces, voices = stripper.feed(delta)
        clean_parts.append(clean)
        face_cues.extend(faces)
        voice_cues.extend(voices)

    assert "".join(clean_parts) == "先听我说，我真的很想你。"
    assert [(cue.name, cue.intensity, cue.at_char) for cue in face_cues] == [("smile_soft", 0.8, 4)]
    assert [(cue.recipe, cue.intensity, cue.rate, cue.at_char) for cue in voice_cues] == [
        ("tender_ache", 0.72, 0.88, 8)
    ]


def test_speech_segmenter_flushes_old_style_before_voice_transition() -> None:
    segmenter = SpeechSegmenter()
    warm = VoiceExpressionCue(
        recipe="warm_pride",
        intensity=0.68,
        rate=0.98,
        at_char=0,
    )
    tender = VoiceExpressionCue(
        recipe="tender_ache",
        intensity=0.72,
        rate=0.88,
        at_char=5,
    )

    assert segmenter.feed("我很想你", voice_cues=(warm,)) == []
    segments = segmenter.feed(
        "，可是我会等。真的。",
        voice_cues=(tender,),
    )

    assert [segment.text for segment in segments] == [
        "我很想你，",
        "可是我会等。",
        "真的。",
    ]
    assert [segment.style.recipe for segment in segments] == [
        "warm_pride",
        "tender_ache",
        "tender_ache",
    ]
    assert [segment.style.intensity for segment in segments] == [0.68, 0.72, 0.72]
    assert [segment.style.rate for segment in segments] == [0.98, 0.88, 0.88]


def test_emotion_plan_assigns_distinct_acoustic_style_to_each_sentence() -> None:
    styles = _voice_style_schedule(
        {
            "primary": {"name": "hurt", "intensity": 0.84},
            "voice_segments": [
                {
                    "recipe": "hurt_composed",
                    "intensity": 0.72,
                    "rate": 0.82,
                    "breathiness": 0.64,
                    "tremor": 0.18,
                    "pitch_stability": 0.66,
                    "energy": 0.28,
                    "pause_before_ms": 420,
                    "ending": "held",
                    "emphasis": ["真走"],
                },
                {
                    "recipe": "tender_ache",
                    "intensity": 0.84,
                    "rate": 0.88,
                    "breathiness": 0.48,
                    "tremor": 0.58,
                    "pitch_stability": 0.36,
                    "energy": 0.48,
                    "pause_before_ms": 240,
                    "ending": "voice_break",
                    "emphasis": ["舍得"],
                },
            ]
        }
    )
    segmenter = SpeechSegmenter()
    segmenter.set_style_schedule(styles)

    segments = segmenter.feed("你真的要走吗？你怎么舍得。")

    assert [segment.style.recipe for segment in segments] == [
        "hurt_composed",
        "tender_ache",
    ]
    assert segments[0].style.pause_before_ms == 420
    assert segments[1].style.ending == "voice_break"
    assert segments[1].style.emphasis == ("舍得",)


def test_non_vulnerable_emotion_uses_venomous_sister_voice() -> None:
    styles = _voice_style_schedule(
        {
            "primary": {"name": "attachment", "intensity": 0.6},
            "voice_segments": [
                {
                    "recipe": "shy_longing",
                    "intensity": 0.7,
                    "rate": 0.9,
                }
            ],
        }
    )

    assert styles[0].recipe == "venomous_sister"


def test_genuine_hurt_preserves_vulnerable_voice_layer() -> None:
    styles = _voice_style_schedule(
        {
            "primary": {"name": "hurt", "intensity": 0.8},
            "voice_segments": [
                {
                    "recipe": "hurt_composed",
                    "intensity": 0.8,
                    "rate": 0.86,
                }
            ],
        }
    )

    assert styles[0].recipe == "hurt_composed"


def test_voice_context_exposes_micro_acoustics_and_semantic_emphasis() -> None:
    style = _voice_style_schedule(
        {
            "voice_segments": [
                {
                    "recipe": "tender_ache",
                    "intensity": 0.86,
                    "rate": 0.84,
                    "breathiness": 0.70,
                    "tremor": 0.62,
                    "pitch_stability": 0.34,
                    "energy": 0.48,
                    "pause_before_ms": 260,
                    "ending": "voice_break",
                    "emphasis": ["真走", "别丢下我"],
                }
            ]
        }
    )[0]

    context = " ".join(_voice_context_texts(style, ()))

    assert "不规则颤动和短暂破音" in context
    assert "音高稳定度约 0.34" in context
    assert "句前停顿目标约 260 毫秒" in context
    assert "真走、别丢下我" in context


class _RecordingDoubaoAdapter(DoubaoSeedTTSAdapter):
    def __init__(self) -> None:
        super().__init__(
            api_key="test-api-key",
            send_interval_ms=0,
            context_texts=("默认近距离自然交谈",),
        )
        self._websocket = object()
        self.sent_messages: list[Message] = []
        self._responses = deque(
            (
                Message(type=MsgType.AUDIO_ONLY_SERVER, payload=b"first-audio"),
                Message(
                    type=MsgType.FULL_SERVER_RESPONSE,
                    event=EventType.SESSION_FINISHED,
                    payload=b"{}",
                ),
                Message(type=MsgType.AUDIO_ONLY_SERVER, payload=b"second-audio"),
                Message(
                    type=MsgType.FULL_SERVER_RESPONSE,
                    event=EventType.SESSION_FINISHED,
                    payload=b"{}",
                ),
            )
        )

    async def _send(self, websocket: Any, message: Message) -> None:
        self.sent_messages.append(message)

    async def _wait_for_event(self, websocket: Any, event: EventType) -> Message:
        assert event == EventType.SESSION_STARTED
        return Message(type=MsgType.FULL_SERVER_RESPONSE, event=event)

    async def _receive(self, websocket: Any) -> Message:
        return self._responses.popleft()


async def _texts(text: str) -> AsyncIterator[str]:
    yield text


@pytest.mark.asyncio
async def test_stream_segments_overrides_context_per_session_without_mutating_default() -> None:
    adapter = _RecordingDoubaoAdapter()

    first_events = [
        event
        async for event in adapter.stream_segments(
            _texts("第一段"),
            context_texts=("温柔、克制、隐约难过",),
        )
    ]
    second_events = [event async for event in adapter.stream_segments(_texts("第二段"))]

    start_messages = [
        message for message in adapter.sent_messages if message.event == EventType.START_SESSION
    ]
    start_payloads = [json.loads(message.payload) for message in start_messages]

    assert [event.audio for event in first_events if event.audio] == [b"first-audio"]
    assert [event.audio for event in second_events if event.audio] == [b"second-audio"]
    assert sum(event.finished for event in first_events) == 1
    assert sum(event.finished for event in second_events) == 1
    assert len(start_messages) == 2
    assert start_messages[0].session_id != start_messages[1].session_id
    assert start_payloads[0]["req_params"]["context_texts"] == ["温柔、克制、隐约难过"]
    assert start_payloads[1]["req_params"]["context_texts"] == ["默认近距离自然交谈"]
