"""HTTP/SSE gateway exposing the interactive session to browser frontends.

Thin adapter only: all digital-life behavior stays in ``DigitalLifeSession``.
One process holds exactly one session (and therefore one AutonomousRuntime).
The gateway advances that runtime independently; ``/api/session/lifecycle`` only
drains presentation events and returns the latest snapshot.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager, suppress
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, SecretStr

from ssa.adapters.doubao_asr import DoubaoASRAdapter, DoubaoASRError
from ssa.adapters.doubao_tts import (
    DEFAULT_DOUBAO_SPEAKER,
    DoubaoSeedTTSAdapter,
)
from ssa.adapters.fish_audio_tts import (
    FishAudioTextSegment,
    FishAudioTTSAdapter,
    fish_audio_emotion_markup,
    list_fish_audio_voices,
)
from ssa.adapters.frontend_llm import build_frontend_llm_adapter
from ssa.config import DatabaseConfig, Settings, load_settings
from ssa.domain.events import Event
from ssa.interfaces.coding_workspace_api import coding_workspace_router
from ssa.interfaces.community_api import community_router
from ssa.interfaces.relationship_preferences_api import relationship_preferences_router
from ssa.interfaces.romantic_persona_api import romantic_persona_router
from ssa.runtime.interactive import (
    DigitalLifeSession,
    SessionStreamEvent,
    build_interactive_session,
)
from ssa.skill_library import SkillPackageError
from ssa.tools.image_generation import resolve_generated_image
from ssa.tools.models import ImageGenerationRoute
from ssa.tools.soda_music import SodaMusicExecutor

logger = logging.getLogger(__name__)
tts_trace_logger = logging.getLogger("uvicorn.error")

_DEV_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
)


class VoiceRouteRequest(BaseModel):
    """Per-turn voice route supplied by the local frontend registry."""

    adapter: str
    base_url: str
    api_key: SecretStr | None = None
    app_id: str | None = None
    access_token: SecretStr | None = None
    model: str
    speaker: str = DEFAULT_DOUBAO_SPEAKER
    section_id: str | None = None
    context_texts: list[str] = Field(default_factory=list, max_length=4)


class FishVoiceListRequest(BaseModel):
    api_key: SecretStr
    self_only: bool = True
    title: str | None = Field(default=None, max_length=120)
    page_size: int = Field(default=30, ge=1, le=100)
    page_number: int = Field(default=1, ge=1)


class InferenceRouteRequest(BaseModel):
    """Active reasoning route supplied by the local frontend registry."""

    protocol: Literal["openai-chat", "openai-responses", "anthropic-messages"]
    base_url: str
    api_key: SecretStr
    model: str


class SendRequest(BaseModel):
    """Inbound chat payload from the web client."""

    content: str
    voice: VoiceRouteRequest | None = None
    inference: InferenceRouteRequest | None = None
    generation: ImageGenerationRoute | None = None
    interaction_mode: Literal["chat", "coding"] = "chat"
    workspace_root: str | None = Field(default=None, max_length=2_048)
    tools_enabled: bool = True
    skills_enabled: bool = True
    enabled_skill_ids: list[str] | None = Field(default=None, max_length=512)


class SkillStateRequest(BaseModel):
    enabled: bool


def _encode(payload: Any) -> str:
    return json.dumps(jsonable_encoder(payload), ensure_ascii=False)


def _sse(event: str, payload: Any) -> str:
    return f"event: {event}\ndata: {_encode(payload)}\n\n"


_PRESENTATION_TAG_RE = re.compile(
    r"\[(face|voice)\s*[:=]\s*([a-z_][a-z0-9_]*)"
    r"(?:\s*\|\s*([0-9]*\.?[0-9]+))?"
    r"(?:\s*\|\s*([0-9]*\.?[0-9]+))?\s*\]\s*",
    re.IGNORECASE,
)
# Keep the original private symbols available for integrations that imported them.
_FACE_TAG_RE = re.compile(
    r"\[face\s*[:=]\s*([a-z_]+)(?:\s*\|\s*(0(?:\.\d+)?|1(?:\.0+)?|\.\d+))?\s*\]\s*",
    re.IGNORECASE,
)
_FACE_TAG_PREFIX_RE = re.compile(
    r"\[(?:f(?:a(?:c(?:e(?:\s*(?:[:=]\s*[a-z_]*(?:\s*\|\s*[01]?\.?\d*)?)?)?)?)?)?)?",
    re.IGNORECASE,
)
_FACE_TAG_MAX_LEN = 48
_LIFECYCLE_POLL_SECONDS = 2.0
_MAX_BUFFERED_LIFECYCLE_EVENTS = 256


@dataclasses.dataclass(frozen=True)
class VoiceRecipe:
    instruction: str
    default_intensity: float
    default_rate: float


_VOICE_RECIPES: dict[str, VoiceRecipe] = {
    "tender_ache": VoiceRecipe(
        "温柔贴近，声音里有克制的心疼；气息略软，句尾轻收，不把难过演得沉重",
        0.72,
        0.90,
    ),
    "jealous_soft": VoiceRecipe(
        "带一点轻微吃醋和嘴硬，仍然保持亲近；重音落在在意的词上，不尖锐也不讽刺",
        0.66,
        0.98,
    ),
    "hurt_composed": VoiceRecipe(
        "受伤但努力维持平静；音量略低，停顿清楚，句尾克制，避免哭腔和播报腔",
        0.74,
        0.88,
    ),
    "relieved_tears": VoiceRecipe(
        "像终于松了一口气，声音微微发亮又带一点湿润感；先轻后暖，笑意不要夸张",
        0.76,
        0.92,
    ),
    "playful_tease": VoiceRecipe(
        "亲昵调皮，语气有轻巧起伏；关键词稍微上扬，保持自然，不做卡通式表演",
        0.70,
        1.06,
    ),
    "anxious_care": VoiceRecipe(
        "明显担心但专注可靠；语速放慢，重音清晰，短暂停顿后给出安定感",
        0.78,
        0.88,
    ),
    "warm_pride": VoiceRecipe(
        "温柔而真诚地自豪；声音明亮稳定，肯定词有柔和重音，不像正式表扬",
        0.68,
        0.98,
    ),
    "shy_longing": VoiceRecipe(
        "带羞涩和想念，开头略轻，停顿细腻，句尾温柔落下，不故意撒娇",
        0.68,
        0.90,
    ),
    "venomous_sister": VoiceRecipe(
        "冷静贴近、字字带刺，像毒舌姐姐毫不客气地拆穿借口；重音干脆，句尾利落，底层仍有可靠的照顾感",
        0.76,
        1.02,
    ),
    "calm": VoiceRecipe(
        "自然温柔、平稳连贯，像近距离日常交谈；呼吸和停顿真实，不使用播报腔",
        0.55,
        1.00,
    ),
}
_DEFAULT_VOICE_RECIPE = "venomous_sister"
_MIN_VOICE_RATE = 0.72
_MAX_VOICE_RATE = 1.18


class _LifecycleEventBuffer:
    """Bounded, insertion-ordered delivery buffer for proactive UI events."""

    def __init__(self, max_events: int = _MAX_BUFFERED_LIFECYCLE_EVENTS) -> None:
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self._max_events = max_events
        self._events: dict[str, Event] = {}

    def extend(self, events: Sequence[Event]) -> None:
        for event in events:
            self._events[event.id] = event
        while len(self._events) > self._max_events:
            self._events.pop(next(iter(self._events)))

    def drain(self) -> tuple[Event, ...]:
        events = tuple(self._events.values())
        self._events.clear()
        return events


async def _run_lifecycle_pump(
    session: DigitalLifeSession,
    event_buffer: _LifecycleEventBuffer,
    *,
    poll_seconds: float = _LIFECYCLE_POLL_SECONDS,
) -> None:
    """Advance durable lifecycle jobs even when the browser is idle or throttled."""
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    while True:
        if not session.inference_ready:
            await asyncio.sleep(poll_seconds)
            continue
        try:
            event_buffer.extend(await session.poll_lifecycle())
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("background lifecycle tick failed")
        await asyncio.sleep(poll_seconds)


_PRESENTATION_TAG_BODY_RE = re.compile(
    r"\s*(?:[:=]\s*[a-z0-9_]*(?:\s*\|\s*[0-9.]*){0,2}\s*)?",
    re.IGNORECASE,
)
_PRESENTATION_TAG_MAX_LEN = 96


@dataclasses.dataclass(frozen=True)
class FaceExpressionCue:
    name: str
    intensity: float
    at_char: int


@dataclasses.dataclass(frozen=True)
class VoiceExpressionCue:
    recipe: str
    intensity: float
    rate: float
    at_char: int


@dataclasses.dataclass(frozen=True)
class VoiceStyle:
    recipe: str
    intensity: float
    rate: float
    breathiness: float = 0.20
    tremor: float = 0.04
    pitch_stability: float = 0.88
    energy: float = 0.50
    pause_before_ms: int = 0
    ending: str = "natural_fall"
    emphasis: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class SpeechSegment:
    text: str
    style: VoiceStyle | None = None


@dataclasses.dataclass(frozen=True)
class FishVoiceControls:
    speed: float = 1.0
    volume: float = 0.0
    temperature: float = 0.7
    top_p: float = 0.7


@dataclasses.dataclass(frozen=True)
class DoubaoVoiceControls:
    speech_rate: int
    loudness_rate: int


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _is_incomplete_presentation_tag(candidate: str) -> bool:
    if (
        not candidate.startswith("[")
        or "]" in candidate
        or len(candidate) > _PRESENTATION_TAG_MAX_LEN
    ):
        return False
    body = candidate[1:].lower()
    for tag_name in ("face", "voice"):
        if tag_name.startswith(body):
            return True
        if body.startswith(tag_name) and _PRESENTATION_TAG_BODY_RE.fullmatch(
            body[len(tag_name) :]
        ):
            return True
    return False


class PresentationTagStripper:
    """Strip hidden face and voice control tags from streamed model text."""

    def __init__(self) -> None:
        self._buffer = ""
        self._clean_length = 0

    def feed(
        self,
        delta: str,
    ) -> tuple[str, list[FaceExpressionCue], list[VoiceExpressionCue]]:
        self._buffer += delta
        return self._drain(final=False)

    def flush(
        self,
    ) -> tuple[str, list[FaceExpressionCue], list[VoiceExpressionCue]]:
        return self._drain(final=True)

    def _drain(
        self,
        *,
        final: bool,
    ) -> tuple[str, list[FaceExpressionCue], list[VoiceExpressionCue]]:
        face_cues: list[FaceExpressionCue] = []
        voice_cues: list[VoiceExpressionCue] = []
        out: list[str] = []
        buffer = self._buffer
        cursor = 0

        def emit(segment: str) -> None:
            out.append(segment)
            self._clean_length += len(segment)

        while cursor < len(buffer):
            bracket = buffer.find("[", cursor)
            if bracket == -1:
                emit(buffer[cursor:])
                cursor = len(buffer)
                break
            match = _PRESENTATION_TAG_RE.match(buffer, bracket)
            if match is not None:
                emit(buffer[cursor:bracket])
                tag_kind = match.group(1).lower()
                tag_name = match.group(2).lower()
                intensity_text = match.group(3)
                rate_text = match.group(4)
                if tag_kind == "face":
                    intensity = _clamp(
                        float(intensity_text) if intensity_text else 1.0,
                        0.0,
                        1.0,
                    )
                    face_cues.append(
                        FaceExpressionCue(tag_name, intensity, self._clean_length)
                    )
                else:
                    recipe_name = (
                        tag_name if tag_name in _VOICE_RECIPES else _DEFAULT_VOICE_RECIPE
                    )
                    recipe = _VOICE_RECIPES[recipe_name]
                    intensity = _clamp(
                        float(intensity_text)
                        if intensity_text is not None
                        else recipe.default_intensity,
                        0.0,
                        1.0,
                    )
                    rate = _clamp(
                        float(rate_text) if rate_text is not None else recipe.default_rate,
                        _MIN_VOICE_RATE,
                        _MAX_VOICE_RATE,
                    )
                    voice_cues.append(
                        VoiceExpressionCue(
                            recipe=recipe_name,
                            intensity=intensity,
                            rate=rate,
                            at_char=self._clean_length,
                        )
                    )
                cursor = match.end()
                continue
            tail = buffer[bracket:]
            if not final and _is_incomplete_presentation_tag(tail):
                emit(buffer[cursor:bracket])
                self._buffer = tail
                return "".join(out), face_cues, voice_cues
            emit(buffer[cursor : bracket + 1])
            cursor = bracket + 1
        self._buffer = ""
        return "".join(out), face_cues, voice_cues


class FaceTagStripper:
    """Compatibility wrapper for the original face-only parser API."""

    def __init__(self) -> None:
        self._stripper = PresentationTagStripper()

    def feed(self, delta: str) -> tuple[str, list[tuple[str, float, int]]]:
        text, face_cues, _ = self._stripper.feed(delta)
        return text, [
            (cue.name, cue.intensity, cue.at_char) for cue in face_cues
        ]

    def flush(self) -> str:
        text, _, _ = self._stripper.flush()
        return text


_TTS_STRONG_BREAK_CHARS = frozenset("。！？!?；;\n")
_TTS_SOFT_BREAK_CHARS = frozenset("，,、：: ")


class SpeechSegmenter:
    """Split clean model text into bounded segments carrying vocal style."""

    def __init__(self, max_chars: int = 96, min_soft_chars: int = 36) -> None:
        self._buffer = ""
        self._max_chars = max_chars
        self._min_soft_chars = min_soft_chars
        self._style: VoiceStyle | None = None
        self._style_schedule: tuple[VoiceStyle, ...] = ()
        self._segment_index = 0
        self._received_chars = 0

    def set_style_schedule(self, styles: Sequence[VoiceStyle]) -> None:
        if self._buffer or self._segment_index:
            raise RuntimeError("voice style schedule must be set before speech text")
        self._style_schedule = tuple(styles)

    def feed(
        self,
        delta: str,
        voice_cues: Sequence[VoiceExpressionCue] = (),
    ) -> list[SpeechSegment]:
        segments: list[SpeechSegment] = []
        base_char = self._received_chars
        cursor = 0
        for cue in sorted(voice_cues, key=lambda item: item.at_char):
            local_char = cue.at_char - base_char
            if local_char < cursor or local_char > len(delta):
                continue
            segments.extend(self._feed_text(delta[cursor:local_char]))
            segments.extend(
                self._set_style(VoiceStyle(cue.recipe, cue.intensity, cue.rate))
            )
            cursor = local_char
        segments.extend(self._feed_text(delta[cursor:]))
        self._received_chars += len(delta)
        return segments

    def _feed_text(self, delta: str) -> list[SpeechSegment]:
        self._buffer += delta
        segments: list[SpeechSegment] = []
        while self._buffer:
            strong_index = next(
                (
                    index
                    for index, character in enumerate(self._buffer)
                    if character in _TTS_STRONG_BREAK_CHARS
                ),
                -1,
            )
            if 0 <= strong_index < self._max_chars:
                split_at = strong_index + 1
            elif len(self._buffer) >= self._max_chars:
                split_at = self._max_chars
                for index in range(
                    self._max_chars - 1,
                    self._min_soft_chars - 1,
                    -1,
                ):
                    if self._buffer[index] in _TTS_SOFT_BREAK_CHARS:
                        split_at = index + 1
                        break
            else:
                break
            segment = self._buffer[:split_at].strip()
            self._buffer = self._buffer[split_at:]
            if segment:
                segments.append(SpeechSegment(segment, self._next_style()))
                self._segment_index += 1
        return segments

    def _set_style(self, style: VoiceStyle) -> list[SpeechSegment]:
        if style == self._style:
            return []
        segments = self._flush_buffer()
        self._style = style
        return segments

    def _flush_buffer(self) -> list[SpeechSegment]:
        segment = self._buffer.strip()
        self._buffer = ""
        if not segment:
            return []
        spoken = SpeechSegment(segment, self._next_style())
        self._segment_index += 1
        return [spoken]

    def _next_style(self) -> VoiceStyle | None:
        if self._style is not None:
            return self._style
        if not self._style_schedule:
            return None
        return self._style_schedule[min(self._segment_index, len(self._style_schedule) - 1)]

    def flush(self) -> list[SpeechSegment]:
        return self._flush_buffer()


def _voice_context_texts(
    style: VoiceStyle | None,
    fallback: Sequence[str],
) -> tuple[str, ...]:
    if style is None:
        combined = "。".join(text.strip() for text in fallback if text.strip())
        return (combined,) if combined else ()
    recipe = _VOICE_RECIPES.get(style.recipe, _VOICE_RECIPES[_DEFAULT_VOICE_RECIPE])
    if style.intensity < 0.35:
        intensity_instruction = "情绪只轻轻显露"
    elif style.intensity > 0.78:
        intensity_instruction = "情绪清晰可感，但保持真实克制"
    else:
        intensity_instruction = "情绪自然可感，不要刻意表演"
    if style.rate < 0.93:
        rate_instruction = "语速偏慢，保留细小呼吸和句内停顿"
    elif style.rate > 1.07:
        rate_instruction = "语速稍快但咬字完整，句子不要粘连"
    else:
        rate_instruction = "语速接近日常交谈，停顿连贯"
    if style.breathiness >= 0.62:
        breath_instruction = "气息泄露明显，换气细碎但仍可听清"
    elif style.breathiness >= 0.38:
        breath_instruction = "保留柔软可感的呼吸声"
    else:
        breath_instruction = "气息稳定干净"
    if style.tremor >= 0.55:
        tremor_instruction = "声带出现不规则颤动和短暂破音，避免机械等幅抖动"
    elif style.tremor >= 0.22:
        tremor_instruction = "关键词附近轻微发颤"
    else:
        tremor_instruction = "音色保持稳定"
    if style.energy >= 0.70:
        energy_instruction = "能量明显抬升后自然回落"
    elif style.energy <= 0.32:
        energy_instruction = "音量和能量偏低，像刚压住一阵情绪"
    else:
        energy_instruction = "能量保持日常近讲范围"
    emphasis_instruction = (
        f"只在这些词附近形成自然重音：{'、'.join(style.emphasis)}。"
        if style.emphasis
        else "重音跟随句义自然变化，不做固定节拍强调。"
    )
    return (
        f"本段声音表达：{recipe.instruction}；{intensity_instruction}；{rate_instruction}；"
        f"{breath_instruction}；{tremor_instruction}；{energy_instruction}；"
        f"音高稳定度约 {style.pitch_stability:.2f}，句尾方式为 {style.ending}；"
        f"句前停顿目标约 {style.pause_before_ms} 毫秒。{emphasis_instruction}"
        "保持与前后片段完全相同的说话人音色、音量距离和口腔质感，避免重新起调。",
    )


def _doubao_voice_controls(style: VoiceStyle | None) -> DoubaoVoiceControls | None:
    if style is None:
        return None
    return DoubaoVoiceControls(
        speech_rate=round(_clamp((style.rate - 1.0) * 100.0, -28.0, 24.0)),
        loudness_rate=round(_clamp((style.energy - 0.5) * 36.0, -18.0, 16.0)),
    )


def _fish_voice_controls(segments: Sequence[SpeechSegment]) -> FishVoiceControls:
    styled = [segment for segment in segments if segment.style is not None]
    if not styled:
        return FishVoiceControls()
    total_weight = sum(max(1, len(segment.text.strip())) for segment in styled)

    def average(value_of: Callable[[VoiceStyle], float]) -> float:
        return (
            sum(
                value_of(segment.style) * max(1, len(segment.text.strip()))
                for segment in styled
                if segment.style is not None
            )
            / total_weight
        )

    rate = average(lambda style: style.rate)
    energy = average(lambda style: style.energy)
    intensity = average(lambda style: style.intensity)
    tremor = average(lambda style: style.tremor)
    return FishVoiceControls(
        speed=_clamp(rate, 0.92, 1.06),
        volume=_clamp((energy - 0.5) * 5.0, -2.5, 2.0),
        temperature=_clamp(0.56 + intensity * 0.16 + tremor * 0.05, 0.55, 0.76),
        top_p=_clamp(0.62 + energy * 0.16, 0.62, 0.78),
    )


def _voice_style_schedule(data: dict[str, object] | None) -> tuple[VoiceStyle, ...]:
    if data is None:
        return ()
    raw_segments = data.get("voice_segments")
    if not isinstance(raw_segments, list):
        return ()
    primary_value = data.get("primary")
    primary = primary_value.get("name") if isinstance(primary_value, dict) else None
    preserve_vulnerable_voice = primary in {
        "fear_of_loss",
        "hurt",
        "sadness",
        "concern",
    }
    styles: list[VoiceStyle] = []
    for raw in raw_segments[:8]:
        if not isinstance(raw, dict):
            continue
        recipe_value = raw.get("recipe")
        recipe = (
            recipe_value
            if isinstance(recipe_value, str) and recipe_value in _VOICE_RECIPES
            else _DEFAULT_VOICE_RECIPE
        )
        if not preserve_vulnerable_voice:
            recipe = "venomous_sister"
        ending_value = raw.get("ending")
        ending = ending_value if isinstance(ending_value, str) else "natural_fall"
        emphasis_value = raw.get("emphasis")
        emphasis = (
            tuple(item[:32] for item in emphasis_value[:8] if isinstance(item, str) and item)
            if isinstance(emphasis_value, list)
            else ()
        )
        styles.append(
            VoiceStyle(
                recipe=recipe,
                intensity=_clamp(_float_value(raw.get("intensity"), 0.55), 0.0, 1.0),
                rate=_clamp(
                    _float_value(raw.get("rate"), 1.0),
                    _MIN_VOICE_RATE,
                    _MAX_VOICE_RATE,
                ),
                breathiness=_clamp(
                    _float_value(raw.get("breathiness"), 0.20), 0.0, 1.0
                ),
                tremor=_clamp(_float_value(raw.get("tremor"), 0.04), 0.0, 1.0),
                pitch_stability=_clamp(
                    _float_value(raw.get("pitch_stability"), 0.88), 0.0, 1.0
                ),
                energy=_clamp(_float_value(raw.get("energy"), 0.50), 0.0, 1.0),
                pause_before_ms=max(
                    0,
                    min(2_500, int(_float_value(raw.get("pause_before_ms"), 0.0))),
                ),
                ending=ending[:64],
                emphasis=emphasis,
            )
        )
    return tuple(styles)


def _float_value(value: object, fallback: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return float(value)


def _log_tts_trace(event: str, **fields: Any) -> None:
    tts_trace_logger.info(
        "tts_trace %s",
        json.dumps(
            {"event": event, **fields},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def create_app(
    settings: Settings,
    *,
    conversation_id: str = "cli-primary",
    face_tags: bool = True,
) -> FastAPI:
    """Build the ASGI app around a single lazily-initialized session."""
    session: DigitalLifeSession | None = None
    lifecycle_task: asyncio.Task[None] | None = None
    lifecycle_events = _LifecycleEventBuffer()
    soda_music = SodaMusicExecutor()
    started_at = time.time()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        nonlocal session, lifecycle_task
        built = build_interactive_session(
            settings,
            conversation_id=conversation_id,
            face_tags=face_tags,
            frontend_routing=True,
        )
        await built.initialize()
        session = built
        lifecycle_task = asyncio.create_task(
            _run_lifecycle_pump(built, lifecycle_events),
            name=f"web-lifecycle:{conversation_id}",
        )
        logger.info("web gateway ready (conversation=%s)", conversation_id)
        try:
            yield
        finally:
            if lifecycle_task is not None:
                lifecycle_task.cancel()
                with suppress(asyncio.CancelledError):
                    await lifecycle_task
                lifecycle_task = None
            session = None
            built.close()

    app = FastAPI(title="hdsc-web-gateway", lifespan=lifespan)
    app.include_router(coding_workspace_router())
    app.include_router(
        community_router(
            settings.database.path,
            busy_timeout_ms=settings.database.busy_timeout_ms,
        )
    )
    app.include_router(
        relationship_preferences_router(
            settings.database.path,
            busy_timeout_ms=settings.database.busy_timeout_ms,
        )
    )
    app.include_router(
        romantic_persona_router(
            settings.database.path,
            busy_timeout_ms=settings.database.busy_timeout_ms,
        )
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(_DEV_ORIGINS),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def require_session() -> DigitalLifeSession:
        if session is None:
            raise HTTPException(status_code=503, detail="session is still initializing")
        return session

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "ready": session is not None,
            "started_at": started_at,
            "lifecycle_worker": lifecycle_task is not None and not lifecycle_task.done(),
            "lifecycle_poll_seconds": _LIFECYCLE_POLL_SECONDS,
            "inference_ready": session is not None and session.inference_ready,
        }

    @app.get("/api/session/snapshot")
    async def get_snapshot() -> Any:
        return jsonable_encoder(require_session().snapshot())

    @app.get("/api/session/skills")
    async def get_skills() -> Any:
        return {"skills": list(require_session().skill_catalog())}

    @app.post("/api/session/skills/upload")
    async def upload_skill(file: Annotated[UploadFile, File()]) -> Any:
        filename = file.filename or "SKILL.md"
        try:
            payload = await file.read(20 * 1024 * 1024 + 1)
            skill = require_session().install_skill(filename, payload)
        except SkillPackageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            await file.close()
        return {"skill": skill.public_record()}

    @app.patch("/api/session/skills/{skill_id}")
    async def update_skill(skill_id: str, body: SkillStateRequest) -> Any:
        try:
            skill = require_session().set_skill_enabled(skill_id, body.enabled)
        except SkillPackageError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"skill": skill.public_record()}

    @app.delete("/api/session/skills/{skill_id}")
    async def delete_skill(skill_id: str) -> Any:
        try:
            require_session().remove_skill(skill_id)
        except SkillPackageError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"deleted": skill_id}

    @app.get("/api/soda-music/status")
    async def get_soda_music_status() -> Any:
        """Expose the real Soda Music renderer login state to the HUD."""
        try:
            return jsonable_encoder(await soda_music.login_status())
        except Exception as exc:
            logger.warning("Soda Music login status probe failed: %s", exc)
            return {
                "status": "unavailable",
                "logged_in": None,
                "error": str(exc),
            }

    @app.post("/api/session/inference")
    async def configure_inference(route: InferenceRouteRequest) -> Any:
        try:
            adapter, routed_model = build_frontend_llm_adapter(
                protocol=route.protocol,
                base_url=route.base_url,
                api_key=route.api_key.get_secret_value(),
                model=route.model,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        await require_session().configure_llm(adapter, routed_model)
        return {"model": routed_model}

    @app.post("/api/voice/fish/models")
    async def list_fish_voices(body: FishVoiceListRequest) -> Response:
        try:
            page = await list_fish_audio_voices(
                api_key=body.api_key.get_secret_value(),
                self_only=body.self_only,
                title=body.title,
                page_size=body.page_size,
                page_number=body.page_number,
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning("Fish Audio voice sync failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=502)
        return JSONResponse(jsonable_encoder(page))

    @app.get("/api/session/lifecycle")
    async def poll_lifecycle() -> Any:
        active = require_session()
        events = lifecycle_events.drain()
        return jsonable_encoder({"events": events, "snapshot": active.snapshot()})

    @app.get("/api/generated-images/{asset_id}")
    async def get_generated_image(asset_id: str) -> Response:
        resolved = resolve_generated_image(asset_id)
        if resolved is None:
            raise HTTPException(status_code=404, detail="generated image was not found")
        path, mime_type = resolved
        return FileResponse(
            path,
            media_type=mime_type,
            headers={"Cache-Control": "private, max-age=86400"},
        )

    @app.post("/api/asr/transcribe")
    async def transcribe_audio(request: Request) -> Response:
        pcm = await request.body()
        if len(pcm) < 4_000:
            return JSONResponse({"text": "", "duration_ms": 0})
        if len(pcm) > 16_000 * 2 * 120:
            return JSONResponse({"error": "audio exceeds 120 seconds"}, status_code=413)
        app_id = settings.secrets.doubao_app_id
        access_token = settings.secrets.doubao_access_token.get_secret_value()
        if not app_id or not access_token:
            return JSONResponse({"error": "Doubao ASR credentials are missing"}, status_code=503)
        try:
            result = await DoubaoASRAdapter(
                app_id=app_id,
                access_token=access_token,
            ).transcribe(pcm)
        except DoubaoASRError as exc:
            logger.warning("Doubao ASR failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=502)
        return JSONResponse(
            {
                "text": result.text,
                "duration_ms": result.duration_ms,
                "log_id": result.log_id,
            }
        )

    @app.post("/api/session/send")
    async def send(body: SendRequest, request: Request) -> Response:
        content = body.content.strip()
        if not content:
            return JSONResponse({"error": "content must not be empty"}, status_code=400)
        active = require_session()
        llm_override = None
        model_override = None
        if body.inference is not None:
            try:
                llm_override, model_override = build_frontend_llm_adapter(
                    protocol=body.inference.protocol,
                    base_url=body.inference.base_url,
                    api_key=body.inference.api_key.get_secret_value(),
                    model=body.inference.model,
                )
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        voice_route = (
            body.voice
            if body.voice is not None
            and body.voice.adapter in {"doubao-seed-tts", "fish-audio-tts"}
            else None
        )
        # Voice tags must stay hidden even when facial cues are explicitly disabled.
        stripper = PresentationTagStripper() if (face_tags or voice_route is not None) else None
        tts_turn_id = f"tts-{time.time_ns():x}" if voice_route is not None else None
        tts_input: asyncio.Queue[SpeechSegment | None] | None = (
            asyncio.Queue() if voice_route is not None else None
        )
        tts_segmenter = SpeechSegmenter() if tts_input is not None else None
        tts_input_closed = False
        tts_task: asyncio.Task[None] | None = None

        def start_tts() -> None:
            nonlocal tts_task
            if tts_input is not None and tts_task is None:
                tts_task = asyncio.create_task(run_tts())

        def feed_tts(
            text: str,
            voice_cues: Sequence[VoiceExpressionCue] = (),
        ) -> None:
            if tts_input is None or tts_segmenter is None:
                return
            if not text and not voice_cues:
                return
            segments = tts_segmenter.feed(text, voice_cues)
            if text.strip() or segments:
                start_tts()
            for segment in segments:
                tts_input.put_nowait(segment)

        def close_tts_input() -> None:
            nonlocal tts_input_closed
            if tts_input is None or tts_segmenter is None or tts_input_closed:
                return
            final_segments = tts_segmenter.flush()
            if final_segments:
                start_tts()
            for segment in final_segments:
                tts_input.put_nowait(segment)
            tts_input.put_nowait(None)
            tts_input_closed = True

        async def run_tts() -> None:
            assert voice_route is not None
            assert tts_input is not None
            assert tts_turn_id is not None
            await queue.put(("stream", {"kind": "tts_started", "round_index": 0}))
            sequence = 0
            session_count = 0
            total_audio_bytes = 0
            connection_usage: dict[str, Any] | None = None
            aggregate_usage: dict[str, Any] = {}
            _log_tts_trace(
                "turn_started",
                turn_id=tts_turn_id,
                model=voice_route.model,
                speaker=voice_route.speaker,
                section_id=voice_route.section_id,
                default_context_texts=voice_route.context_texts,
            )
            try:
                adapter: FishAudioTTSAdapter | DoubaoSeedTTSAdapter
                if voice_route.adapter == "fish-audio-tts":
                    adapter = FishAudioTTSAdapter(
                        api_key=(
                            voice_route.api_key.get_secret_value()
                            if voice_route.api_key is not None
                            else ""
                        ),
                        websocket_url=voice_route.base_url,
                        model=voice_route.model,
                        reference_id=voice_route.speaker,
                        timeout_seconds=60,
                    )
                else:
                    adapter = DoubaoSeedTTSAdapter(
                        api_key=(
                            voice_route.api_key.get_secret_value()
                            if voice_route.api_key is not None
                            else settings.secrets.doubao_api_key.get_secret_value() or None
                        ),
                        app_id=voice_route.app_id or settings.secrets.doubao_app_id or None,
                        access_token=(
                            voice_route.access_token.get_secret_value()
                            if voice_route.access_token is not None
                            else settings.secrets.doubao_access_token.get_secret_value() or None
                        ),
                        websocket_url=voice_route.base_url,
                        resource_id=voice_route.model,
                        speaker=voice_route.speaker,
                        section_id=voice_route.section_id,
                        enable_subtitle=True,
                        timeout_seconds=60,
                        context_texts=tuple(voice_route.context_texts),
                    )
                if voice_route.adapter == "fish-audio-tts":
                    assert isinstance(adapter, FishAudioTTSAdapter)
                    collected_segments: list[SpeechSegment] = []
                    while True:
                        segment = await tts_input.get()
                        if segment is None:
                            break
                        if segment.text.strip():
                            collected_segments.append(segment)
                    if not collected_segments:
                        raise ValueError("Fish Audio TTS text must not be empty")
                    fish_controls = _fish_voice_controls(collected_segments)
                    session_count = 1
                    session_chunks = 0
                    session_audio_bytes = 0
                    _log_tts_trace(
                        "session_started",
                        turn_id=tts_turn_id,
                        session=session_count,
                        recipe="continuous_fish_voice",
                        model=voice_route.model,
                        speaker=voice_route.speaker,
                        prosody_speed=fish_controls.speed,
                        prosody_volume=fish_controls.volume,
                        temperature=fish_controls.temperature,
                        top_p=fish_controls.top_p,
                        speed_policy="bounded_emotion_average",
                    )
                    _log_tts_trace(
                        "fish_voice_policy",
                        turn_id=tts_turn_id,
                        speed=fish_controls.speed,
                        volume=fish_controls.volume,
                        temperature=fish_controls.temperature,
                        top_p=fish_controls.top_p,
                        policy="bounded_emotion_average",
                        model_pacing="emotion_tags_and_punctuation",
                        segments=len(collected_segments),
                        characters=sum(len(item.text.strip()) for item in collected_segments),
                    )

                    async def fish_segments() -> AsyncIterator[FishAudioTextSegment]:
                        nonlocal sequence
                        for segment in collected_segments:
                            sequence += 1
                            style = segment.style
                            markup = (
                                fish_audio_emotion_markup(
                                    recipe=style.recipe,
                                    intensity=style.intensity,
                                    breathiness=style.breathiness,
                                    tremor=style.tremor,
                                    energy=style.energy,
                                    pause_before_ms=style.pause_before_ms,
                                    ending=style.ending,
                                    has_emphasis=bool(style.emphasis),
                                )
                                if style is not None
                                else None
                            )
                            _log_tts_trace(
                                "fish_segment",
                                turn_id=tts_turn_id,
                                sequence=sequence,
                                recipe=style.recipe if style is not None else "route_default",
                                intensity=style.intensity if style is not None else None,
                                characters=len(segment.text),
                            )
                            yield FishAudioTextSegment(segment.text, markup)

                    async with adapter:
                        chunk_index = 0
                        async for fish_event in adapter.stream_styled_segments(
                            fish_segments(),
                            prosody_speed=fish_controls.speed,
                            prosody_volume=fish_controls.volume,
                            temperature=fish_controls.temperature,
                            top_p=fish_controls.top_p,
                        ):
                            if fish_event.audio:
                                chunk_index += 1
                                session_chunks += 1
                                session_audio_bytes += len(fish_event.audio)
                                total_audio_bytes += len(fish_event.audio)
                                await queue.put(
                                    (
                                        "stream",
                                        {
                                            "kind": "tts_audio",
                                            "round_index": 0,
                                            "sequence": 1,
                                            "chunk_index": chunk_index,
                                            "mime": "audio/mpeg",
                                            "audio_base64": base64.b64encode(
                                                fish_event.audio
                                            ).decode("ascii"),
                                        },
                                    )
                                )
                        if chunk_index:
                            await queue.put(
                                (
                                    "stream",
                                    {
                                        "kind": "tts_segment",
                                        "round_index": 0,
                                        "sequence": 1,
                                        "chunks": chunk_index,
                                        "segments": sequence,
                                        "sessions": 1,
                                        "mime": "audio/mpeg",
                                    },
                                )
                            )
                    _log_tts_trace(
                        "session_finished",
                        turn_id=tts_turn_id,
                        session=1,
                        recipe="continuous_fish_voice",
                        segments=sequence,
                        chunks=session_chunks,
                        audio_bytes=session_audio_bytes,
                    )
                    _log_tts_trace(
                        "turn_finished",
                        turn_id=tts_turn_id,
                        sessions=1,
                        segments=sequence,
                        chunks=session_chunks,
                        audio_bytes=total_audio_bytes,
                        session_usage=None,
                        connection_usage=None,
                    )
                    return
                assert isinstance(adapter, DoubaoSeedTTSAdapter)
                async with adapter:
                    chunk_index = 0
                    pending: SpeechSegment | None = None
                    input_done = False
                    session_segment_count = 0
                    session_chars = 0

                    async def matching_segments(
                        initial: SpeechSegment,
                        style: VoiceStyle | None,
                    ) -> AsyncIterator[str]:
                        nonlocal input_done, pending, sequence
                        nonlocal session_chars, session_segment_count
                        current = initial
                        while True:
                            sequence += 1
                            session_segment_count += 1
                            session_chars += len(current.text)
                            yield current.text
                            next_segment = await tts_input.get()
                            if next_segment is None:
                                input_done = True
                                return
                            if next_segment.style != style:
                                pending = next_segment
                                return
                            current = next_segment

                    while pending is not None or not input_done:
                        first_segment: SpeechSegment | None
                        if pending is not None:
                            first_segment = pending
                            pending = None
                        else:
                            first_segment = await tts_input.get()
                        if first_segment is None:
                            input_done = True
                            break
                        session_count += 1
                        style = first_segment.style
                        session_contexts = _voice_context_texts(
                            style,
                            voice_route.context_texts,
                        )
                        session_segment_count = 0
                        session_chars = 0
                        session_chunks = 0
                        session_audio_bytes = 0
                        session_usage: dict[str, Any] | None = None

                        _log_tts_trace(
                            "session_started",
                            turn_id=tts_turn_id,
                            session=session_count,
                            recipe=(
                                style.recipe if style is not None else "route_default"
                            ),
                            intensity=(style.intensity if style is not None else None),
                            rate=(style.rate if style is not None else None),
                            breathiness=(style.breathiness if style is not None else None),
                            tremor=(style.tremor if style is not None else None),
                            pitch_stability=(
                                style.pitch_stability if style is not None else None
                            ),
                            energy=(style.energy if style is not None else None),
                            pause_before_ms=(
                                style.pause_before_ms if style is not None else None
                            ),
                            ending=(style.ending if style is not None else None),
                            emphasis=(style.emphasis if style is not None else None),
                            context_texts=session_contexts,
                        )

                        stream_kwargs: dict[str, Any] = {"context_texts": session_contexts}
                        doubao_controls = _doubao_voice_controls(style)
                        if doubao_controls is not None:
                            stream_kwargs.update(
                                speech_rate=doubao_controls.speech_rate,
                                loudness_rate=doubao_controls.loudness_rate,
                            )
                            _log_tts_trace(
                                "doubao_voice_controls",
                                turn_id=tts_turn_id,
                                session=session_count,
                                speech_rate=doubao_controls.speech_rate,
                                loudness_rate=doubao_controls.loudness_rate,
                                policy="native_bounded_prosody",
                            )
                        async for doubao_event in adapter.stream_segments(
                            matching_segments(first_segment, style),
                            **stream_kwargs,
                        ):
                            if doubao_event.audio:
                                chunk_index += 1
                                session_chunks += 1
                                session_audio_bytes += len(doubao_event.audio)
                                total_audio_bytes += len(doubao_event.audio)
                                await queue.put(
                                    (
                                        "stream",
                                        {
                                            "kind": "tts_audio",
                                            "round_index": 0,
                                            "sequence": session_count,
                                            "chunk_index": chunk_index,
                                            "mime": "audio/mpeg",
                                            "audio_base64": base64.b64encode(
                                                doubao_event.audio
                                            ).decode("ascii"),
                                        },
                                    )
                                )
                            if doubao_event.subtitle is not None:
                                await queue.put(
                                    (
                                        "stream",
                                        {
                                            "kind": "tts_subtitle",
                                            "round_index": 0,
                                            "sequence": session_count,
                                            "data": doubao_event.subtitle,
                                            "log_id": doubao_event.log_id,
                                        },
                                    )
                                )
                            if doubao_event.finished and doubao_event.usage:
                                session_usage = doubao_event.usage
                                for key, value in doubao_event.usage.items():
                                    current_value = aggregate_usage.get(key)
                                    if (
                                        isinstance(value, (int, float))
                                        and not isinstance(value, bool)
                                        and isinstance(current_value, (int, float))
                                        and not isinstance(current_value, bool)
                                    ):
                                        aggregate_usage[key] = current_value + value
                                    else:
                                        aggregate_usage[key] = value
                        _log_tts_trace(
                            "session_finished",
                            turn_id=tts_turn_id,
                            session=session_count,
                            recipe=(
                                style.recipe if style is not None else "route_default"
                            ),
                            intensity=(style.intensity if style is not None else None),
                            rate=(style.rate if style is not None else None),
                            breathiness=(style.breathiness if style is not None else None),
                            tremor=(style.tremor if style is not None else None),
                            pitch_stability=(
                                style.pitch_stability if style is not None else None
                            ),
                            energy=(style.energy if style is not None else None),
                            emphasis=(style.emphasis if style is not None else None),
                            segments=session_segment_count,
                            characters=session_chars,
                            chunks=session_chunks,
                            audio_bytes=session_audio_bytes,
                            usage=session_usage,
                            log_id=adapter.log_id,
                        )
                    if chunk_index:
                        await queue.put(
                            (
                                "stream",
                                {
                                    "kind": "tts_segment",
                                    "round_index": 0,
                                    "sequence": session_count,
                                    "chunks": chunk_index,
                                    "segments": sequence,
                                    "sessions": session_count,
                                    "mime": "audio/mpeg",
                                    "usage": aggregate_usage or None,
                                    "log_id": adapter.log_id,
                                },
                            )
                        )
                connection_usage = adapter.connection_usage
                _log_tts_trace(
                    "turn_finished",
                    turn_id=tts_turn_id,
                    sessions=session_count,
                    segments=sequence,
                    chunks=chunk_index,
                    audio_bytes=total_audio_bytes,
                    session_usage=aggregate_usage or None,
                    connection_usage=connection_usage,
                )
            except asyncio.CancelledError:
                _log_tts_trace(
                    "turn_cancelled",
                    turn_id=tts_turn_id,
                    sessions=session_count,
                    segments=sequence,
                    audio_bytes=total_audio_bytes,
                )
                raise
            except Exception as exc:
                logger.warning("Voice TTS turn failed: %s", exc)
                _log_tts_trace(
                    "turn_failed",
                    turn_id=tts_turn_id,
                    sessions=session_count,
                    segments=sequence,
                    audio_bytes=total_audio_bytes,
                    error=str(exc),
                )
                await queue.put(
                    (
                        "stream",
                        {
                            "kind": "tts_error",
                            "round_index": 0,
                            "text": str(exc),
                        },
                    )
                )
            finally:
                await queue.put(
                    (
                        "stream",
                        {
                            "kind": "tts_finished",
                            "round_index": 0,
                            "segments": sequence,
                            "sessions": session_count,
                            "usage": connection_usage,
                        },
                    )
                )

        def on_stream(event: SessionStreamEvent) -> None:
            if event.kind == "emotion_plan" and tts_segmenter is not None:
                schedule = _voice_style_schedule(event.data)
                if schedule:
                    tts_segmenter.set_style_schedule(schedule)
                if tts_turn_id is not None:
                    primary = event.data.get("primary") if event.data is not None else None
                    _log_tts_trace(
                        "emotion_plan",
                        turn_id=tts_turn_id,
                        primary=primary,
                        inhibition=(
                            event.data.get("inhibition")
                            if event.data is not None
                            else None
                        ),
                        trajectory=(
                            event.data.get("trajectory")
                            if event.data is not None
                            else None
                        ),
                        voice_segments=len(schedule),
                    )
            if stripper is not None and event.kind == "assistant_delta":
                clean, face_cues, voice_cues = stripper.feed(event.text)
                for face_cue in face_cues:
                    queue.put_nowait(
                        (
                            "stream",
                            {
                                "kind": "expression",
                                "round_index": event.round_index,
                                "name": face_cue.name,
                                "intensity": face_cue.intensity,
                                "at_char": face_cue.at_char,
                            },
                        )
                    )
                for voice_cue in voice_cues:
                    if tts_turn_id is not None:
                        _log_tts_trace(
                            "emotion_cue",
                            turn_id=tts_turn_id,
                            round_index=event.round_index,
                            recipe=voice_cue.recipe,
                            intensity=voice_cue.intensity,
                            rate=voice_cue.rate,
                            at_char=voice_cue.at_char,
                        )
                    queue.put_nowait(
                        (
                            "stream",
                            {
                                "kind": "voice_expression",
                                "round_index": event.round_index,
                                "name": voice_cue.recipe,
                                "intensity": voice_cue.intensity,
                                "rate": voice_cue.rate,
                                "at_char": voice_cue.at_char,
                            },
                        )
                    )
                feed_tts(clean, voice_cues)
                if clean:
                    queue.put_nowait(("stream", dataclasses.replace(event, text=clean)))
                return
            if (
                stripper is not None
                and event.kind == "assistant_done"
                and not event.has_tool_calls
            ):
                tail, face_cues, voice_cues = stripper.flush()
                for face_cue in face_cues:
                    queue.put_nowait(
                        (
                            "stream",
                            {
                                "kind": "expression",
                                "round_index": event.round_index,
                                "name": face_cue.name,
                                "intensity": face_cue.intensity,
                                "at_char": face_cue.at_char,
                            },
                        )
                    )
                for voice_cue in voice_cues:
                    if tts_turn_id is not None:
                        _log_tts_trace(
                            "emotion_cue",
                            turn_id=tts_turn_id,
                            round_index=event.round_index,
                            recipe=voice_cue.recipe,
                            intensity=voice_cue.intensity,
                            rate=voice_cue.rate,
                            at_char=voice_cue.at_char,
                        )
                    queue.put_nowait(
                        (
                            "stream",
                            {
                                "kind": "voice_expression",
                                "round_index": event.round_index,
                                "name": voice_cue.recipe,
                                "intensity": voice_cue.intensity,
                                "rate": voice_cue.rate,
                                "at_char": voice_cue.at_char,
                            },
                        )
                    )
                feed_tts(tail, voice_cues)
                if tail:
                    queue.put_nowait(
                        (
                            "stream",
                            {
                                "kind": "assistant_delta",
                                "round_index": event.round_index,
                                "text": tail,
                            },
                        )
                    )
                close_tts_input()
            elif event.kind == "assistant_delta":
                feed_tts(event.text)
            elif event.kind == "assistant_done" and not event.has_tool_calls:
                close_tts_input()
            queue.put_nowait(("stream", event))

        async def run_turn() -> None:
            try:
                result = await active.send(
                    content,
                    on_stream=on_stream,
                    llm_override=llm_override,
                    model_override=model_override,
                    realtime=voice_route is not None,
                    interaction_mode=body.interaction_mode,
                    workspace_root=body.workspace_root,
                    image_generation_route=body.generation,
                    tools_enabled=body.tools_enabled,
                    skills_enabled=body.skills_enabled,
                    enabled_skill_ids=(
                        tuple(body.enabled_skill_ids)
                        if body.enabled_skill_ids is not None
                        else None
                    ),
                )
                close_tts_input()
                if tts_task is not None:
                    await tts_task
            except asyncio.CancelledError:
                if tts_task is not None and not tts_task.done():
                    tts_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await tts_task
                raise
            except Exception as exc:  # surface LLM/tool failures to the client
                if tts_task is not None and not tts_task.done():
                    tts_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await tts_task
                logger.exception("interactive turn failed")
                await queue.put(("error", str(exc)))
            else:
                await queue.put(("result", {"snapshot": result.snapshot}))

        async def event_stream() -> AsyncIterator[str]:
            task = asyncio.create_task(run_turn())
            try:
                while True:
                    try:
                        kind, payload = await asyncio.wait_for(queue.get(), timeout=1.0)
                    except TimeoutError:
                        # Client went away: cancel the LLM turn instead of burning tokens.
                        if await request.is_disconnected():
                            task.cancel()
                            return
                        continue
                    yield _sse(kind, payload)
                    if kind in {"result", "error"}:
                        return
            finally:
                if not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hdsc-web",
        description="HDSC HTTP/SSE gateway for browser frontends",
    )
    parser.add_argument("--env", default="development", help="configuration environment")
    parser.add_argument("--database", metavar="PATH", help="override the configured SQLite path")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=18787, help="bind port (default: 18787)")
    parser.add_argument(
        "--conversation",
        default="cli-primary",
        metavar="ID",
        help="persistent conversation ID (default: cli-primary)",
    )
    parser.add_argument(
        "--no-face-tags",
        dest="face_tags",
        action="store_false",
        help="disable [face:...] micro-expression annotation prompt and SSE stripping",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the gateway under uvicorn and return a process exit code."""
    args = _parser().parse_args(argv)
    settings = load_settings(str(args.env))
    if args.database is not None:
        settings = settings.model_copy(
            update={
                "database": DatabaseConfig(
                    path=str(args.database),
                    busy_timeout_ms=settings.database.busy_timeout_ms,
                )
            }
        )
    app = create_app(
        settings,
        conversation_id=str(args.conversation),
        face_tags=bool(args.face_tags),
    )

    import uvicorn

    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
