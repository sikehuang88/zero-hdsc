"""Low-latency layered emotion planning for interactive and spoken turns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ssa.clock import Clock
from ssa.domain.appraisal import AppraisalResult
from ssa.domain.emotion_frame import (
    EmotionComponent,
    EmotionFrame,
    VoicePerformanceSegment,
)
from ssa.domain.emotional_memory import EmotionalMemory
from ssa.domain.events import Event
from ssa.domain.relationship import RelationshipState
from ssa.domain.state import OrganismState
from ssa.domain.traces import ActivatedTrace


@dataclass(frozen=True)
class RealtimeEmotionPlan:
    appraisal: AppraisalResult
    frame: EmotionFrame


@dataclass(frozen=True)
class _EmotionProfile:
    valence: float
    arousal: float
    cues: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class _VoiceSpec:
    recipe: str
    rate: float
    breathiness: float
    tremor: float
    pitch_stability: float
    energy: float
    pause: int
    ending: str


_PROFILES: Final[dict[str, _EmotionProfile]] = {
    "fear_of_loss": _EmotionProfile(
        -0.88,
        0.90,
        (
            ("分手", 0.75),
            ("离开", 0.58),
            ("丢下", 0.72),
            ("不要我", 0.78),
            ("真走", 0.66),
            ("滚吧", 0.48),
            ("舍得", 0.42),
            ("甩掉", 0.55),
            ("不回来", 0.68),
            ("leave me", 0.72),
            ("break up", 0.78),
        ),
    ),
    "hurt": _EmotionProfile(
        -0.78,
        0.66,
        (
            ("背叛", 0.82),
            ("伤害", 0.68),
            ("心疼", 0.42),
            ("疼", 0.36),
            ("骗我", 0.66),
            ("辜负", 0.72),
            ("捅", 0.48),
            ("委屈", 0.62),
            ("betray", 0.82),
            ("hurt", 0.60),
        ),
    ),
    "sadness": _EmotionProfile(
        -0.72,
        0.56,
        (
            ("哭", 0.52),
            ("眼泪", 0.48),
            ("哽咽", 0.70),
            ("抽泣", 0.72),
            ("撕心裂肺", 0.94),
            ("失声", 0.68),
            ("叹气", 0.34),
            ("难过", 0.58),
            ("sad", 0.58),
            ("cry", 0.54),
        ),
    ),
    "anger": _EmotionProfile(
        -0.66,
        0.92,
        (
            ("生气", 0.60),
            ("气死", 0.66),
            ("凭什么", 0.58),
            ("讨厌", 0.48),
            ("滚", 0.36),
            ("吼", 0.62),
            ("喊", 0.42),
            ("烂", 0.38),
            ("angry", 0.65),
            ("hate", 0.50),
        ),
    ),
    "attachment": _EmotionProfile(
        0.64,
        0.52,
        (
            ("爱你", 0.72),
            ("想你", 0.62),
            ("陪我", 0.54),
            ("抱", 0.36),
            ("回来", 0.46),
            ("清雪", 0.28),
            ("老婆", 0.30),
            ("老公", 0.30),
            ("别走", 0.58),
            ("love", 0.68),
            ("miss you", 0.62),
        ),
    ),
    "concern": _EmotionProfile(
        -0.30,
        0.67,
        (
            ("担心", 0.62),
            ("害怕", 0.62),
            ("焦虑", 0.66),
            ("不舒服", 0.48),
            ("累", 0.30),
            ("加班", 0.30),
            ("睡", 0.22),
            ("worried", 0.62),
            ("afraid", 0.62),
        ),
    ),
    "relief": _EmotionProfile(
        0.58,
        0.42,
        (
            ("回来了", 0.72),
            ("还在", 0.52),
            ("终于", 0.48),
            ("没走", 0.58),
            ("放心", 0.44),
            ("松口气", 0.68),
            ("relief", 0.62),
        ),
    ),
    "longing": _EmotionProfile(
        0.14,
        0.44,
        (
            ("想见", 0.62),
            ("等你", 0.52),
            ("想抱", 0.56),
            ("想念", 0.68),
            ("盼", 0.42),
            ("longing", 0.64),
        ),
    ),
    "playfulness": _EmotionProfile(
        0.52,
        0.70,
        (
            ("逗", 0.50),
            ("开玩笑", 0.62),
            ("调皮", 0.64),
            ("嘴贫", 0.44),
            ("测试", 0.28),
            ("演技", 0.36),
            ("哈哈", 0.46),
            ("tease", 0.58),
        ),
    ),
    "pride": _EmotionProfile(
        0.62,
        0.48,
        (
            ("骄傲", 0.70),
            ("厉害", 0.48),
            ("做到了", 0.62),
            ("真棒", 0.56),
            ("优秀", 0.54),
            ("proud", 0.66),
        ),
    ),
    "shame": _EmotionProfile(
        -0.48,
        0.54,
        (
            ("丢脸", 0.62),
            ("不好意思", 0.54),
            ("羞", 0.46),
            ("难为情", 0.62),
            ("ashamed", 0.64),
        ),
    ),
    "calm": _EmotionProfile(0.06, 0.18, ()),
}

_MEMORY_EMOTION_MAP: Final[dict[str, tuple[str, ...]]] = {
    "attachment": ("attachment", "longing"),
    "concern": ("concern", "fear_of_loss"),
    "frustration": ("hurt", "anger"),
    "anticipation": ("longing",),
    "contentment": ("relief", "attachment"),
    "curiosity": ("playfulness",),
    "ambivalence": ("hurt", "attachment"),
}

_MASK_CUES: Final[tuple[str, ...]] = (
    "没事",
    "算了",
    "随便",
    "舍得",
    "平静",
    "冷静",
    "克制",
    "小声",
    "悄悄",
    "不说",
    "啥也别说",
    "装作",
)
_EXPRESSIVE_CUES: Final[tuple[str, ...]] = (
    "撕心裂肺",
    "大哭",
    "吼",
    "喊",
    "爆发",
    "失控",
)

_ACTION_TENDENCIES: Final[dict[str, str]] = {
    "fear_of_loss": "seek reassurance while preparing for separation",
    "hurt": "protect vulnerability and ask for recognition",
    "sadness": "release pressure and seek quiet closeness",
    "anger": "confront the rupture and demand a response",
    "attachment": "preserve closeness",
    "concern": "check safety and stay attentive",
    "relief": "restore contact and settle",
    "longing": "move closer and sustain connection",
    "playfulness": "invite reciprocal teasing",
    "pride": "affirm and celebrate",
    "shame": "hide exposure while testing acceptance",
    "calm": "remain present",
}

_CAUSE_SUMMARIES: Final[dict[str, str]] = {
    "fear_of_loss": "a possible rupture or withdrawal in the close relationship",
    "hurt": "a relational wound or lack of recognition",
    "sadness": "an emotionally costly loss, pressure, or requested release",
    "anger": "a boundary, expectation, or bond appears to have been violated",
    "attachment": "the bond and desire for continued closeness became salient",
    "concern": "the owner's wellbeing or emotional safety appears uncertain",
    "relief": "contact or safety appears to have been restored after uncertainty",
    "longing": "desired closeness is currently absent or incomplete",
    "playfulness": "the interaction invites light reciprocal intimacy",
    "pride": "the owner's effort or achievement deserves recognition",
    "shame": "exposure creates uncertainty about acceptance",
    "calm": "the interaction contains no strong threat or opportunity signal",
}

_INNER_CONFLICTS: Final[dict[str, str]] = {
    "fear_of_loss": "reach for reassurance while hiding how frightening the loss feels",
    "hurt": "ask to be understood while protecting the most exposed part of the wound",
    "sadness": "let the feeling move through the voice without turning it into performance",
    "anger": "confront the rupture without using force that would deepen it",
    "attachment": "move closer while preserving enough autonomy to sound sincere",
    "concern": "show urgency without making the owner carry her anxiety",
    "relief": "release held tension while keeping the reunion grounded",
    "longing": "admit desire for closeness without demanding immediate reassurance",
    "playfulness": "tease lightly without hiding the real relational signal",
    "pride": "affirm warmly without sounding ceremonial or exaggerated",
    "shame": "test acceptance while resisting the impulse to disappear behind composure",
    "calm": "stay emotionally available without inventing intensity",
}

_EMOTION_COUPLING: Final[dict[str, tuple[tuple[str, float], ...]]] = {
    "fear_of_loss": (("attachment", 0.44), ("hurt", 0.38)),
    "hurt": (("attachment", 0.24), ("anger", 0.18)),
    "sadness": (("hurt", 0.30), ("attachment", 0.20)),
    "anger": (("hurt", 0.34),),
    "relief": (("attachment", 0.34),),
    "longing": (("attachment", 0.42),),
    "shame": (("concern", 0.20), ("attachment", 0.18)),
}

def _voice_spec(
    recipe: str,
    rate: float,
    breathiness: float,
    tremor: float,
    pitch_stability: float,
    energy: float,
    pause: int,
    ending: str,
) -> _VoiceSpec:
    return _VoiceSpec(
        recipe=recipe,
        rate=rate,
        breathiness=breathiness,
        tremor=tremor,
        pitch_stability=pitch_stability,
        energy=energy,
        pause=pause,
        ending=ending,
    )


_PHASE_VOICE: Final[dict[str, _VoiceSpec]] = {
    "settled_presence": _voice_spec("venomous_sister", 1.02, 0.16, 0.03, 0.94, 0.54, 50, "clipped_fall"),
    "stunned": _voice_spec("hurt_composed", 0.80, 0.48, 0.10, 0.72, 0.24, 520, "suspended"),
    "held_breath": _voice_spec("hurt_composed", 0.82, 0.62, 0.14, 0.66, 0.28, 420, "held"),
    "restrained_hurt": _voice_spec("hurt_composed", 0.86, 0.50, 0.24, 0.58, 0.38, 220, "restrained_fall"),
    "anger_leak": _voice_spec("jealous_soft", 1.03, 0.25, 0.22, 0.54, 0.72, 120, "clipped"),
    "sob_rise": _voice_spec("hurt_composed", 0.84, 0.76, 0.70, 0.26, 0.82, 300, "broken"),
    "voice_break": _voice_spec("tender_ache", 0.82, 0.68, 0.58, 0.34, 0.48, 260, "voice_break"),
    "spent_release": _voice_spec("relieved_tears", 0.86, 0.72, 0.30, 0.48, 0.24, 360, "fading"),
    "soft_reaching": _voice_spec("tender_ache", 0.90, 0.42, 0.18, 0.70, 0.38, 180, "soft_fall"),
    "relief_bloom": _voice_spec("relieved_tears", 0.94, 0.38, 0.16, 0.72, 0.52, 160, "warm_release"),
    "anxious_focus": _voice_spec("anxious_care", 0.88, 0.36, 0.20, 0.64, 0.52, 180, "steady"),
    "longing_soft": _voice_spec("shy_longing", 0.90, 0.44, 0.16, 0.70, 0.40, 200, "lingering"),
    "playful_deflection": _voice_spec("venomous_sister", 1.06, 0.16, 0.04, 0.90, 0.68, 50, "dry_lift"),
    "warm_affirmation": _voice_spec("venomous_sister", 1.00, 0.18, 0.04, 0.92, 0.60, 70, "firm_fall"),
    "shy_opening": _voice_spec("shy_longing", 0.90, 0.48, 0.20, 0.66, 0.34, 240, "soft_linger"),
}


class RealtimeEmotionService:
    """Produce variable, evidence-linked appraisal and voice trajectories without an LLM call."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    def plan(
        self,
        event: Event,
        organism: OrganismState,
        relationship: RelationshipState,
        emotional_memories: list[EmotionalMemory],
        activated_traces: list[ActivatedTrace],
        *,
        baseline: AppraisalResult | None = None,
    ) -> RealtimeEmotionPlan:
        lowered = event.content.casefold()
        scores = dict.fromkeys(_PROFILES, 0.0)
        matched_cues: list[str] = []

        for name, profile in _PROFILES.items():
            for cue, weight in profile.cues:
                count = min(2, lowered.count(cue.casefold()))
                if count:
                    scores[name] += weight * (1.0 + 0.18 * (count - 1))
                    matched_cues.append(cue)

        relationship_base = 0.08 + 0.12 * relationship.closeness
        scores["attachment"] += relationship_base
        scores["concern"] += 0.06 * (1.0 - organism.safety)
        scores["fear_of_loss"] += 0.05 * organism.arousal

        for memory in emotional_memories[:6]:
            mapped = _MEMORY_EMOTION_MAP.get(memory.emotion_type.value, ())
            for name in mapped:
                scores[name] += 0.13 * memory.intensity / max(1, len(mapped))

        for item in activated_traces[:6]:
            if item.trace.valence < -0.25:
                scores["hurt"] += 0.06 * item.score
            elif item.trace.valence > 0.25:
                scores["attachment"] += 0.05 * item.score
            if item.trace.arousal > 0.60:
                scores["concern"] += 0.04 * item.score

        if baseline is not None:
            if baseline.valence_signal < -0.12:
                scores["hurt"] += 0.22 * abs(baseline.valence_signal)
            elif baseline.valence_signal > 0.12:
                scores["attachment"] += 0.18 * baseline.valence_signal
            scores["concern"] += 0.10 * baseline.urgency

        direct_scores = dict(scores)
        if max(direct_scores.values()) < 0.22:
            calm_score = 0.30 + 0.08 * (1.0 - organism.arousal)
            direct_scores["calm"] = calm_score
            scores["calm"] = calm_score
        for source, couplings in _EMOTION_COUPLING.items():
            source_score = direct_scores[source]
            for target, weight in couplings:
                scores[target] += source_score * weight

        scores = {name: _clip(value) for name, value in scores.items()}
        direct_scores = {name: _clip(value) for name, value in direct_scores.items()}
        primary_name, primary_intensity = max(
            direct_scores.items(),
            key=lambda item: item[1],
        )
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        secondary = [
            EmotionComponent(name=name, intensity=intensity)
            for name, intensity in ranked
            if intensity >= 0.22 and name not in {primary_name, "calm"}
        ][:4]

        explicit_mask = any(cue in lowered for cue in _MASK_CUES)
        expressive = any(cue in lowered for cue in _EXPRESSIVE_CUES)
        inhibition = _clip(
            0.26
            + (0.30 if explicit_mask else 0.0)
            + 0.10 * (1.0 - organism.safety)
            - (0.28 if expressive else 0.0)
        )
        mask = self._surface_mask(primary_name, scores, inhibition, explicit_mask)
        cue_certainty = min(0.48, 0.07 * len(set(matched_cues)))
        certainty = _clip(0.38 + cue_certainty + (0.08 if baseline is not None else 0.0))
        valence, arousal = self._affective_coordinates(
            primary_name,
            primary_intensity,
            secondary,
            organism,
            baseline,
        )
        trajectory = self._trajectory(primary_name, {item.name for item in secondary}, inhibition)
        frame_id = f"emotion-frame:{event.correlation_id}"
        emphasis = list(dict.fromkeys(matched_cues))[:4]
        voice_segments = self._voice_segments(
            frame_id,
            trajectory,
            primary_intensity,
            inhibition,
            emphasis,
        )
        trace_ids = [item.trace.id for item in activated_traces[:8]]
        frame = EmotionFrame(
            id=frame_id,
            conversation_id=event.conversation_id,
            correlation_id=event.correlation_id,
            query_event_id=event.id,
            primary=EmotionComponent(name=primary_name, intensity=primary_intensity),
            secondary=secondary,
            surface_mask=mask,
            inhibition=inhibition,
            certainty=certainty,
            valence=valence,
            arousal=arousal,
            cause_summary=self._cause_summary(primary_name, emphasis),
            inner_conflict=_INNER_CONFLICTS[primary_name],
            regulation_strategy=self._regulation_strategy(
                primary_name,
                inhibition,
                mask,
            ),
            action_tendency=_ACTION_TENDENCIES[primary_name],
            trajectory=trajectory,
            voice_segments=voice_segments,
            source_event_ids=[event.id],
            source_trace_ids=trace_ids,
            created_at_ms=self._clock.now_ms(),
        )
        supported_memory_ids = [item.id for item in emotional_memories[:5]]
        relationship_relevance = _clip(
            0.42
            + 0.34 * scores["attachment"]
            + 0.24 * scores["fear_of_loss"]
            + 0.16 * scores["hurt"]
        )
        appraisal = AppraisalResult(
            novelty=_clip(0.30 + 0.055 * len(set(matched_cues))),
            goal_congruence=valence,
            controllability=_clip(0.72 - 0.42 * arousal + 0.16 * inhibition),
            certainty=certainty,
            self_agency=0.24,
            user_agency=0.82 if relationship_relevance >= 0.55 else 0.60,
            external_agency=0.04,
            relationship_relevance=relationship_relevance,
            urgency=_clip(0.72 * arousal + 0.18 * scores["fear_of_loss"]),
            valence_signal=valence,
            arousal_signal=arousal,
            supported_event_ids=[event.id],
            supported_memory_ids=supported_memory_ids,
            explanation=(
                f"layered realtime emotion: primary={primary_name}; "
                f"secondary={','.join(item.name for item in secondary) or 'none'}; "
                f"mask={mask.name if mask is not None else 'none'}; "
                f"trajectory={','.join(trajectory)}"
            ),
        )
        return RealtimeEmotionPlan(appraisal=appraisal, frame=frame)

    @staticmethod
    def _cause_summary(primary: str, emphasis: list[str]) -> str:
        base = _CAUSE_SUMMARIES[primary]
        if not emphasis:
            return base
        return f"{base}; strongest cues: {', '.join(emphasis)}"

    @staticmethod
    def _regulation_strategy(
        primary: str,
        inhibition: float,
        mask: EmotionComponent | None,
    ) -> str:
        if primary == "relief":
            return "release held tension in a small warm bloom, then make the restored contact feel ordinary"
        if primary == "fear_of_loss":
            return "hold the first panic behind a composed opening, then reach for reassurance without demanding it"
        if primary == "sadness":
            return "let the feeling crest openly, then lower the energy so the listener is not trapped inside it"
        if inhibition >= 0.55:
            return "hold the first impulse, reveal the feeling gradually, then make one clear bid"
        if primary in {"sadness", "hurt", "fear_of_loss"} and inhibition < 0.18:
            return "allow an uneven emotional crest, then settle the ending instead of sustaining it"
        if mask is not None:
            return "let the surface mask soften across the reply so the underlying feeling becomes legible"
        return "express the feeling directly with natural variation and a grounded ending"

    @staticmethod
    def _surface_mask(
        primary: str,
        scores: dict[str, float],
        inhibition: float,
        explicit_mask: bool,
    ) -> EmotionComponent | None:
        if scores["playfulness"] >= 0.42 and primary in {"hurt", "fear_of_loss", "anger"}:
            return EmotionComponent(
                name="playful_deflection",
                intensity=_clip(0.38 + 0.34 * scores["playfulness"]),
            )
        if inhibition >= 0.48 or explicit_mask:
            return EmotionComponent(name="composed", intensity=inhibition)
        return None

    @staticmethod
    def _affective_coordinates(
        primary: str,
        primary_intensity: float,
        secondary: list[EmotionComponent],
        organism: OrganismState,
        baseline: AppraisalResult | None,
    ) -> tuple[float, float]:
        components = [EmotionComponent(name=primary, intensity=primary_intensity), *secondary]
        total = sum(item.intensity for item in components) or 1.0
        valence = sum(
            _PROFILES[item.name].valence * item.intensity for item in components
        ) / total
        arousal = sum(
            _PROFILES[item.name].arousal * item.intensity for item in components
        ) / total
        valence = 0.88 * valence + 0.12 * organism.valence
        arousal = 0.90 * arousal + 0.10 * organism.arousal
        if baseline is not None:
            valence = 0.72 * valence + 0.28 * baseline.valence_signal
            arousal = 0.72 * arousal + 0.28 * baseline.arousal_signal
        return _clip_signed(valence), _clip(arousal)

    @staticmethod
    def _trajectory(primary: str, secondary: set[str], inhibition: float) -> list[str]:
        if primary in {"sadness", "hurt"} and inhibition < 0.28:
            return ["held_breath", "sob_rise", "voice_break", "spent_release"]
        if primary in {"fear_of_loss", "hurt"} and "anger" in secondary:
            return ["stunned", "restrained_hurt", "anger_leak", "soft_reaching"]
        if primary == "fear_of_loss":
            return ["stunned", "restrained_hurt", "voice_break", "soft_reaching"]
        if primary in {"sadness", "hurt"}:
            return ["held_breath", "restrained_hurt", "voice_break", "spent_release"]
        if primary == "anger":
            return ["held_breath", "anger_leak", "restrained_hurt"]
        if primary == "concern":
            return ["anxious_focus", "soft_reaching", "settled_presence"]
        if primary == "relief":
            return ["held_breath", "relief_bloom", "soft_reaching"]
        if primary in {"attachment", "longing"}:
            return ["longing_soft", "soft_reaching", "settled_presence"]
        if primary == "playfulness":
            return ["playful_deflection", "settled_presence"]
        if primary == "pride":
            return ["warm_affirmation", "settled_presence"]
        if primary == "shame":
            return ["shy_opening", "soft_reaching"]
        return ["settled_presence"]

    @staticmethod
    def _voice_segments(
        frame_id: str,
        trajectory: list[str],
        primary_intensity: float,
        inhibition: float,
        emphasis: list[str],
    ) -> list[VoicePerformanceSegment]:
        segments: list[VoicePerformanceSegment] = []
        for sequence, phase in enumerate(trajectory):
            spec = _PHASE_VOICE[phase]
            raw_intensity = 0.36 + 0.58 * primary_intensity
            if phase in {"sob_rise", "anger_leak", "voice_break"}:
                raw_intensity += 0.10 * (1.0 - inhibition)
            segments.append(
                VoicePerformanceSegment(
                    id=f"{frame_id}:voice:{sequence}",
                    sequence=sequence,
                    phase=phase,
                    recipe=spec.recipe,
                    intensity=_clip(raw_intensity),
                    rate=spec.rate,
                    breathiness=_clip(spec.breathiness + 0.12 * inhibition),
                    tremor=_clip(spec.tremor + 0.08 * (1.0 - inhibition)),
                    pitch_stability=_clip(spec.pitch_stability + 0.10 * inhibition),
                    energy=_clip(spec.energy * (0.82 + 0.30 * primary_intensity)),
                    pause_before_ms=spec.pause,
                    ending=spec.ending,
                    emphasis=emphasis,
                )
            )
        return segments


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clip_signed(value: float) -> float:
    return max(-1.0, min(1.0, value))


__all__ = ["RealtimeEmotionPlan", "RealtimeEmotionService"]
