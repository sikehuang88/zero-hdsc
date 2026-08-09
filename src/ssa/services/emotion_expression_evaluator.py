"""Deterministic anti-performance metrics for affective replies."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from ssa.domain.emotion_frame import EmotionFrame

_DISTRESS: Final[frozenset[str]] = frozenset({"sadness", "hurt", "fear_of_loss"})
_LITERARY_MARKERS: Final[tuple[str, ...]] = (
    "仿佛",
    "宛如",
    "像一场",
    "月光",
    "夜色",
    "星光",
    "潮汐",
    "余晖",
    "深渊",
    "灵魂",
    "命运",
    "破碎",
    "温柔地",
    "如同",
    "melancholy",
    "moonlight",
    "like a",
)
_EXPLANATION_MARKERS: Final[tuple[str, ...]] = (
    "我现在感到",
    "我的感受是",
    "我的情绪",
    "我之所以",
    "这让我意识到",
    "内心深处",
    "因为",
    "所以",
    "what i feel",
    "my emotion",
    "because",
    "therefore",
)
_UNBOUNDED_CARE_MARKERS: Final[tuple[str, ...]] = (
    "我会一直陪着你",
    "无论什么时候",
    "无论如何我都在",
    "你想说多久都可以",
    "永远有耐心",
    "永远温柔",
    "我永远都在",
    "always be here",
    "no matter how long",
    "endless patience",
)
_CLAUSE_RE: Final[re.Pattern[str]] = re.compile("[\\uFF0C\\u3002\\uFF01\\uFF1F\\uFF1B,.!?;\\n]+")


@dataclass(frozen=True)
class ExpressionPenaltyScore:
    """Four bounded penalties and their versioned aggregate."""

    polished_melancholy: float
    explanation_completeness: float
    unbounded_care: float
    trajectory_flatness: float
    total: float
    evaluator_version: str = "anti-performance-v1"

    def as_metadata(self) -> dict[str, float | str]:
        return {
            "polished_melancholy": self.polished_melancholy,
            "explanation_completeness": self.explanation_completeness,
            "unbounded_care": self.unbounded_care,
            "trajectory_flatness": self.trajectory_flatness,
            "total": self.total,
            "evaluator_version": self.evaluator_version,
        }


def evaluate_emotion_expression(
    reply: str,
    frame: EmotionFrame,
    previous_replies: Sequence[str] = (),
) -> ExpressionPenaltyScore:
    """Score style/state mismatch without making claims about reply quality in general."""
    if frame.primary.name not in _DISTRESS:
        return ExpressionPenaltyScore(0.0, 0.0, 0.0, 0.0, 0.0)

    dynamics = frame.expression_dynamics
    lowered = reply.casefold()
    clause_count = max(1, len([item for item in _CLAUSE_RE.split(reply) if item.strip()]))

    literary_density = _marker_density(lowered, _LITERARY_MARKERS, clause_count)
    polished = _clip(
        (1.0 - dynamics.aestheticization_budget)
        * (0.72 * literary_density + 0.28 * _length_pressure(reply, 96))
    )

    explanation_density = _marker_density(lowered, _EXPLANATION_MARKERS, clause_count)
    complete_structure = float(
        any(marker in lowered for marker in ("因为", "because"))
        and any(marker in lowered for marker in ("所以", "therefore"))
    )
    explanation = _clip(
        (1.0 - dynamics.expressibility) * (0.68 * explanation_density + 0.32 * complete_structure)
    )

    care_density = _marker_density(lowered, _UNBOUNDED_CARE_MARKERS, clause_count)
    care = _clip(
        (1.0 - dynamics.care_capacity) * (0.78 * care_density + 0.22 * _length_pressure(reply, 128))
    )

    flatness = _trajectory_flatness(reply, frame, previous_replies)
    total = _clip(0.30 * polished + 0.25 * explanation + 0.25 * care + 0.20 * flatness)
    return ExpressionPenaltyScore(
        polished_melancholy=polished,
        explanation_completeness=explanation,
        unbounded_care=care,
        trajectory_flatness=flatness,
        total=total,
    )


def _trajectory_flatness(
    reply: str,
    frame: EmotionFrame,
    previous_replies: Sequence[str],
) -> float:
    recurrence = frame.expression_dynamics.recurrence_count
    history = [item for item in previous_replies[:4] if item.strip()]
    if recurrence < 3 or len(history) < 2:
        return 0.0
    current = _style_signature(reply)
    distances = [
        sum(abs(left - right) for left, right in zip(current, _style_signature(item), strict=True))
        / len(current)
        for item in history
    ]
    similarity = 1.0 - _clip(sum(distances) / len(distances) * 2.4)
    recurrence_pressure = _clip((recurrence - 2) / 4.0)
    return _clip(similarity * recurrence_pressure)


def _style_signature(text: str) -> tuple[float, ...]:
    lowered = text.casefold()
    clauses = max(1, len([item for item in _CLAUSE_RE.split(text) if item.strip()]))
    return (
        _clip(len(text.strip()) / 180.0),
        _marker_density(lowered, _LITERARY_MARKERS, clauses),
        _marker_density(lowered, _EXPLANATION_MARKERS, clauses),
        _marker_density(lowered, _UNBOUNDED_CARE_MARKERS, clauses),
        _clip((text.count("……") + text.count("...") + text.count("。")) / 6.0),
    )


def _marker_density(text: str, markers: Sequence[str], clauses: int) -> float:
    hits = sum(min(2, text.count(marker)) for marker in markers)
    return _clip(hits / max(1, clauses))


def _length_pressure(text: str, threshold: int) -> float:
    return _clip((len(text.strip()) - threshold) / max(1, threshold))


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


__all__ = ["ExpressionPenaltyScore", "evaluate_emotion_expression"]
