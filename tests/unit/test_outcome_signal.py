"""Deterministic tests for language-derived outcome evidence."""

from __future__ import annotations

import pytest

from ssa.services.reflective_learning_service import outcome_signal


@pytest.mark.parametrize(
    ("content", "polarity", "has_confidence"),
    [
        ("这个没有用", -1.0, True),
        ("我不喜欢这样", -1.0, True),
        ("不好意思,我刚才没说清楚", 0.0, False),
        ("这样对不对?", 0.0, False),
        ("我错了,是我记错了", 0.0, False),
        ("我没用过这个功能", 0.0, False),
        ("你这次说得特别准,帮我省了很多时间", 0.0, False),
        ("这个判断完全偏离事实", 0.0, False),
    ],
)
def test_known_polarity_failures_abstain_or_flip(
    content: str,
    polarity: float,
    has_confidence: bool,
) -> None:
    signal = outcome_signal(content)

    assert signal.polarity == polarity
    assert (signal.confidence > 0.0) is has_confidence


@pytest.mark.parametrize(
    ("content", "polarity", "confidence"),
    [
        ("谢谢", 1.0, 0.8),
        ("错了", -1.0, 0.8),
        ("喜欢", 1.0, 0.8),
        ("不喜欢", -1.0, 0.6),
        ("有用", 1.0, 0.8),
        ("没有用", -1.0, 0.6),
        ("没什么用", -1.0, 0.8),
        ("我错了", 0.0, 0.0),
        ("你错了", -1.0, 0.8),
        ("谢谢,不过这里不对", 0.0, 0.0),
    ],
)
def test_signal_pipeline(content: str, polarity: float, confidence: float) -> None:
    signal = outcome_signal(content)

    assert signal.polarity == polarity
    assert signal.confidence == confidence


@pytest.mark.parametrize(
    "idiom",
    (
        "不好意思",
        "对不对",
        "不客气",
        "没什么",
        "没关系",
        "好不好",
        "行不行",
        "是不是",
        "没用过",
        "不至于",
    ),
)
def test_masked_idioms_abstain(idiom: str) -> None:
    signal = outcome_signal(idiom)

    assert signal.confidence == 0.0
    assert signal.reason == "no_usable_signal"


def test_multiple_same_direction_cues_raise_confidence() -> None:
    signal = outcome_signal("谢谢,这个方法很好而且有用")

    assert signal.polarity == 1.0
    assert signal.confidence == 0.95
    assert set(signal.matched) == {"谢谢", "很好", "有用"}
