from __future__ import annotations

from pathlib import Path

import pytest

from ssa.services.emotion_expression_library import EmotionExpressionLibrary

LIBRARY_ROOT = Path(__file__).resolve().parents[3] / "emotion-value-library"
pytestmark = pytest.mark.skipif(
    not LIBRARY_ROOT.is_dir(),
    reason=f"authored emotion expression library is absent: {LIBRARY_ROOT}",
)


def test_library_loads_authored_scenes_and_discourse_entries() -> None:
    library = EmotionExpressionLibrary(LIBRARY_ROOT)

    assert library.scene_count == 20
    assert library.entry_count >= 30


def test_recall_selects_impatient_care_for_repeated_self_neglect() -> None:
    recall = EmotionExpressionLibrary(LIBRARY_ROOT).recall(
        "我又熬夜了，也忘记吃饭，现在还在硬撑。"
    )

    forms = {entry.form for entry in recall.entries}
    assert "啧。" in forms
    assert any("吃饭" in situation or "睡觉" in situation for entry in recall.entries for situation in entry.situations)


def test_recall_selects_complex_ambivalence_without_copying_example() -> None:
    recall = EmotionExpressionLibrary(LIBRARY_ROOT).recall(
        "我已经原谅了，可是再听见他的承诺还是不敢相信。"
    )
    context = recall.prompt_context()

    assert any(scene.id == "CE-03" for scene in recall.scenes)
    assert "zero to two discourse particles" in context
    assert "coverage references only" in context
    assert "Never copy, paraphrase, or replay a scene example" in context
    assert "我没有怪你了" not in context


def test_neutral_query_uses_bounded_default_expression_palette() -> None:
    recall = EmotionExpressionLibrary(LIBRARY_ROOT).recall("今天处理一下普通事项")

    assert 1 <= len(recall.entries) <= 5
    assert len(recall.scenes) <= 2
