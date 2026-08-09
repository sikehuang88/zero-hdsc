"""Retrieval over Qingxue's authored complex-emotion and discourse lexicon."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCENE_HEADING_RE = re.compile(r"^##\s+(\d{2})｜(.+?)\s*$", re.MULTILINE)
_QUOTE_RE = re.compile(r"^>\s*(.+?)\s*$", re.MULTILINE)
_FIELD_RE = re.compile(r"^-\s+\*\*(.+?)\*\*：\s*(.+?)\s*$", re.MULTILINE)
_ASCII_WORD_RE = re.compile(r"[a-z0-9_+-]{2,}", re.IGNORECASE)
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")


@dataclass(frozen=True)
class ExpressionEntry:
    form: str
    function: str
    situations: tuple[str, ...]
    delivery: str
    avoid: str
    triggers: tuple[str, ...]


@dataclass(frozen=True)
class ComplexEmotionScene:
    id: str
    title: str
    example: str
    fields: dict[str, str]


@dataclass(frozen=True)
class EmotionExpressionRecall:
    rules: tuple[str, ...]
    entries: tuple[ExpressionEntry, ...]
    scenes: tuple[ComplexEmotionScene, ...]

    def prompt_context(self) -> str:
        lines = [
            "<qingxue_emotion_expression_library>",
            "Private expressive guidance, not facts and not text to quote verbatim.",
            "Authored scenes are coverage references only. Never enact one as the current "
            "situation; project the current emotion frame's continuous state axes instead.",
        ]
        lines.extend(f"- rule: {rule}" for rule in self.rules)
        for entry in self.entries:
            lines.append(
                f"- discourse {entry.form} | function={entry.function} | "
                f"use_when={'/'.join(entry.situations)} | delivery={entry.delivery} | "
                f"avoid={entry.avoid}"
            )
        for scene in self.scenes:
            selected = [
                f"{key}={value}"
                for key, value in scene.fields.items()
                if key in {"情绪杂交", "内心状态", "外在表达", "潜台词", "TTS 标签", "情绪轨迹"}
            ]
            lines.append(f"- complex scene {scene.id} / {scene.title}: " + " | ".join(selected))
        lines.extend(
            [
                "Choose zero to two discourse particles only when they reveal a real shift in "
                "stance. Never copy, paraphrase, or replay a scene example, and never "
                "mechanically imitate fillers.",
                "</qingxue_emotion_expression_library>",
            ]
        )
        return "\n".join(lines)


class EmotionExpressionLibrary:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._rules, self._entries = self._load_lexicon()
        self._scenes = self._load_scenes()

    @property
    def scene_count(self) -> int:
        return len(self._scenes)

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def recall(self, query: str, *, entry_limit: int = 5, scene_limit: int = 2) -> EmotionExpressionRecall:
        query_terms = _terms(query)
        ranked_entries = sorted(
            (
                (self._entry_score(entry, query, query_terms), index, entry)
                for index, entry in enumerate(self._entries)
            ),
            key=lambda item: (-item[0], item[1]),
        )
        selected_entries = [entry for score, _, entry in ranked_entries if score > 0][:entry_limit]
        if not selected_entries:
            selected_entries = [
                entry
                for entry in self._entries
                if entry.form in {"嗯。", "嗯？", "嗯……", "啧。", "哦？"}
            ][:entry_limit]
        ranked_scenes = sorted(
            (
                (self._scene_score(scene, query_terms), scene)
                for scene in self._scenes
            ),
            key=lambda item: (-item[0], item[1].id),
        )
        selected_scenes = [scene for score, scene in ranked_scenes if score >= 0.12][
            :scene_limit
        ]
        return EmotionExpressionRecall(
            rules=self._rules,
            entries=tuple(selected_entries),
            scenes=tuple(selected_scenes),
        )

    def _load_lexicon(self) -> tuple[tuple[str, ...], tuple[ExpressionEntry, ...]]:
        path = self.root / "qingxue-expression-lexicon.json"
        if not path.is_file():
            return (), ()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return (), ()
        rules = tuple(str(item).strip() for item in payload.get("rules", []) if str(item).strip())
        entries: list[ExpressionEntry] = []
        raw_entries = payload.get("entries")
        if isinstance(raw_entries, list):
            for raw in raw_entries:
                if not isinstance(raw, dict) or not isinstance(raw.get("form"), str):
                    continue
                entries.append(
                    ExpressionEntry(
                        form=raw["form"].strip(),
                        function=str(raw.get("function") or "").strip(),
                        situations=_strings(raw.get("situations")),
                        delivery=str(raw.get("delivery") or "").strip(),
                        avoid=str(raw.get("avoid") or "").strip(),
                        triggers=_strings(raw.get("triggers")),
                    )
                )
        return rules[:8], tuple(entries)

    def _load_scenes(self) -> tuple[ComplexEmotionScene, ...]:
        path = self.root / "complex-emotions.md"
        if not path.is_file():
            return ()
        text = path.read_text(encoding="utf-8")
        matches = list(_SCENE_HEADING_RE.finditer(text))
        scenes: list[ComplexEmotionScene] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            section = text[match.end() : end]
            quote = _QUOTE_RE.search(section)
            fields = {key.strip(): value.strip() for key, value in _FIELD_RE.findall(section)}
            scenes.append(
                ComplexEmotionScene(
                    id=f"CE-{match.group(1)}",
                    title=match.group(2).strip(),
                    example=quote.group(1).strip() if quote is not None else "",
                    fields=fields,
                )
            )
        return tuple(scenes)

    @staticmethod
    def _entry_score(entry: ExpressionEntry, query: str, query_terms: set[str]) -> float:
        score = sum(1.0 for trigger in entry.triggers if trigger.casefold() in query.casefold())
        entry_terms = _terms(" ".join((entry.function, *entry.situations, *entry.triggers)))
        return score + _overlap(query_terms, entry_terms)

    @staticmethod
    def _scene_score(scene: ComplexEmotionScene, query_terms: set[str]) -> float:
        searchable = " ".join((scene.title, scene.example, *scene.fields.values()))
        return _overlap(query_terms, _terms(searchable))


def resolve_emotion_expression_root(path: str = "emotion-value-library") -> Path:
    configured = Path(path).expanduser()
    if configured.is_absolute():
        return configured
    candidates = (
        Path.cwd() / configured,
        Path.cwd().parent / configured,
        Path(__file__).resolve().parents[4] / configured,
    )
    return next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _terms(value: str) -> set[str]:
    lowered = value.casefold()
    terms = set(_ASCII_WORD_RE.findall(lowered))
    for run in _CJK_RUN_RE.findall(lowered):
        terms.add(run)
        terms.update(run[index : index + 2] for index in range(len(run) - 1))
    return terms


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, min(len(left), len(right)))


__all__ = [
    "ComplexEmotionScene",
    "EmotionExpressionLibrary",
    "EmotionExpressionRecall",
    "ExpressionEntry",
    "resolve_emotion_expression_root",
]
