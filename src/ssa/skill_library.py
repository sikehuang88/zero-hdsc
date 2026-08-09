"""Persistent official-format Skill package library for Zero."""

from __future__ import annotations

import io
import json
import re
import shutil
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from ssa.skills import DEFAULT_SKILLS, SkillDefinition

_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_FRONTMATTER_RE = re.compile(r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|\Z)", re.DOTALL)
_TOOL_NAME_RE = re.compile(
    r"\b(?:firecrawl_[a-z0-9_]+|web_[a-z0-9_]+|truth_[a-z0-9_]+|"
    r"coding_[a-z0-9_]+|win32_[a-z0-9_]+|"
    r"mineradio_[a-z0-9_]+|soda_music[a-z0-9_]*|read_file|write_file|powershell|generate_image|ask_gpt)\b"
)
_ALLOWED_TOP_LEVEL = {"SKILL.md", "agents", "scripts", "references", "assets"}
_RESERVED_SKILL_IDS = {skill.id for skill in DEFAULT_SKILLS} | {"web-research"}
_DESCRIPTION_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}|[\u4e00-\u9fff]{2,}")
_GENERIC_DESCRIPTION_TERMS = {
    "when",
    "with",
    "from",
    "this",
    "that",
    "skill",
    "user",
    "users",
    "使用",
    "用户",
    "技能",
    "用于",
    "可以",
    "需要",
}


@dataclass(frozen=True)
class InstalledSkill:
    id: str
    name: str
    description: str
    body: str
    path: Path
    enabled: bool
    source: str
    files: tuple[str, ...]
    tool_patterns: tuple[str, ...]

    def to_definition(self) -> SkillDefinition:
        return SkillDefinition(
            id=self.id,
            name=self.name,
            description=self.description,
            instruction=self.body,
            tool_patterns=self.tool_patterns,
            triggers=_description_triggers(self.name, self.description),
            priority=60,
        )

    def public_record(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "source": self.source,
            "files": list(self.files),
            "tool_patterns": list(self.tool_patterns),
            "custom": True,
        }


class SkillPackageError(ValueError):
    pass


class SkillLibrary:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._state_path = self.root / "library.json"

    def list(self) -> tuple[InstalledSkill, ...]:
        state = self._load_state()
        installed: list[InstalledSkill] = []
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            try:
                installed.append(
                    self._load_directory(
                        directory,
                        enabled=bool(state.get(directory.name, {}).get("enabled", True)),
                        source=str(state.get(directory.name, {}).get("source", "upload")),
                    )
                )
            except SkillPackageError:
                continue
        return tuple(installed)

    def install(self, filename: str, payload: bytes) -> InstalledSkill:
        if not payload:
            raise SkillPackageError("技能文件为空")
        if len(payload) > 20 * 1024 * 1024:
            raise SkillPackageError("技能包不能超过 20 MB")
        suffix = Path(filename).suffix.lower()
        if suffix in {".md", ".markdown"}:
            return self._install_markdown(payload)
        if suffix == ".zip":
            return self._install_zip(payload)
        raise SkillPackageError("仅支持 SKILL.md、Markdown 或 ZIP 技能包")

    def set_enabled(self, skill_id: str, enabled: bool) -> InstalledSkill:
        skill = self.get(skill_id)
        state = self._load_state()
        state.setdefault(skill_id, {})["enabled"] = enabled
        state[skill_id].setdefault("source", skill.source)
        self._save_state(state)
        return self._load_directory(skill.path, enabled=enabled, source=skill.source)

    def remove(self, skill_id: str) -> None:
        skill = self.get(skill_id)
        if skill.path.parent != self.root:
            raise SkillPackageError("技能目录不在用户技能库中")
        shutil.rmtree(skill.path)
        state = self._load_state()
        state.pop(skill_id, None)
        self._save_state(state)

    def get(self, skill_id: str) -> InstalledSkill:
        if not _NAME_RE.fullmatch(skill_id):
            raise SkillPackageError("技能 ID 格式无效")
        path = self.root / skill_id
        if not path.is_dir():
            raise SkillPackageError("技能不存在")
        state = self._load_state().get(skill_id, {})
        return self._load_directory(
            path,
            enabled=bool(state.get("enabled", True)),
            source=str(state.get("source", "upload")),
        )

    def _install_markdown(self, payload: bytes) -> InstalledSkill:
        text = _decode_markdown(payload)
        name, _description, _body = _parse_skill_markdown(text)
        if name in _RESERVED_SKILL_IDS:
            raise SkillPackageError(f"技能 {name} 是内置技能名称")
        destination = self.root / name
        if destination.exists():
            raise SkillPackageError(f"技能 {name} 已存在")
        destination.mkdir()
        (destination / "SKILL.md").write_text(text, encoding="utf-8")
        return self._record_install(destination, source="markdown")

    def _install_zip(self, payload: bytes) -> InstalledSkill:
        try:
            archive = zipfile.ZipFile(io.BytesIO(payload))
        except zipfile.BadZipFile as exc:
            raise SkillPackageError("ZIP 文件已损坏") from exc
        with archive:
            entries = [entry for entry in archive.infolist() if not entry.is_dir()]
            if not entries or len(entries) > 512:
                raise SkillPackageError("技能包文件数量必须在 1 到 512 之间")
            if sum(entry.file_size for entry in entries) > 50 * 1024 * 1024:
                raise SkillPackageError("技能包解压后不能超过 50 MB")
            paths = [_safe_archive_path(entry) for entry in entries]
            skill_candidates = [path for path in paths if path.name.casefold() == "skill.md"]
            if len(skill_candidates) != 1:
                raise SkillPackageError("ZIP 中必须且只能包含一个 SKILL.md")
            skill_path = skill_candidates[0]
            prefix = skill_path.parent
            skill_entry = entries[paths.index(skill_path)]
            text = _decode_markdown(archive.read(skill_entry))
            name, _description, _body = _parse_skill_markdown(text)
            if name in _RESERVED_SKILL_IDS:
                raise SkillPackageError(f"技能 {name} 是内置技能名称")
            destination = self.root / name
            if destination.exists():
                raise SkillPackageError(f"技能 {name} 已存在")
            destination.mkdir()
            try:
                for entry, path in zip(entries, paths, strict=True):
                    try:
                        relative = path.relative_to(prefix) if prefix.parts else path
                    except ValueError:
                        continue
                    if not relative.parts or relative.parts[0] not in _ALLOWED_TOP_LEVEL:
                        continue
                    target = (destination / Path(*relative.parts)).resolve()
                    if destination not in target.parents and target != destination:
                        raise SkillPackageError("技能包包含越界路径")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(entry))
                (destination / "SKILL.md").write_text(text, encoding="utf-8")
            except Exception:
                shutil.rmtree(destination, ignore_errors=True)
                raise
        return self._record_install(destination, source="zip")

    def _record_install(self, destination: Path, *, source: str) -> InstalledSkill:
        skill = self._load_directory(destination, enabled=True, source=source)
        state = self._load_state()
        state[skill.id] = {"enabled": True, "source": source}
        self._save_state(state)
        return skill

    def _load_directory(self, path: Path, *, enabled: bool, source: str) -> InstalledSkill:
        skill_file = path / "SKILL.md"
        if not skill_file.is_file():
            raise SkillPackageError("技能目录缺少 SKILL.md")
        text = skill_file.read_text(encoding="utf-8")
        name, description, body = _parse_skill_markdown(text)
        if name != path.name:
            raise SkillPackageError("技能目录名必须与 frontmatter name 一致")
        files = tuple(
            item.relative_to(path).as_posix() for item in sorted(path.rglob("*")) if item.is_file()
        )
        return InstalledSkill(
            id=name,
            name=name,
            description=description,
            body=body,
            path=path,
            enabled=enabled,
            source=source,
            files=files,
            tool_patterns=tuple(sorted(set(_TOOL_NAME_RE.findall(body)))),
        )

    def _load_state(self) -> dict[str, dict[str, Any]]:
        if not self._state_path.is_file():
            return {}
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _save_state(self, state: dict[str, dict[str, Any]]) -> None:
        temporary = self._state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self._state_path)


def _decode_markdown(payload: bytes) -> str:
    if len(payload) > 2 * 1024 * 1024:
        raise SkillPackageError("SKILL.md 不能超过 2 MB")
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SkillPackageError("SKILL.md 必须使用 UTF-8 编码") from exc


def _parse_skill_markdown(text: str) -> tuple[str, str, str]:
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise SkillPackageError("SKILL.md 缺少 YAML frontmatter")
    try:
        metadata = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise SkillPackageError("SKILL.md frontmatter 不是有效 YAML") from exc
    if not isinstance(metadata, dict):
        raise SkillPackageError("SKILL.md frontmatter 必须是对象")
    if set(metadata) != {"name", "description"}:
        raise SkillPackageError("官方格式只允许 name 和 description 两个 frontmatter 字段")
    name = metadata.get("name")
    description = metadata.get("description")
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name) or len(name) > 64:
        raise SkillPackageError("技能 name 必须是 64 字符内的小写连字符格式")
    if not isinstance(description, str) or not description.strip():
        raise SkillPackageError("技能 description 不能为空")
    body = text[match.end() :].strip()
    if not body:
        raise SkillPackageError("SKILL.md 正文不能为空")
    if len(body) > 100_000:
        raise SkillPackageError("SKILL.md 正文过长")
    return name, description.strip(), body


def _safe_archive_path(entry: zipfile.ZipInfo) -> PurePosixPath:
    mode = entry.external_attr >> 16
    if stat.S_IFMT(mode) == stat.S_IFLNK:
        raise SkillPackageError("技能包不能包含符号链接")
    raw = entry.filename.replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SkillPackageError("技能包包含不安全路径")
    if path.parts[0].endswith(":"):
        raise SkillPackageError("技能包包含绝对 Windows 路径")
    return path


def _description_triggers(name: str, description: str) -> tuple[str, ...]:
    """Derive compact intent terms from official Skill metadata."""
    terms: set[str] = {part for part in name.split("-") if len(part) >= 2}
    for token in _DESCRIPTION_TOKEN_RE.findall(description.casefold()):
        if token in _GENERIC_DESCRIPTION_TERMS:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            terms.update(token[index : index + 2] for index in range(len(token) - 1))
        else:
            terms.add(token)
    return tuple(sorted(terms, key=lambda item: (-len(item), item)))


__all__ = ["InstalledSkill", "SkillLibrary", "SkillPackageError"]
