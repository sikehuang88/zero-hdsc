"""Declarative skill selection layered over Zero's existing tool registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ssa.builtin_skills import load_skill_body

SkillMode = Literal["chat", "coding", "any"]


@dataclass(frozen=True)
class SkillDefinition:
    id: str
    name: str
    description: str
    instruction: str
    tool_patterns: tuple[str, ...]
    triggers: tuple[str, ...]
    mode: SkillMode = "any"
    priority: int = 0

    def supports_mode(self, mode: Literal["chat", "coding"]) -> bool:
        return self.mode == "any" or self.mode == mode

    def matches(self, content: str) -> bool:
        normalized = content.casefold()
        return any(trigger.casefold() in normalized for trigger in self.triggers)

    def accepts_tool(self, name: str) -> bool:
        return any(
            name == pattern or (pattern.endswith("*") and name.startswith(pattern[:-1]))
            for pattern in self.tool_patterns
        )

    def public_record(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "mode": self.mode,
            "tool_patterns": list(self.tool_patterns),
            "priority": self.priority,
        }


@dataclass(frozen=True)
class SkillSelection:
    skills: tuple[SkillDefinition, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(skill.id for skill in self.skills)

    def accepts_tool(self, name: str) -> bool:
        return any(skill.accepts_tool(name) for skill in self.skills)

    def prompt_context(self) -> str:
        if not self.skills:
            return ""
        lines = [
            "<active_skills>",
            "Use these selected skills as private capability guidance. Choose tools only from "
            "their exposed set and combine skills when the task genuinely crosses domains.",
        ]
        lines.extend(f"- {skill.id} / {skill.name}: {skill.instruction}" for skill in self.skills)
        lines.append("</active_skills>")
        return "\n".join(lines)


class SkillRegistry:
    def __init__(self, definitions: tuple[SkillDefinition, ...] | None = None) -> None:
        self._skills = {skill.id: skill for skill in definitions or DEFAULT_SKILLS}

    def catalog(self) -> tuple[SkillDefinition, ...]:
        return tuple(sorted(self._skills.values(), key=lambda item: (-item.priority, item.id)))

    def select(
        self,
        content: str,
        *,
        mode: Literal["chat", "coding"],
        enabled_ids: set[str] | None = None,
        max_skills: int = 3,
    ) -> SkillSelection:
        enabled = set(enabled_ids) if enabled_ids is not None else set(self._skills)
        if "web-research" in enabled:
            enabled.discard("web-research")
            enabled.add("zero-web-search")
        selected = [
            skill
            for skill in self._skills.values()
            if skill.id in enabled and skill.supports_mode(mode) and skill.matches(content)
        ]
        if mode == "coding" and "coding" in enabled:
            coding = self._skills.get("coding")
            if coding is not None and coding not in selected:
                selected.append(coding)
        selected.sort(key=lambda item: (-item.priority, item.id))
        return SkillSelection(tuple(selected[:max_skills]))


DEFAULT_SKILLS = (
    SkillDefinition(
        "zero-web-search",
        "Zero Web Search",
        "使用 Firecrawl 进行实时全网搜索、网页抓取、多源核验与可追溯引用。",
        load_skill_body("zero-web-search"),
        ("firecrawl_*", "web_*", "truth_*"),
        (
            "搜索",
            "查询",
            "查一下",
            "全网",
            "网上",
            "网页",
            "网站",
            "最新消息",
            "帮我研究",
            "找资料",
            "github",
            "新闻",
            "天气",
            "翻译",
        ),
        priority=90,
    ),
    SkillDefinition(
        "coding",
        "Coding",
        "理解工程、检索代码、修改文件并执行验证。",
        "Inspect before editing, make precise changes, then test.",
        ("coding_*", "read_file", "write_file", "powershell"),
        ("代码", "项目", "修复", "开发", "实现", "构建", "测试", "报错", "文件"),
        mode="coding",
        priority=100,
    ),
    SkillDefinition(
        "system-observer",
        "系统观察",
        "读取 Windows 当前窗口、进程、磁盘、目录与剪贴板状态。",
        "Observe host state before making factual claims about the device.",
        ("win32_*", "powershell"),
        ("电脑", "系统", "磁盘", "c盘", "e盘", "窗口", "进程", "剪贴板", "目录", "剩余空间"),
        priority=80,
    ),
    SkillDefinition(
        "music",
        "音乐控制",
        "搜索并播放多平台音乐; 控制 ZERO 音乐模式; 同时保留汽水桌面播放能力。",
        "Prefer Mineradio for ZERO's native music mode. Use Soda Music only when the user explicitly names it.",
        ("mineradio_*", "soda_music*"),
        ("音乐", "歌曲", "播放", "暂停", "下一首", "上一首", "汽水", "想听"),
        priority=95,
    ),
    SkillDefinition(
        "files",
        "文件操作",
        "读取、创建、覆盖或追加本机文件。",
        "Use exact paths, preserve unrelated content, and verify important writes.",
        ("read_file", "write_file", "powershell"),
        ("读取文件", "打开文件", "写入", "保存文件", "创建文件", "修改文件"),
        priority=75,
    ),
    SkillDefinition(
        "image-studio",
        "图像工作室",
        "根据自然语言生成图片并切换到 Zero 舞台展示。",
        "Build a concrete prompt and generate only when the user wants a new image.",
        ("generate_image",),
        ("生成图片", "画一张", "生图", "做张图", "绘制", "图片生成"),
        priority=85,
    ),
    SkillDefinition(
        "second-opinion",
        "第二视角",
        "调用独立模型进行交叉分析或多模态复核。",
        "Request a bounded second opinion, then critically integrate it.",
        ("ask_gpt",),
        ("第二意见", "另一个模型", "复核", "交叉分析", "再分析一次"),
        priority=70,
    ),
)


__all__ = ["DEFAULT_SKILLS", "SkillDefinition", "SkillRegistry", "SkillSelection"]
