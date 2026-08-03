"""Bundled official-format skills shipped with Zero."""

from __future__ import annotations

from importlib.resources import files


def load_skill_body(skill_id: str) -> str:
    text = files("ssa.builtin_skills").joinpath(skill_id, "SKILL.md").read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"bundled skill {skill_id!r} is missing frontmatter")
    _opening, separator, remainder = text.partition("\n---")
    if not separator:
        raise ValueError(f"bundled skill {skill_id!r} has invalid frontmatter")
    body = remainder.lstrip("\r\n").strip()
    if not body:
        raise ValueError(f"bundled skill {skill_id!r} has an empty body")
    return body


__all__ = ["load_skill_body"]
