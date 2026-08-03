from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from ssa.skill_library import SkillLibrary, SkillPackageError


def _skill_markdown(name: str = "weather-brief") -> bytes:
    return (
        "---\n"
        f"name: {name}\n"
        "description: 查询天气并整理简洁的天气提醒\n"
        "---\n\n"
        "# Weather Brief\n\nUse `web_search` to collect current weather facts.\n"
    ).encode()


def _zip_package(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return output.getvalue()


def test_install_standalone_skill_and_persist_enabled_state(tmp_path: Path) -> None:
    library = SkillLibrary(tmp_path / "skills")

    installed = library.install("SKILL.md", _skill_markdown())
    disabled = library.set_enabled(installed.id, False)

    assert installed.id == "weather-brief"
    assert installed.tool_patterns == ("web_search",)
    assert disabled.enabled is False
    assert SkillLibrary(tmp_path / "skills").get(installed.id).enabled is False


def test_install_nested_zip_preserves_official_resource_directories(tmp_path: Path) -> None:
    payload = _zip_package(
        {
            "weather-brief/SKILL.md": _skill_markdown(),
            "weather-brief/agents/openai.yaml": b"interface:\n  display_name: Weather\n",
            "weather-brief/scripts/fetch.py": b"print('resource only')\n",
            "weather-brief/references/schema.md": b"# Schema\n",
            "weather-brief/assets/icon.txt": b"sun",
            "weather-brief/README.md": b"ignored",
        }
    )

    installed = SkillLibrary(tmp_path / "skills").install("weather.zip", payload)

    assert set(installed.files) == {
        "SKILL.md",
        "agents/openai.yaml",
        "scripts/fetch.py",
        "references/schema.md",
        "assets/icon.txt",
    }


def test_zip_path_traversal_is_rejected(tmp_path: Path) -> None:
    payload = _zip_package({"bundle/SKILL.md": _skill_markdown(), "bundle/../outside.txt": b"bad"})

    with pytest.raises(SkillPackageError, match="不安全路径"):
        SkillLibrary(tmp_path / "skills").install("bad.zip", payload)


@pytest.mark.parametrize(
    "payload",
    [
        b"# missing frontmatter",
        b"---\nname: demo\ndescription: ok\nextra: no\n---\nbody",
        b"---\nname: Bad Name\ndescription: ok\n---\nbody",
    ],
)
def test_invalid_official_metadata_is_rejected(tmp_path: Path, payload: bytes) -> None:
    with pytest.raises(SkillPackageError):
        SkillLibrary(tmp_path / "skills").install("SKILL.md", payload)


def test_duplicate_reserved_and_delete_behaviors(tmp_path: Path) -> None:
    library = SkillLibrary(tmp_path / "skills")
    library.install("SKILL.md", _skill_markdown())

    with pytest.raises(SkillPackageError, match="已存在"):
        library.install("SKILL.md", _skill_markdown())
    with pytest.raises(SkillPackageError, match="内置技能"):
        library.install("SKILL.md", _skill_markdown("coding"))
    with pytest.raises(SkillPackageError, match="内置技能"):
        library.install("SKILL.md", _skill_markdown("web-research"))

    library.remove("weather-brief")
    assert library.list() == ()
