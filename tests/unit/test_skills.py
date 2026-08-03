from __future__ import annotations

from ssa.skills import SkillRegistry


def test_web_research_selects_only_research_skill_in_chat() -> None:
    selection = SkillRegistry().select("搜索 GitHub 上的免费 API", mode="chat")

    assert selection.ids == ("zero-web-search",)
    assert selection.accepts_tool("firecrawl_search") is True
    assert selection.accepts_tool("web_search") is True
    assert selection.accepts_tool("truth_news") is True
    assert selection.accepts_tool("write_file") is False
    assert "firecrawl_search" in selection.skills[0].instruction


def test_coding_mode_always_loads_coding_skill() -> None:
    selection = SkillRegistry().select("优化这个功能", mode="coding")

    assert selection.ids == ("coding",)
    assert selection.accepts_tool("coding_search") is True
    assert selection.accepts_tool("write_file") is True
    assert selection.accepts_tool("soda_music_control") is False


def test_multiple_domains_compose_up_to_the_bounded_limit() -> None:
    selection = SkillRegistry().select(
        "搜索资料后生成图片，并检查电脑磁盘空间",
        mode="chat",
    )

    assert selection.ids == ("zero-web-search", "image-studio", "system-observer")
    assert selection.accepts_tool("web_read") is True
    assert selection.accepts_tool("generate_image") is True
    assert selection.accepts_tool("win32_drives") is True


def test_disabled_skill_is_not_selected() -> None:
    selection = SkillRegistry().select(
        "播放一首歌",
        mode="chat",
        enabled_ids={"zero-web-search"},
    )

    assert selection.ids == ()
    assert selection.accepts_tool("soda_music_search_play") is False


def test_legacy_web_research_id_maps_to_zero_web_search() -> None:
    selection = SkillRegistry().select(
        "搜索最新资料",
        mode="chat",
        enabled_ids={"web-research"},
    )

    assert selection.ids == ("zero-web-search",)


def test_catalog_is_stable_and_public_safe() -> None:
    catalog = [skill.public_record() for skill in SkillRegistry().catalog()]

    assert [item["id"] for item in catalog] == [
        "coding",
        "music",
        "zero-web-search",
        "image-studio",
        "system-observer",
        "files",
        "second-opinion",
    ]
    assert all("instruction" not in item for item in catalog)
