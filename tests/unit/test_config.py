"""Tests for configuration loading."""

from __future__ import annotations

import pytest

from ssa.config import (
    AblationConfig,
    Environment,
    FirecrawlConfig,
    HDSCConfig,
    LLMConfig,
    MultimodalConfig,
    PerceptionConfig,
    ReasoningEffort,
    Secrets,
    Settings,
    SettingsError,
    ThinkingMode,
    Win32Config,
    load_settings,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_load_defaults_only():
    """Loading with no env overrides should still produce valid settings."""
    s = load_settings(Environment.DEVELOPMENT, environ={})
    assert isinstance(s, Settings)
    assert s.app.name == "hdsc"
    # development.toml overrides timezone.
    assert s.app.timezone == "Asia/Shanghai"
    # defaults.toml provides the model.
    assert s.llm.model == "deepseek/deepseek-v4-flash"
    assert s.llm.reasoning_model == "deepseek/deepseek-v4-pro"
    assert s.llm.base_url == "https://api.deepseek.com"
    assert s.hdsc.h2_shadow_enabled is True
    assert s.hdsc.h2_active_capacity == 8
    assert s.hdsc.h2_cluster_bits == 16
    assert s.identity.review_interval_agent_events == 4
    assert s.identity.review_window_events == 24
    assert s.tools.enabled is True
    assert s.tools.max_rounds == 8
    assert s.tools.allow_elevation is True
    assert s.win32 == Win32Config()
    assert s.firecrawl == FirecrawlConfig()
    assert s.multimodal.enabled is True
    assert s.multimodal.model == "gpt-5.6-luna"
    assert s.multimodal.base_url == "https://sky1818.com/v1"
    assert s.inner_life.heartbeat_seconds == 60
    assert s.inner_life.base_wait_minutes == 45
    assert s.inner_life.max_wait_minutes == 720
    assert s.inner_life.offline_entry_threshold == 0.58
    assert s.reflective_learning.schedule_interval_minutes == 15
    assert s.reflective_learning.consolidation_interval_minutes == 15
    assert s.emotion_library.enabled is True
    assert s.emotion_library.recall_limit == 6
    assert s.reflective_learning.outcome_interval_minutes == 60
    assert s.reflective_learning.min_critic_score == 0.62


def test_production_environment_has_no_timezone_override():
    s = load_settings(Environment.PRODUCTION, environ={})
    # No production.toml exists, so defaults apply.
    assert s.app.timezone == "UTC"


def test_unknown_timezone_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone"):
        Settings(app={"timezone": "Mars/Olympus_Mons"})


def test_default_config_dir_prefers_current_working_tree(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "defaults.toml").write_text(
        '[app]\nname = "installed-ssa"\ntimezone = "UTC"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    settings = load_settings(Environment.PRODUCTION, environ={})

    assert settings.app.name == "installed-ssa"


# ---------------------------------------------------------------------------
# Env var overrides
# ---------------------------------------------------------------------------


def test_env_overrides_llm_model():
    s = load_settings(
        Environment.DEVELOPMENT,
        environ={"SSA_LLM_MODEL": "deepseek/deepseek-v4-pro"},
    )
    assert s.llm.model == "deepseek/deepseek-v4-pro"


def test_hdsc_env_prefix_overrides_legacy_alias() -> None:
    settings = load_settings(
        Environment.DEVELOPMENT,
        environ={
            "SSA_LLM_MODEL": "deepseek/deepseek-v4-flash",
            "HDSC_LLM_MODEL": "deepseek/deepseek-v4-pro",
            "HDSC_RETRIEVAL_FINAL_K": "12",
        },
    )

    assert settings.llm.model == "deepseek/deepseek-v4-pro"
    assert settings.retrieval.final_k == 12


def test_env_overrides_deepseek_v4_controls():
    s = load_settings(
        Environment.DEVELOPMENT,
        environ={
            "SSA_LLM_REASONING_MODEL": "deepseek/deepseek-v4-pro",
            "SSA_LLM_BASE_URL": "https://proxy.example/v1/",
            "SSA_LLM_BETA_BASE_URL": "https://proxy.example/beta/",
            "SSA_LLM_TOP_P": "0.95",
            "SSA_LLM_MAX_TOKENS": "393216",
            "SSA_LLM_TIMEOUT_SECONDS": "300",
            "SSA_LLM_RETRY_COUNT": "4",
            "SSA_LLM_JSON_RETRY_COUNT": "2",
            "SSA_LLM_THINKING_MODE": "enabled",
            "SSA_LLM_REASONING_EFFORT": "max",
            "SSA_DEEPSEEK_USER_ID": "ssa_test_user",
        },
    )
    assert s.llm.base_url == "https://proxy.example/v1"
    assert s.llm.beta_base_url == "https://proxy.example/beta"
    assert s.llm.temperature is None
    assert s.llm.top_p == 0.95
    assert s.llm.max_tokens == 393_216
    assert s.llm.timeout_seconds == 300
    assert s.llm.retry_count == 4
    assert s.llm.json_retry_count == 2
    assert s.llm.thinking_mode == ThinkingMode.ENABLED
    assert s.llm.reasoning_effort == ReasoningEffort.MAX
    assert s.llm.user_id == "ssa_test_user"


def test_env_overrides_temperature():
    s = load_settings(
        Environment.DEVELOPMENT,
        environ={"SSA_LLM_TEMPERATURE": "1.5"},
    )
    assert s.llm.temperature == 1.5


def test_env_overrides_embedding_dim():
    s = load_settings(
        Environment.DEVELOPMENT,
        environ={"SSA_EMBEDDING_DIM": "1024"},
    )
    assert s.embedding.dim == 1024


def test_env_overrides_retrieval():
    s = load_settings(
        Environment.DEVELOPMENT,
        environ={
            "SSA_RETRIEVAL_CANDIDATES": "100",
            "SSA_RETRIEVAL_FINAL_K": "12",
        },
    )
    assert s.retrieval.candidates == 100
    assert s.retrieval.final_k == 12


def test_hdsc_h2_env_overrides_take_precedence() -> None:
    settings = load_settings(
        Environment.DEVELOPMENT,
        environ={
            "SSA_H2_ACTIVE_CAPACITY": "3",
            "HDSC_H2_ACTIVE_CAPACITY": "12",
            "HDSC_H2_CANDIDATE_TOP_K": "5",
            "HDSC_H2_CLUSTER_BITS": "20",
            "HDSC_H2_SHADOW_ENABLED": "false",
            "HDSC_H2_RETENTION": "0.70",
            "HDSC_H2_INJECTION_RATE": "0.20",
            "HDSC_H2_HYSTERESIS": "0.04",
        },
    )

    assert settings.hdsc.h2_active_capacity == 12
    assert settings.hdsc.h2_candidate_top_k == 5
    assert settings.hdsc.h2_cluster_bits == 20
    assert settings.hdsc.h2_shadow_enabled is False
    assert settings.hdsc.h2_retention == 0.70
    assert settings.hdsc.h2_injection_rate == 0.20
    assert settings.hdsc.h2_hysteresis == 0.04


def test_env_overrides_initiative():
    s = load_settings(
        Environment.DEVELOPMENT,
        environ={
            "SSA_INITIATIVE_DAILY_LIMIT": "5",
            "SSA_INITIATIVE_COOLDOWN_MINUTES": "120",
            "SSA_QUIET_HOURS_START": "22:00",
            "SSA_QUIET_HOURS_END": "06:00",
        },
    )
    assert s.initiative.daily_limit == 5
    assert s.initiative.cooldown_minutes == 120
    assert s.initiative.quiet_hours_start == "22:00"
    assert s.initiative.quiet_hours_end == "06:00"


def test_env_overrides_budget():
    s = load_settings(
        Environment.DEVELOPMENT,
        environ={"SSA_BACKGROUND_DAILY_LLM_BUDGET": "100"},
    )
    assert s.budget.background_daily_llm_budget == 100


def test_env_overrides_identity_review_schedule():
    settings = load_settings(
        Environment.DEVELOPMENT,
        environ={
            "HDSC_IDENTITY_REVIEW_INTERVAL_AGENT_EVENTS": "6",
            "HDSC_IDENTITY_REVIEW_WINDOW_EVENTS": "32",
        },
    )

    assert settings.identity.review_interval_agent_events == 6
    assert settings.identity.review_window_events == 32


def test_env_overrides_global_tool_kernel():
    settings = load_settings(
        Environment.DEVELOPMENT,
        environ={
            "HDSC_TOOLS_ENABLED": "false",
            "HDSC_TOOLS_MAX_ROUNDS": "6",
            "HDSC_TOOLS_DEFAULT_TIMEOUT_SECONDS": "90",
            "HDSC_TOOLS_MAX_OUTPUT_CHARS": "64000",
            "HDSC_TOOLS_ALLOW_ELEVATION": "false",
        },
    )

    assert settings.tools.enabled is False
    assert settings.tools.max_rounds == 6
    assert settings.tools.default_timeout_seconds == 90
    assert settings.tools.max_output_chars == 64_000
    assert settings.tools.allow_elevation is False


def test_env_overrides_win32_registry() -> None:
    settings = load_settings(
        Environment.DEVELOPMENT,
        environ={
            "HDSC_WIN32_ENABLED": "false",
            "HDSC_WIN32_CLIPBOARD_ENABLED": "false",
            "HDSC_WIN32_MAX_WINDOWS": "20",
            "HDSC_WIN32_MAX_PROCESSES": "300",
            "HDSC_WIN32_MAX_DIRECTORY_ENTRIES": "500",
            "HDSC_WIN32_MAX_CLIPBOARD_CHARS": "8000",
        },
    )

    assert settings.win32 == Win32Config(
        enabled=False,
        clipboard_enabled=False,
        max_windows=20,
        max_processes=300,
        max_directory_entries=500,
        max_clipboard_chars=8_000,
    )


def test_env_overrides_external_truth_kernel() -> None:
    settings = load_settings(
        Environment.DEVELOPMENT,
        environ={
            "HDSC_EXTERNAL_TRUTH_ENABLED": "true",
            "HDSC_EXTERNAL_TRUTH_SOURCES": "weather,news,service_status",
            "HDSC_EXTERNAL_TRUTH_TIMEOUT_SECONDS": "9",
            "HDSC_EXTERNAL_TRUTH_CACHE_TTL_SECONDS": "600",
            "HDSC_EXTERNAL_TRUTH_MAX_RESPONSE_BYTES": "128000",
            "HDSC_EXTERNAL_TRUTH_MIN_REQUEST_INTERVAL_MS": "250",
        },
    )

    assert settings.external_truth.enabled is True
    assert settings.external_truth.enabled_sources == ["weather", "news", "service_status"]
    assert settings.external_truth.request_timeout_seconds == 9
    assert settings.external_truth.cache_ttl_seconds == 600
    assert settings.external_truth.max_response_bytes == 128_000
    assert settings.external_truth.min_request_interval_ms == 250


def test_env_overrides_firecrawl_kernel() -> None:
    settings = load_settings(
        Environment.DEVELOPMENT,
        environ={
            "HDSC_FIRECRAWL_ENABLED": "false",
            "HDSC_FIRECRAWL_BASE_URL": "https://firecrawl.example/v2/",
            "HDSC_FIRECRAWL_TIMEOUT_SECONDS": "45",
            "HDSC_FIRECRAWL_MAX_RESPONSE_BYTES": "8000000",
            "HDSC_FIRECRAWL_MAX_RESULT_CONTENT_CHARS": "12000",
        },
    )

    assert settings.firecrawl == FirecrawlConfig(
        enabled=False,
        base_url="https://firecrawl.example/v2",
        request_timeout_seconds=45,
        max_response_bytes=8_000_000,
        max_result_content_chars=12_000,
    )


def test_env_overrides_multimodal_adapter() -> None:
    settings = load_settings(
        Environment.DEVELOPMENT,
        environ={
            "HDSC_MULTIMODAL_ENABLED": "false",
            "HDSC_MULTIMODAL_MODEL": "gpt-5.6-luna",
            "HDSC_MULTIMODAL_BASE_URL": "https://gateway.example/v1/",
            "HDSC_MULTIMODAL_REASONING_EFFORT": "high",
            "HDSC_MULTIMODAL_TIMEOUT_SECONDS": "240",
            "HDSC_MULTIMODAL_MAX_FILES": "5",
            "HDSC_MULTIMODAL_MAX_FILE_BYTES": "10485760",
            "HDSC_MULTIMODAL_MAX_TOTAL_BYTES": "20971520",
        },
    )

    assert settings.multimodal == MultimodalConfig(
        enabled=False,
        model="gpt-5.6-luna",
        base_url="https://gateway.example/v1",
        reasoning_effort="high",
        timeout_seconds=240,
        max_files=5,
        max_file_bytes=10_485_760,
        max_total_bytes=20_971_520,
    )


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


def test_secrets_from_env():
    sec = Secrets.from_env(
        {
            "SSA_LLM_API_KEY": "sk-test",
            "HDSC_MULTIMODAL_API_KEY": "sk-multimodal",
            "HDSC_FIRECRAWL_API_KEY": "fc-test",
            "SSA_TELEGRAM_BOT_TOKEN": "tok",
            "SSA_TELEGRAM_USER_ID": "12345",
            "SSA_EXPORT_PASSWORD": "pw",
        }
    )
    assert sec.llm_api_key.get_secret_value() == "sk-test"
    assert sec.multimodal_api_key.get_secret_value() == "sk-multimodal"
    assert sec.firecrawl_api_key.get_secret_value() == "fc-test"
    assert sec.telegram_bot_token.get_secret_value() == "tok"
    assert sec.telegram_user_id == 12345
    assert sec.export_password.get_secret_value() == "pw"


def test_hdsc_secrets_take_precedence_over_legacy_aliases() -> None:
    secrets = Secrets.from_env(
        {
            "HDSC_LLM_API_KEY": "hdsc-key",
            "SSA_LLM_API_KEY": "legacy-key",
            "HDSC_TELEGRAM_BOT_TOKEN": "hdsc-token",
            "HDSC_TELEGRAM_USER_ID": "77",
            "HDSC_EXPORT_PASSWORD": "hdsc-password",
        }
    )

    assert secrets.llm_api_key.get_secret_value() == "hdsc-key"
    assert secrets.telegram_bot_token.get_secret_value() == "hdsc-token"
    assert secrets.telegram_user_id == 77
    assert secrets.export_password.get_secret_value() == "hdsc-password"


def test_secrets_accept_official_deepseek_key_name():
    sec = Secrets.from_env({"DEEPSEEK_API_KEY": "sk-deepseek"})
    assert sec.llm_api_key.get_secret_value() == "sk-deepseek"


def test_secrets_empty_by_default():
    sec = Secrets.from_env({})
    assert sec.llm_api_key.get_secret_value() == ""
    assert sec.telegram_bot_token.get_secret_value() == ""
    assert sec.telegram_user_id == 0


def test_secrets_require_llm_raises_when_missing():
    sec = Secrets.from_env({})
    with pytest.raises(SettingsError, match="SSA_LLM_API_KEY"):
        sec.require_llm()


def test_secrets_require_telegram_raises_when_token_missing():
    sec = Secrets.from_env({})
    with pytest.raises(SettingsError, match="HDSC_TELEGRAM_BOT_TOKEN"):
        sec.require_telegram()


def test_secrets_require_telegram_raises_when_user_id_missing():
    sec = Secrets.from_env({"SSA_TELEGRAM_BOT_TOKEN": "tok"})
    with pytest.raises(SettingsError, match="HDSC_TELEGRAM_USER_ID"):
        sec.require_telegram()


def test_secrets_require_telegram_returns_tuple_when_present():
    sec = Secrets.from_env({"SSA_TELEGRAM_BOT_TOKEN": "tok", "SSA_TELEGRAM_USER_ID": "42"})
    token, uid = sec.require_telegram()
    assert token == "tok"
    assert uid == 42


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_ablation_baseline_must_be_known():
    with pytest.raises(ValueError, match="baseline"):
        AblationConfig(baseline="Z9")


def test_ablation_all_baselines_valid():
    for b in ["B0", "B1", "B2", "B3", "B4", "B5"]:
        assert AblationConfig(baseline=b).baseline == b


def test_temperature_out_of_range_rejected():
    with pytest.raises(ValueError, match="temperature"):
        LLMConfig(temperature=5.0)


def test_deprecated_model_alias_rejected():
    with pytest.raises(ValueError, match="DeepSeek V4"):
        LLMConfig(model="deepseek/deepseek-chat")


def test_sampling_controls_are_mutually_exclusive():
    with pytest.raises(ValueError, match="temperature or top_p"):
        LLMConfig(temperature=0.5, top_p=0.9)


@pytest.mark.parametrize(
    "base_url",
    ["https://", "ftp://api.deepseek.com", "https://api.deepseek.com?tenant=x"],
)
def test_invalid_base_url_is_rejected(base_url: str):
    with pytest.raises(ValueError, match="base URL"):
        LLMConfig(base_url=base_url)


def test_dotenv_is_loaded_when_environment_mapping_is_implicit(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (tmp_path / ".env").write_text(
        "SSA_LLM_MODEL=deepseek/deepseek-v4-pro\nDEEPSEEK_API_KEY=sk-from-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SSA_LLM_MODEL", raising=False)
    monkeypatch.delenv("SSA_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    settings = load_settings(
        Environment.DEVELOPMENT,
        config_dir=config_dir,
    )

    assert settings.llm.model == "deepseek/deepseek-v4-pro"
    assert settings.secrets.llm_api_key.get_secret_value() == "sk-from-dotenv"


def test_final_k_out_of_range_rejected():
    from ssa.config import RetrievalConfig

    with pytest.raises(ValueError):
        RetrievalConfig(final_k=0)
    with pytest.raises(ValueError):
        RetrievalConfig(final_k=51)


@pytest.mark.parametrize(
    "values",
    [
        {"h2_active_capacity": 0},
        {"h2_active_capacity": 2, "h2_candidate_top_k": 3},
        {"h2_cluster_bits": 3},
        {"h2_retention": 1.0},
        {"h2_retention": 0.9, "h2_injection_rate": 0.2},
        {"h2_hysteresis": -0.1},
        {"h2_mass_tolerance": 0.0},
        {"h2_gain_margin": 1.0},
        {"h2_score_lipschitz_bound": float("inf")},
    ],
)
def test_invalid_h2_configuration_is_rejected(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        HDSCConfig(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"context_window_events": 1},
        {"fresh_gap_minutes": 0},
        {"fresh_gap_minutes": 60, "long_gap_hours": 1.0},
        {"mode_temperature": 0.0},
    ],
)
def test_invalid_perception_configuration_is_rejected(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PerceptionConfig(**values)


# ---------------------------------------------------------------------------
# No secret leakage
# ---------------------------------------------------------------------------


def test_settings_str_does_not_leak_secret():
    """Pipeline §10.4: logs must never output secret values."""
    s = load_settings(
        Environment.DEVELOPMENT,
        environ={"SSA_LLM_API_KEY": "sk-super-secret"},
    )
    rendered = repr(s)
    assert "sk-super-secret" not in rendered
    assert "sk-super-secret" not in str(s)
    assert "sk-super-secret" not in s.model_dump_json()
