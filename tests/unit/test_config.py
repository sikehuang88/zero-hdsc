"""Tests for configuration loading."""

from __future__ import annotations

import pytest

from ssa.config import (
    AblationConfig,
    Environment,
    Secrets,
    Settings,
    SettingsError,
    load_settings,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_load_defaults_only():
    """Loading with no env overrides should still produce valid settings."""
    s = load_settings(Environment.DEVELOPMENT, environ={})
    assert isinstance(s, Settings)
    assert s.app.name == "ssa"
    # development.toml overrides timezone.
    assert s.app.timezone == "Asia/Shanghai"
    # defaults.toml provides the model.
    assert s.llm.model == "deepseek/deepseek-chat"


def test_production_environment_has_no_timezone_override():
    s = load_settings(Environment.PRODUCTION, environ={})
    # No production.toml exists, so defaults apply.
    assert s.app.timezone == "UTC"


# ---------------------------------------------------------------------------
# Env var overrides
# ---------------------------------------------------------------------------


def test_env_overrides_llm_model():
    s = load_settings(Environment.DEVELOPMENT, environ={"SSA_LLM_MODEL": "claude-3-haiku"})
    assert s.llm.model == "claude-3-haiku"


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


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


def test_secrets_from_env():
    sec = Secrets.from_env(
        {
            "SSA_LLM_API_KEY": "sk-test",
            "SSA_TELEGRAM_BOT_TOKEN": "tok",
            "SSA_TELEGRAM_USER_ID": "12345",
            "SSA_EXPORT_PASSWORD": "pw",
        }
    )
    assert sec.llm_api_key.get_secret_value() == "sk-test"
    assert sec.telegram_bot_token.get_secret_value() == "tok"
    assert sec.telegram_user_id == 12345
    assert sec.export_password.get_secret_value() == "pw"


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
    with pytest.raises(SettingsError, match="SSA_TELEGRAM_BOT_TOKEN"):
        sec.require_telegram()


def test_secrets_require_telegram_raises_when_user_id_missing():
    sec = Secrets.from_env({"SSA_TELEGRAM_BOT_TOKEN": "tok"})
    with pytest.raises(SettingsError, match="SSA_TELEGRAM_USER_ID"):
        sec.require_telegram()


def test_secrets_require_telegram_returns_tuple_when_present():
    sec = Secrets.from_env(
        {"SSA_TELEGRAM_BOT_TOKEN": "tok", "SSA_TELEGRAM_USER_ID": "42"}
    )
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
    from ssa.config import LLMConfig

    with pytest.raises(ValueError, match="temperature"):
        LLMConfig(temperature=5.0)


def test_final_k_out_of_range_rejected():
    from ssa.config import RetrievalConfig

    with pytest.raises(ValueError):
        RetrievalConfig(final_k=0)
    with pytest.raises(ValueError):
        RetrievalConfig(final_k=51)


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
