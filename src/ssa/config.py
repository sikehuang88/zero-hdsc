"""Configuration loading — layered, typed, secret-aware.

Pipeline §10.2 defines the precedence:

    CLI override  >  env vars / .env  >  config/<env>.toml  >  config/defaults.toml

Secrets (API keys, tokens, passwords) live ONLY in environment variables
and are never read from TOML files. Behavior/experiment parameters live in
version-controlled TOML and may be overridden by env vars for experiments.
"""

from __future__ import annotations

import os
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, SecretStr, field_validator


class Environment(StrEnum):
    DEVELOPMENT = "development"
    EVALUATION = "evaluation"
    PRODUCTION = "production"


# ---------------------------------------------------------------------------
# Secret bundle — sourced exclusively from environment variables.
# ---------------------------------------------------------------------------


class Secrets(BaseModel):
    """API keys and passwords. Never serialized to disk."""

    llm_api_key: SecretStr = SecretStr("")
    telegram_bot_token: SecretStr = SecretStr("")
    telegram_user_id: int = 0
    export_password: SecretStr = SecretStr("")

    @classmethod
    def from_env(cls, env: dict[str, str]) -> Secrets:
        return cls(
            llm_api_key=SecretStr(env.get("SSA_LLM_API_KEY", "")),
            telegram_bot_token=SecretStr(env.get("SSA_TELEGRAM_BOT_TOKEN", "")),
            telegram_user_id=int(env.get("SSA_TELEGRAM_USER_ID", "0") or "0"),
            export_password=SecretStr(env.get("SSA_EXPORT_PASSWORD", "")),
        )

    def require_llm(self) -> str:
        v = self.llm_api_key.get_secret_value()
        if not v:
            raise SettingsError(
                "Missing required secret SSA_LLM_API_KEY. "
                "Set it in your .env file or environment."
            )
        return v

    def require_telegram(self) -> tuple[str, int]:
        token = self.telegram_bot_token.get_secret_value()
        if not token:
            raise SettingsError(
                "Missing required secret SSA_TELEGRAM_BOT_TOKEN. "
                "Set it in your .env file or environment."
            )
        if self.telegram_user_id == 0:
            raise SettingsError(
                "Missing required secret SSA_TELEGRAM_USER_ID. "
                "Set it in your .env file or environment."
            )
        return token, self.telegram_user_id


# ---------------------------------------------------------------------------
# Behavior / experiment parameter bundles — version-controlled.
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    model: str = "deepseek/deepseek-chat"
    temperature: float = 0.8
    max_tokens: int = 1024
    timeout_seconds: int = 30
    retry_count: int = 2

    @field_validator("temperature")
    @classmethod
    def _check_temp(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        return v

    @field_validator("max_tokens")
    @classmethod
    def _check_tokens(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-small-zh-v1.5"
    dim: int = 512
    cache_dir: str = "models"
    batch_size: int = 32

    @field_validator("dim")
    @classmethod
    def _check_dim(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("embedding dim must be positive")
        return v


class RetrievalConfig(BaseModel):
    candidates: int = 50
    final_k: int = 8
    semantic_weight: float = 0.35
    recency_weight: float = 0.15
    importance_weight: float = 0.15
    relationship_weight: float = 0.15
    unresolved_weight: float = 0.10
    source_trust_weight: float = 0.10
    dedup_similarity_threshold: float = 0.92
    radiation_decay: float = 0.5

    @field_validator("final_k")
    @classmethod
    def _check_k(cls, v: int) -> int:
        if not 1 <= v <= 50:
            raise ValueError("final_k must be in [1, 50]")
        return v


class InitiativeConfig(BaseModel):
    daily_limit: int = 3
    cooldown_minutes: int = 240
    quiet_hours_start: str = "23:00"
    quiet_hours_end: str = "07:00"
    min_urgency: float = 0.70
    min_gap_hours: int = 4

    @field_validator("daily_limit")
    @classmethod
    def _check_daily(cls, v: int) -> int:
        if v < 0:
            raise ValueError("daily_limit must be >= 0")
        return v

    @field_validator("min_urgency")
    @classmethod
    def _check_urgency(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("min_urgency must be in [0, 1]")
        return v


class BudgetConfig(BaseModel):
    background_daily_llm_budget: int = 20
    sync_llm_per_turn: int = 3
    goal_sync_llm_per_turn: int = 2

    @field_validator("background_daily_llm_budget")
    @classmethod
    def _check_budget(cls, v: int) -> int:
        if v < 0:
            raise ValueError("background_daily_llm_budget must be >= 0")
        return v


class DatabaseConfig(BaseModel):
    path: str = "data/ssa.db"
    busy_timeout_ms: int = 5000


class StateConfig(BaseModel):
    valence_decay: float = 0.75
    valence_signal_weight: float = 0.25
    arousal_signal_weight: float = 0.30
    arousal_decay_per_hour: float = 0.10
    refresh_delta_threshold: float = 0.02


class RelationshipConfig(BaseModel):
    normal_delta_cap: float = 0.03
    major_delta_cap: float = 0.10


class GoalConfig(BaseModel):
    max_active_self: int = 3
    max_active_shared: int = 3
    min_project_days: int = 7
    urgency_priority_weight: float = 0.35
    urgency_due_weight: float = 0.20
    urgency_need_weight: float = 0.20
    urgency_opportunity_weight: float = 0.15
    urgency_continuity_weight: float = 0.10


class IdentityConfig(BaseModel):
    candidate_confidence_cap: float = 0.6
    active_confidence_threshold: float = 0.7
    active_min_age_days: int = 7


class AblationConfig(BaseModel):
    """Baseline switches. See pipeline §2.6."""

    baseline: str = "B4"
    enable_memory: bool = True
    enable_appraisal: bool = True
    enable_state: bool = True
    enable_relationship: bool = True
    enable_identity: bool = True
    enable_goals: bool = True
    enable_lifecycle: bool = True

    @field_validator("baseline")
    @classmethod
    def _check_baseline(cls, v: str) -> str:
        allowed = {"B0", "B1", "B2", "B3", "B4", "B5"}
        if v not in allowed:
            raise ValueError(f"baseline must be one of {allowed}")
        return v


class AppConfig(BaseModel):
    name: str = "ssa"
    timezone: str = "UTC"


class Settings(BaseModel):
    """Top-level settings object."""

    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    initiative: InitiativeConfig = Field(default_factory=InitiativeConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    relationship: RelationshipConfig = Field(default_factory=RelationshipConfig)
    goal: GoalConfig = Field(default_factory=GoalConfig)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    ablation: AblationConfig = Field(default_factory=AblationConfig)
    secrets: Secrets = Field(default_factory=Secrets)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SettingsError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `overlay` into `base` (returns a new dict)."""
    result: dict[str, Any] = {**base}
    for key, val in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def _env_overrides(env: dict[str, str]) -> dict[str, Any]:
    """Map selected SSA_* env vars into the settings tree."""
    overrides: dict[str, Any] = {}

    def set_section(section: str, key: str, value: Any) -> None:
        overrides.setdefault(section, {})[key] = value

    if "SSA_TIMEZONE" in env:
        set_section("app", "timezone", env["SSA_TIMEZONE"])
    if "SSA_LLM_MODEL" in env:
        set_section("llm", "model", env["SSA_LLM_MODEL"])
    if "SSA_LLM_TEMPERATURE" in env:
        set_section("llm", "temperature", float(env["SSA_LLM_TEMPERATURE"]))
    if "SSA_EMBEDDING_MODEL" in env:
        set_section("embedding", "model", env["SSA_EMBEDDING_MODEL"])
    if "SSA_EMBEDDING_DIM" in env:
        set_section("embedding", "dim", int(env["SSA_EMBEDDING_DIM"]))
    if "SSA_RETRIEVAL_CANDIDATES" in env:
        set_section("retrieval", "candidates", int(env["SSA_RETRIEVAL_CANDIDATES"]))
    if "SSA_RETRIEVAL_FINAL_K" in env:
        set_section("retrieval", "final_k", int(env["SSA_RETRIEVAL_FINAL_K"]))
    if "SSA_INITIATIVE_DAILY_LIMIT" in env:
        set_section("initiative", "daily_limit", int(env["SSA_INITIATIVE_DAILY_LIMIT"]))
    if "SSA_INITIATIVE_COOLDOWN_MINUTES" in env:
        set_section(
            "initiative", "cooldown_minutes", int(env["SSA_INITIATIVE_COOLDOWN_MINUTES"])
        )
    if "SSA_BACKGROUND_DAILY_LLM_BUDGET" in env:
        set_section(
            "budget",
            "background_daily_llm_budget",
            int(env["SSA_BACKGROUND_DAILY_LLM_BUDGET"]),
        )
    if "SSA_QUIET_HOURS_START" in env:
        set_section("initiative", "quiet_hours_start", env["SSA_QUIET_HOURS_START"])
    if "SSA_QUIET_HOURS_END" in env:
        set_section("initiative", "quiet_hours_end", env["SSA_QUIET_HOURS_END"])

    return overrides


def load_settings(
    env: Environment | str = Environment.DEVELOPMENT,
    *,
    config_dir: Path | None = None,
    environ: dict[str, str] | None = None,
) -> Settings:
    """Load settings following the documented precedence.

    Parameters
    ----------
    env:
        Which environment TOML overlay to apply.
    config_dir:
        Directory containing `defaults.toml` and `<env>.toml`.
        Defaults to `<package>/../../config`.
    environ:
        Environment mapping. Defaults to `os.environ`.
    """
    if isinstance(env, str):
        env = Environment(env)
    cfg_dir = config_dir or _CONFIG_DIR
    env_map = environ if environ is not None else dict(os.environ)

    # 1. defaults.toml
    merged = _load_toml(cfg_dir / "defaults.toml")
    # 2. <env>.toml overlay
    merged = _deep_merge(merged, _load_toml(cfg_dir / f"{env.value}.toml"))
    # 3. env var overrides
    merged = _deep_merge(merged, _env_overrides(env_map))

    # 4. secrets (env only)
    merged["secrets"] = Secrets.from_env(env_map).model_dump()

    return Settings.model_validate(merged)


__all__ = [
    "AblationConfig",
    "AppConfig",
    "BudgetConfig",
    "DatabaseConfig",
    "EmbeddingConfig",
    "Environment",
    "GoalConfig",
    "IdentityConfig",
    "InitiativeConfig",
    "LLMConfig",
    "RelationshipConfig",
    "RetrievalConfig",
    "Secrets",
    "Settings",
    "SettingsError",
    "StateConfig",
    "load_settings",
]
