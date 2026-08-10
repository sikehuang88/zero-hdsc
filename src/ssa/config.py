"""Configuration loading — layered, typed, secret-aware.

Pipeline §10.2 defines the precedence:

    CLI override  >  env vars / .env  >  config/<env>.toml  >  config/defaults.toml

Secrets (API keys, tokens, passwords) live ONLY in environment variables
and are never read from TOML files. Behavior/experiment parameters live in
version-controlled TOML and may be overridden by env vars for experiments.
"""

from __future__ import annotations

import math
import os
import re
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import dotenv_values
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


class Environment(StrEnum):
    DEVELOPMENT = "development"
    EVALUATION = "evaluation"
    PRODUCTION = "production"


class ThinkingMode(StrEnum):
    ENABLED = "enabled"
    DISABLED = "disabled"


class WarpedResonanceMode(StrEnum):
    DISABLED = "disabled"
    SHADOW = "shadow"
    LIVE = "live"


class EngramMode(StrEnum):
    TRUNCATED_POWER = "truncated_power"
    PPR = "ppr"
    BOUNDED_ACTIVE = "bounded_active"


class ReasoningEffort(StrEnum):
    HIGH = "high"
    MAX = "max"


# ---------------------------------------------------------------------------
# Secret bundle — sourced exclusively from environment variables.
# ---------------------------------------------------------------------------


class Secrets(BaseModel):
    """API keys and passwords. Never serialized to disk."""

    llm_api_key: SecretStr = SecretStr("")
    anthropic_auth_token: SecretStr = SecretStr("")
    multimodal_api_key: SecretStr = SecretStr("")
    firecrawl_api_key: SecretStr = SecretStr("")
    telegram_bot_token: SecretStr = SecretStr("")
    telegram_user_id: int = 0
    export_password: SecretStr = SecretStr("")
    doubao_api_key: SecretStr = SecretStr("")
    doubao_app_id: str = ""
    doubao_access_token: SecretStr = SecretStr("")
    doubao_secret_key: SecretStr = SecretStr("")
    opencode_server_password: SecretStr = SecretStr("")
    opencode_server_username: str = "opencode"
    broker_token: SecretStr = SecretStr("")

    @classmethod
    def from_env(cls, env: dict[str, str]) -> Secrets:
        return cls(
            llm_api_key=SecretStr(
                env.get("HDSC_LLM_API_KEY")
                or env.get("SSA_LLM_API_KEY")
                or env.get("DEEPSEEK_API_KEY", "")
            ),
            anthropic_auth_token=SecretStr(
                env.get("HDSC_ANTHROPIC_AUTH_TOKEN")
                or env.get("SSA_ANTHROPIC_AUTH_TOKEN")
                or env.get("ANTHROPIC_AUTH_TOKEN", "")
            ),
            multimodal_api_key=SecretStr(
                env.get("HDSC_MULTIMODAL_API_KEY")
                or env.get("SSA_MULTIMODAL_API_KEY")
                or env.get("MAGICAI_API_KEY", "")
            ),
            firecrawl_api_key=SecretStr(
                env.get("HDSC_FIRECRAWL_API_KEY") or env.get("FIRECRAWL_API_KEY", "")
            ),
            telegram_bot_token=SecretStr(
                env.get("HDSC_TELEGRAM_BOT_TOKEN") or env.get("SSA_TELEGRAM_BOT_TOKEN", "")
            ),
            telegram_user_id=int(
                env.get("HDSC_TELEGRAM_USER_ID") or env.get("SSA_TELEGRAM_USER_ID", "0") or "0"
            ),
            export_password=SecretStr(
                env.get("HDSC_EXPORT_PASSWORD") or env.get("SSA_EXPORT_PASSWORD", "")
            ),
            doubao_api_key=SecretStr(env.get("DOUBAO_API_KEY", "")),
            doubao_app_id=env.get("DOUBAO_APP_ID", ""),
            doubao_access_token=SecretStr(env.get("DOUBAO_ACCESS_TOKEN", "")),
            doubao_secret_key=SecretStr(env.get("DOUBAO_SECRET_KEY", "")),
            opencode_server_password=SecretStr(env.get("OPENCODE_SERVER_PASSWORD", "")),
            opencode_server_username=(env.get("OPENCODE_SERVER_USERNAME") or "opencode").strip()
            or "opencode",
            broker_token=SecretStr(env.get("HDSC_BROKER_TOKEN", "")),
        )

    def require_llm(self) -> str:
        v = self.llm_api_key.get_secret_value()
        if not v:
            raise SettingsError(
                "Missing required secret HDSC_LLM_API_KEY, legacy SSA_LLM_API_KEY, "
                "or DEEPSEEK_API_KEY. "
                "Set one in your .env file or environment."
            )
        return v

    def require_telegram(self) -> tuple[str, int]:
        token = self.telegram_bot_token.get_secret_value()
        if not token:
            raise SettingsError(
                "Missing required secret HDSC_TELEGRAM_BOT_TOKEN. "
                "Set it in your .env file or environment."
            )
        if self.telegram_user_id == 0:
            raise SettingsError(
                "Missing required secret HDSC_TELEGRAM_USER_ID. "
                "Set it in your .env file or environment."
            )
        return token, self.telegram_user_id


# ---------------------------------------------------------------------------
# Behavior / experiment parameter bundles — version-controlled.
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    model: str = "deepseek/deepseek-v4-flash"
    reasoning_model: str = "deepseek/deepseek-v4-pro"
    base_url: str = "https://api.deepseek.com"
    beta_base_url: str = "https://api.deepseek.com/beta"
    anthropic_base_url: str | None = None
    temperature: float | None = 0.8
    top_p: float | None = None
    max_tokens: int = 1024
    timeout_seconds: int = 120
    retry_count: int = 2
    json_retry_count: int = 1
    thinking_mode: ThinkingMode = ThinkingMode.DISABLED
    reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH
    user_id: str = "hdsc-primary"

    @field_validator("model", "reasoning_model")
    @classmethod
    def _check_model_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("model id must not be empty")
        return v

    @field_validator("base_url", "beta_base_url", "anthropic_base_url")
    @classmethod
    def _check_base_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        normalized = v.rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("LLM base URL must be an http(s) URL without query or fragment")
        return normalized

    @field_validator("temperature")
    @classmethod
    def _check_temp(cls, v: float | None) -> float | None:
        if v is None:
            return None
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        return v

    @field_validator("top_p")
    @classmethod
    def _check_top_p(cls, v: float | None) -> float | None:
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError("top_p must be in [0, 1]")
        return v

    @field_validator("max_tokens")
    @classmethod
    def _check_tokens(cls, v: int) -> int:
        if not 1 <= v <= 393_216:
            raise ValueError("max_tokens must be in [1, 393216]")
        return v

    @field_validator("timeout_seconds")
    @classmethod
    def _check_timeout(cls, v: int) -> int:
        if not 1 <= v <= 600:
            raise ValueError("timeout_seconds must be in [1, 600]")
        return v

    @field_validator("retry_count", "json_retry_count")
    @classmethod
    def _check_retry_count(cls, v: int) -> int:
        if not 0 <= v <= 10:
            raise ValueError("retry counts must be in [0, 10]")
        return v

    @field_validator("user_id")
    @classmethod
    def _check_user_id(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,512}", v):
            raise ValueError("user_id must match [A-Za-z0-9_-]{1,512}")
        return v

    @model_validator(mode="after")
    def _sampling_parameters_are_exclusive(self) -> LLMConfig:
        if self.temperature is not None and self.top_p is not None:
            raise ValueError("set temperature or top_p, not both")
        return self


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


class HDSCConfig(BaseModel):
    """HDSC shadow kernels and the evidence-gated resonance retriever."""

    h2_shadow_enabled: bool = True
    h2_active_capacity: int = 8
    h2_candidate_top_k: int = 4
    h2_cluster_bits: int = 16
    h2_retention: float = 0.80
    h2_injection_rate: float = 0.15
    h2_hysteresis: float = 0.02
    h2_score_lipschitz_bound: float = 1.0
    h2_gain_margin: float = 0.05
    h2_mass_tolerance: float = 1e-10
    resonance_enabled: bool = True
    resonance_hop_budget: int = 3
    resonance_emergence_ratio: float = 1.35
    resonance_background_floor: float = 0.02
    resonance_min_semantic_support: float = 0.12
    resonance_content_weight: float = 0.35
    resonance_affect_weight: float = 0.25
    resonance_relation_weight: float = 0.15
    resonance_situation_weight: float = 0.15
    resonance_arc_weight: float = 0.10
    warped_resonance_mode: WarpedResonanceMode = WarpedResonanceMode.LIVE
    warped_resonance_max_nodes: int = 32
    warped_low_information: float = 0.12
    warped_high_information: float = 0.45
    warped_beta: float = 4.0
    warped_bandwidth: float = 1.0
    warped_diffusion_time: float = 0.35
    warped_temperature: float = 0.15
    warped_recall_count: int = 5
    warped_detuning_cap: float = 6.0
    warped_surfacing_margin: float = 0.25

    @model_validator(mode="after")
    def _validate_h2(self) -> HDSCConfig:
        numeric = {
            "h2_retention": self.h2_retention,
            "h2_injection_rate": self.h2_injection_rate,
            "h2_hysteresis": self.h2_hysteresis,
            "h2_score_lipschitz_bound": self.h2_score_lipschitz_bound,
            "h2_gain_margin": self.h2_gain_margin,
            "h2_mass_tolerance": self.h2_mass_tolerance,
            "resonance_emergence_ratio": self.resonance_emergence_ratio,
            "resonance_background_floor": self.resonance_background_floor,
            "resonance_min_semantic_support": self.resonance_min_semantic_support,
            "resonance_content_weight": self.resonance_content_weight,
            "resonance_affect_weight": self.resonance_affect_weight,
            "resonance_relation_weight": self.resonance_relation_weight,
            "resonance_situation_weight": self.resonance_situation_weight,
            "resonance_arc_weight": self.resonance_arc_weight,
            "warped_low_information": self.warped_low_information,
            "warped_high_information": self.warped_high_information,
            "warped_beta": self.warped_beta,
            "warped_bandwidth": self.warped_bandwidth,
            "warped_diffusion_time": self.warped_diffusion_time,
            "warped_temperature": self.warped_temperature,
            "warped_detuning_cap": self.warped_detuning_cap,
            "warped_surfacing_margin": self.warped_surfacing_margin,
        }
        if any(not math.isfinite(value) for value in numeric.values()):
            raise ValueError("H2 numeric settings must be finite")
        if self.h2_active_capacity < 1:
            raise ValueError("h2_active_capacity must be positive")
        if not 1 <= self.h2_candidate_top_k <= self.h2_active_capacity:
            raise ValueError("h2_candidate_top_k must be in [1, h2_active_capacity]")
        if not 4 <= self.h2_cluster_bits <= 32:
            raise ValueError("h2_cluster_bits must be in [4, 32]")
        if not 0.0 <= self.h2_retention < 1.0:
            raise ValueError("h2_retention must be in [0, 1)")
        if not 0.0 <= self.h2_injection_rate <= 1.0:
            raise ValueError("h2_injection_rate must be in [0, 1]")
        if self.h2_retention + self.h2_injection_rate > 1.0:
            raise ValueError("h2_retention + h2_injection_rate must be <= 1")
        if self.h2_hysteresis < 0.0:
            raise ValueError("h2_hysteresis must be non-negative")
        if self.h2_score_lipschitz_bound <= 0.0:
            raise ValueError("h2_score_lipschitz_bound must be positive")
        if not 0.0 < self.h2_gain_margin < 1.0:
            raise ValueError("h2_gain_margin must be in (0, 1)")
        if self.h2_mass_tolerance <= 0.0:
            raise ValueError("h2_mass_tolerance must be positive")
        if not 1 <= self.resonance_hop_budget <= 8:
            raise ValueError("resonance_hop_budget must be in [1, 8]")
        if self.resonance_emergence_ratio <= 1.0:
            raise ValueError("resonance_emergence_ratio must exceed 1")
        if self.resonance_background_floor <= 0.0:
            raise ValueError("resonance_background_floor must be positive")
        if not 0.0 <= self.resonance_min_semantic_support <= 1.0:
            raise ValueError("resonance_min_semantic_support must be in [0, 1]")
        resonance_weights = (
            self.resonance_content_weight,
            self.resonance_affect_weight,
            self.resonance_relation_weight,
            self.resonance_situation_weight,
            self.resonance_arc_weight,
        )
        if any(value < 0.0 for value in resonance_weights) or sum(resonance_weights) <= 0.0:
            raise ValueError("resonance detuning weights must be non-negative with positive sum")
        if not 2 <= self.warped_resonance_max_nodes <= 256:
            raise ValueError("warped_resonance_max_nodes must be in [2, 256]")
        if not 0.0 <= self.warped_low_information < self.warped_high_information <= 1.0:
            raise ValueError("warped information thresholds must satisfy 0 <= low < high <= 1")
        if self.warped_beta < 0.0:
            raise ValueError("warped_beta must be non-negative")
        if (
            self.warped_bandwidth <= 0.0
            or self.warped_diffusion_time <= 0.0
            or self.warped_temperature <= 0.0
            or self.warped_detuning_cap <= 0.0
        ):
            raise ValueError("warped metric, temperature, and gate settings must be positive")
        if not 1 <= self.warped_recall_count <= self.warped_resonance_max_nodes:
            raise ValueError("warped_recall_count must be in [1, warped_resonance_max_nodes]")
        if self.warped_surfacing_margin < 0.0:
            raise ValueError("warped_surfacing_margin must be non-negative")
        return self


class EngramConfig(BaseModel):
    """Typed directed multi-hop retrieval experiment controls."""

    enabled: bool = False
    mode: EngramMode = EngramMode.TRUNCATED_POWER
    max_hops: int = 3
    restart_probability: float = 0.20
    epsilon: float = 1e-6
    max_active: int = 64
    max_results: int = 32
    graph_limit: int = 512

    @model_validator(mode="after")
    def _validate_engram(self) -> EngramConfig:
        if not 1 <= self.max_hops <= 32:
            raise ValueError("engram max_hops must be in [1, 32]")
        if not 0.0 < self.restart_probability <= 1.0:
            raise ValueError("engram restart_probability must be in (0, 1]")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("engram epsilon must be in [0, 1]")
        if not 1 <= self.max_active <= 4_096:
            raise ValueError("engram max_active must be in [1, 4096]")
        if not 1 <= self.max_results <= 256:
            raise ValueError("engram max_results must be in [1, 256]")
        if not 1 <= self.graph_limit <= 10_000:
            raise ValueError("engram graph_limit must be in [1, 10000]")
        return self


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

    @field_validator("cooldown_minutes", "min_gap_hours")
    @classmethod
    def _check_nonnegative_interval(cls, value: int) -> int:
        if value < 0:
            raise ValueError("initiative intervals must be non-negative")
        return value

    @field_validator("min_urgency")
    @classmethod
    def _check_urgency(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("min_urgency must be in [0, 1]")
        return v

    @field_validator("quiet_hours_start", "quiet_hours_end")
    @classmethod
    def _check_clock(cls, value: str) -> str:
        if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
            raise ValueError("quiet hours must use valid HH:MM values")
        return value


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
    # Keep the legacy filename so an upgrade never silently starts with an empty identity.
    path: str = "data/ssa.db"
    busy_timeout_ms: int = 5000


class StateConfig(BaseModel):
    valence_decay: float = 0.75
    valence_signal_weight: float = 0.25
    arousal_signal_weight: float = 0.30
    arousal_decay_per_hour: float = 0.10
    refresh_delta_threshold: float = 0.02


class PerceptionConfig(BaseModel):
    """Deterministic P1 environment-perception operator parameters."""

    context_window_events: int = 16
    fresh_gap_minutes: int = 15
    long_gap_hours: float = 6.0
    mode_temperature: float = 0.30

    @model_validator(mode="after")
    def _validate_perception(self) -> PerceptionConfig:
        if self.context_window_events < 2:
            raise ValueError("context_window_events must be at least 2")
        if self.fresh_gap_minutes < 1:
            raise ValueError("fresh_gap_minutes must be positive")
        if self.long_gap_hours <= self.fresh_gap_minutes / 60.0:
            raise ValueError("long_gap_hours must exceed fresh_gap_minutes")
        if not 0.05 <= self.mode_temperature <= 2.0:
            raise ValueError("mode_temperature must be in [0.05, 2]")
        return self


class WorldConfig(BaseModel):
    """Low-frequency host observations accepted by the autonomous runtime."""

    enabled: bool = True
    observe_interval_seconds: int = 300
    watched_paths: list[str] = Field(default_factory=list)
    calendar_json_paths: list[str] = Field(default_factory=list)
    observe_foreground_window: bool = False
    material_salience: float = 0.60

    @model_validator(mode="after")
    def _validate_world(self) -> WorldConfig:
        if self.observe_interval_seconds < 5:
            raise ValueError("world observe_interval_seconds must be at least 5")
        if not 0.0 <= self.material_salience <= 1.0:
            raise ValueError("world material_salience must be in [0, 1]")
        return self


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
    review_interval_agent_events: int = 4
    review_window_events: int = 24

    @field_validator("review_interval_agent_events")
    @classmethod
    def _check_review_interval(cls, value: int) -> int:
        if value < 1:
            raise ValueError("review_interval_agent_events must be positive")
        return value

    @field_validator("review_window_events")
    @classmethod
    def _check_review_window(cls, value: int) -> int:
        if value < 2:
            raise ValueError("review_window_events must be at least 2")
        return value


class ToolConfig(BaseModel):
    enabled: bool = True
    max_rounds: int = 8
    default_timeout_seconds: int = 60
    max_output_chars: int = 32_000
    allow_elevation: bool = True
    powershell_executable: str = "powershell.exe"

    @field_validator("max_rounds")
    @classmethod
    def _check_tool_rounds(cls, value: int) -> int:
        if not 1 <= value <= 8:
            raise ValueError("tool max_rounds must be in [1, 8]")
        return value

    @field_validator("default_timeout_seconds")
    @classmethod
    def _check_tool_timeout(cls, value: int) -> int:
        if not 1 <= value <= 600:
            raise ValueError("tool default_timeout_seconds must be in [1, 600]")
        return value

    @field_validator("max_output_chars")
    @classmethod
    def _check_tool_output_limit(cls, value: int) -> int:
        if not 1_024 <= value <= 1_000_000:
            raise ValueError("tool max_output_chars must be in [1024, 1000000]")
        return value

    @field_validator("powershell_executable")
    @classmethod
    def _check_powershell_executable(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("powershell_executable must not be empty")
        return value


class Win32Config(BaseModel):
    """Read-only native Windows observation capability limits."""

    enabled: bool = True
    clipboard_enabled: bool = True
    max_windows: int = 64
    max_processes: int = 128
    max_directory_entries: int = 256
    max_clipboard_chars: int = 16_000

    @model_validator(mode="after")
    def _validate_win32(self) -> Win32Config:
        if not 1 <= self.max_windows <= 512:
            raise ValueError("win32 max_windows must be in [1, 512]")
        if not 1 <= self.max_processes <= 4_096:
            raise ValueError("win32 max_processes must be in [1, 4096]")
        if not 1 <= self.max_directory_entries <= 10_000:
            raise ValueError("win32 max_directory_entries must be in [1, 10000]")
        if not 128 <= self.max_clipboard_chars <= 1_000_000:
            raise ValueError("win32 max_clipboard_chars must be in [128, 1000000]")
        return self


class ExternalTruthConfig(BaseModel):
    """Network boundary for provider-reported, keyless external observations."""

    enabled: bool = True
    enabled_sources: list[str] = Field(
        default_factory=lambda: [
            "weather",
            "sun_times",
            "holidays",
            "geocode",
            "news",
            "radio",
            "music",
            "translate",
            "service_status",
        ]
    )
    request_timeout_seconds: int = 12
    max_response_bytes: int = 524_288
    cache_ttl_seconds: int = 300
    min_request_interval_ms: int = 1_100
    user_agent: str = "HDSC/0.5 external-truth"

    @model_validator(mode="after")
    def _validate_external_truth(self) -> ExternalTruthConfig:
        allowed = {
            "weather",
            "sun_times",
            "holidays",
            "geocode",
            "news",
            "radio",
            "music",
            "translate",
            "service_status",
        }
        unknown = set(self.enabled_sources) - allowed
        if unknown:
            raise ValueError(f"unknown external truth sources: {sorted(unknown)}")
        if len(self.enabled_sources) != len(set(self.enabled_sources)):
            raise ValueError("external truth sources must be unique")
        if not 1 <= self.request_timeout_seconds <= 60:
            raise ValueError("external truth timeout must be in [1, 60]")
        if not 4_096 <= self.max_response_bytes <= 4_194_304:
            raise ValueError("external truth max_response_bytes must be in [4096, 4194304]")
        if not 0 <= self.cache_ttl_seconds <= 86_400:
            raise ValueError("external truth cache_ttl_seconds must be in [0, 86400]")
        if not 0 <= self.min_request_interval_ms <= 60_000:
            raise ValueError("external truth min_request_interval_ms must be in [0, 60000]")
        if not self.user_agent.strip():
            raise ValueError("external truth user_agent must not be empty")
        return self


class FirecrawlConfig(BaseModel):
    """Firecrawl live-web search and scrape boundary."""

    enabled: bool = True
    base_url: str = "https://api.firecrawl.dev/v2"
    request_timeout_seconds: int = 60
    max_response_bytes: int = 4_194_304
    max_result_content_chars: int = 8_000

    @field_validator("base_url")
    @classmethod
    def _normalize_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("firecrawl base URL must be an absolute HTTP(S) URL")
        return normalized

    @model_validator(mode="after")
    def _validate_firecrawl(self) -> FirecrawlConfig:
        if not 1 <= self.request_timeout_seconds <= 180:
            raise ValueError("firecrawl timeout must be in [1, 180]")
        if not 4_096 <= self.max_response_bytes <= 16_777_216:
            raise ValueError("firecrawl max_response_bytes must be in [4096, 16777216]")
        if not 500 <= self.max_result_content_chars <= 32_000:
            raise ValueError("firecrawl max_result_content_chars must be in [500, 32000]")
        return self


class MultimodalConfig(BaseModel):
    """Responses API boundary for image and multi-file understanding."""

    enabled: bool = True
    model: str = "gpt-5.6-luna"
    base_url: str = "https://sky1818.com/v1"
    reasoning_effort: str = "medium"
    timeout_seconds: int = 180
    max_files: int = 8
    max_file_bytes: int = 20_971_520
    max_total_bytes: int = 41_943_040
    max_text_chars_per_file: int = 120_000
    max_response_bytes: int = 4_194_304

    @model_validator(mode="after")
    def _validate_multimodal(self) -> MultimodalConfig:
        parsed = urlsplit(self.base_url.rstrip("/"))
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("multimodal base URL must be an http(s) URL without query or fragment")
        self.base_url = self.base_url.rstrip("/")
        if not self.model.strip():
            raise ValueError("multimodal model must not be empty")
        if self.reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("multimodal reasoning_effort must be low, medium, or high")
        if not 1 <= self.timeout_seconds <= 600:
            raise ValueError("multimodal timeout_seconds must be in [1, 600]")
        if not 1 <= self.max_files <= 32:
            raise ValueError("multimodal max_files must be in [1, 32]")
        if not 1_024 <= self.max_file_bytes <= 104_857_600:
            raise ValueError("multimodal max_file_bytes must be in [1024, 104857600]")
        if not self.max_file_bytes <= self.max_total_bytes <= 524_288_000:
            raise ValueError(
                "multimodal max_total_bytes must cover one file and stay under 500 MiB"
            )
        if not 1_024 <= self.max_text_chars_per_file <= 1_000_000:
            raise ValueError("multimodal max_text_chars_per_file must be in [1024, 1000000]")
        if not 4_096 <= self.max_response_bytes <= 16_777_216:
            raise ValueError("multimodal max_response_bytes must be in [4096, 16777216]")
        return self


class OpencodeConfig(BaseModel):
    """Bounded client policy for a local headless opencode service."""

    enabled: bool = False
    base_url: str = "http://127.0.0.1:4096"
    project_dir: str = ""
    request_timeout_seconds: int = 120
    job_timeout_seconds: int = 3_600
    job_retention_seconds: int = 86_400
    job_store_path: str = "data/opencode_jobs.json"
    fire_max_concurrent: int = 2
    poll_max_output_chars: int = 24_000
    default_model: str = "hdsc/deepseek-v4-flash"

    @field_validator("base_url", "default_model")
    @classmethod
    def _opencode_value_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("opencode config value must not be blank")
        return normalized.rstrip("/") if "://" in normalized else normalized

    @field_validator("request_timeout_seconds")
    @classmethod
    def _opencode_timeout_bounded(cls, value: int) -> int:
        if not 1 <= value <= 600:
            raise ValueError("opencode request_timeout_seconds must be in [1, 600]")
        return value

    @field_validator("job_timeout_seconds")
    @classmethod
    def _opencode_job_timeout_bounded(cls, value: int) -> int:
        if not 60 <= value <= 86_400:
            raise ValueError("opencode job_timeout_seconds must be in [60, 86400]")
        return value

    @field_validator("job_retention_seconds")
    @classmethod
    def _opencode_job_retention_bounded(cls, value: int) -> int:
        if not 60 <= value <= 604_800:
            raise ValueError("opencode job_retention_seconds must be in [60, 604800]")
        return value

    @field_validator("job_store_path")
    @classmethod
    def _opencode_job_store_path_normalized(cls, value: str) -> str:
        return value.strip()

    @field_validator("fire_max_concurrent")
    @classmethod
    def _opencode_concurrency_bounded(cls, value: int) -> int:
        if not 1 <= value <= 8:
            raise ValueError("opencode fire_max_concurrent must be in [1, 8]")
        return value

    @field_validator("poll_max_output_chars")
    @classmethod
    def _opencode_output_bounded(cls, value: int) -> int:
        if not 1_000 <= value <= 32_000:
            raise ValueError("opencode poll_max_output_chars must be in [1000, 32000]")
        return value


class OfflineAgencyConfig(BaseModel):
    """Bounded offline learning cycles that run while a worker process is alive."""

    enabled: bool = True
    interval_minutes: int = 30
    min_user_absence_minutes: int = 20
    max_cycles_per_day: int = 12
    external_observation_every_cycles: int = 4
    reflection_max_tokens: int = 480
    artifact_max_chars: int = 4_000
    news_category: str = "ai"
    min_energy: float = 0.35

    @model_validator(mode="after")
    def _validate_offline_agency(self) -> OfflineAgencyConfig:
        if not 5 <= self.interval_minutes <= 1_440:
            raise ValueError("offline interval_minutes must be in [5, 1440]")
        if not 1 <= self.min_user_absence_minutes <= 10_080:
            raise ValueError("offline min_user_absence_minutes must be in [1, 10080]")
        if not 1 <= self.max_cycles_per_day <= 96:
            raise ValueError("offline max_cycles_per_day must be in [1, 96]")
        if not 1 <= self.external_observation_every_cycles <= 96:
            raise ValueError("offline external observation interval must be in [1, 96]")
        if not 64 <= self.reflection_max_tokens <= 2_048:
            raise ValueError("offline reflection_max_tokens must be in [64, 2048]")
        if not 256 <= self.artifact_max_chars <= 32_000:
            raise ValueError("offline artifact_max_chars must be in [256, 32000]")
        if self.news_category not in {
            "ai",
            "business",
            "culture",
            "entertainment",
            "finance",
            "general",
            "health",
            "lifestyle",
            "opinion",
            "politics",
            "science",
            "sports",
            "tech",
            "weather",
            "world",
        }:
            raise ValueError("unknown offline news_category")
        if not 0.0 <= self.min_energy <= 1.0:
            raise ValueError("offline min_energy must be in [0, 1]")
        return self


class InnerLifeConfig(BaseModel):
    """Contextual waiting and bounded hidden-state heartbeat parameters."""

    enabled: bool = True
    heartbeat_seconds: int = 60
    min_wait_minutes: int = 10
    base_wait_minutes: int = 45
    max_wait_minutes: int = 720
    reply_half_life_minutes: int = 120
    affect_half_life_minutes: int = 180
    offline_entry_threshold: float = 0.58
    offline_exit_threshold: float = 0.42
    min_offline_energy: float = 0.30

    @model_validator(mode="after")
    def _validate_inner_life(self) -> InnerLifeConfig:
        if not 10 <= self.heartbeat_seconds <= 3_600:
            raise ValueError("inner-life heartbeat_seconds must be in [10, 3600]")
        if not 1 <= self.min_wait_minutes <= self.base_wait_minutes:
            raise ValueError("inner-life min_wait_minutes must be in [1, base_wait_minutes]")
        if not self.base_wait_minutes <= self.max_wait_minutes <= 10_080:
            raise ValueError("inner-life max_wait_minutes must be in [base_wait_minutes, 10080]")
        if self.reply_half_life_minutes < 1 or self.affect_half_life_minutes < 1:
            raise ValueError("inner-life half-lives must be positive")
        if not 0.0 <= self.offline_exit_threshold < self.offline_entry_threshold <= 1.0:
            raise ValueError("inner-life offline thresholds must satisfy 0 <= exit < entry <= 1")
        if not 0.0 <= self.min_offline_energy <= 1.0:
            raise ValueError("inner-life min_offline_energy must be in [0, 1]")
        return self


class EmotionLibraryConfig(BaseModel):
    """Long-term subjective emotional-memory capture and recall policy."""

    enabled: bool = True
    min_intensity: float = 0.35
    recall_limit: int = 6
    candidate_limit: int = 240
    recency_half_life_days: float = 45.0
    backfill_limit: int = 500
    library_root: str = "emotion-value-library"
    expression_library_path: str | None = None
    expression_entry_limit: int = 5
    expression_scene_limit: int = 2

    @model_validator(mode="after")
    def _validate_emotion_library(self) -> EmotionLibraryConfig:
        if not 0.0 <= self.min_intensity <= 1.0:
            raise ValueError("emotion-library min_intensity must be in [0, 1]")
        if not 1 <= self.recall_limit <= 16:
            raise ValueError("emotion-library recall_limit must be in [1, 16]")
        if not self.recall_limit <= self.candidate_limit <= 2_000:
            raise ValueError("emotion-library candidate_limit must cover recall_limit")
        if not 1.0 <= self.recency_half_life_days <= 3_650.0:
            raise ValueError("emotion-library recency_half_life_days must be in [1, 3650]")
        if not 0 <= self.backfill_limit <= 10_000:
            raise ValueError("emotion-library backfill_limit must be in [0, 10000]")
        if not self.library_root.strip():
            raise ValueError("emotion-library library_root must not be empty")
        if self.expression_library_path is not None and not self.expression_library_path.strip():
            raise ValueError("emotion-library expression_library_path must not be empty")
        if not 1 <= self.expression_entry_limit <= 12:
            raise ValueError("emotion-library expression_entry_limit must be in [1, 12]")
        if not 0 <= self.expression_scene_limit <= 6:
            raise ValueError("emotion-library expression_scene_limit must be in [0, 6]")
        return self

    @property
    def effective_library_root(self) -> str:
        """Configured root, preserving the pre-v0.5 field as a compatibility alias."""
        return self.expression_library_path or self.library_root


class ReflectiveLearningConfig(BaseModel):
    """Evidence gates and cadence for the continuous reflective-learning loop."""

    enabled: bool = True
    schedule_interval_minutes: int = 15
    consolidation_interval_minutes: int = 15
    outcome_interval_minutes: int = 60
    min_reflection_priority: float = 0.35
    min_critic_score: float = 0.62
    min_evidence_count: int = 1
    experiment_duration_hours: int = 6
    experiment_min_observations: int = 1
    min_outcome_confidence: float = 1.2
    max_runs_per_cycle: int = 4
    max_proposals_per_cycle: int = 8
    policy_experiments_enabled: bool = True

    @model_validator(mode="after")
    def _validate_reflective_learning(self) -> ReflectiveLearningConfig:
        intervals = (
            self.schedule_interval_minutes,
            self.consolidation_interval_minutes,
            self.outcome_interval_minutes,
        )
        if any(not 1 <= value <= 1_440 for value in intervals):
            raise ValueError("reflective-learning intervals must be in [1, 1440]")
        if not 0.0 <= self.min_reflection_priority <= 1.0:
            raise ValueError("min_reflection_priority must be in [0, 1]")
        if not 0.0 <= self.min_critic_score <= 1.0:
            raise ValueError("min_critic_score must be in [0, 1]")
        if not 1 <= self.min_evidence_count <= 16:
            raise ValueError("min_evidence_count must be in [1, 16]")
        if not 1 <= self.experiment_duration_hours <= 720:
            raise ValueError("experiment_duration_hours must be in [1, 720]")
        if not 1 <= self.experiment_min_observations <= 100:
            raise ValueError("experiment_min_observations must be in [1, 100]")
        if not 0.1 <= self.min_outcome_confidence <= 100.0:
            raise ValueError("min_outcome_confidence must be in [0.1, 100]")
        if not 1 <= self.max_runs_per_cycle <= 32:
            raise ValueError("max_runs_per_cycle must be in [1, 32]")
        if not 1 <= self.max_proposals_per_cycle <= 64:
            raise ValueError("max_proposals_per_cycle must be in [1, 64]")
        return self


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
    name: str = "hdsc"
    timezone: str = "UTC"

    @field_validator("timezone")
    @classmethod
    def _timezone_must_exist(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown IANA timezone: {value}") from exc
        return value


class Settings(BaseModel):
    """Top-level settings object."""

    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    hdsc: HDSCConfig = Field(default_factory=HDSCConfig)
    engram: EngramConfig = Field(default_factory=EngramConfig)
    initiative: InitiativeConfig = Field(default_factory=InitiativeConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    world: WorldConfig = Field(default_factory=WorldConfig)
    relationship: RelationshipConfig = Field(default_factory=RelationshipConfig)
    goal: GoalConfig = Field(default_factory=GoalConfig)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    win32: Win32Config = Field(default_factory=Win32Config)
    external_truth: ExternalTruthConfig = Field(default_factory=ExternalTruthConfig)
    firecrawl: FirecrawlConfig = Field(default_factory=FirecrawlConfig)
    multimodal: MultimodalConfig = Field(default_factory=MultimodalConfig)
    opencode: OpencodeConfig = Field(default_factory=OpencodeConfig)
    offline_agency: OfflineAgencyConfig = Field(default_factory=OfflineAgencyConfig)
    inner_life: InnerLifeConfig = Field(default_factory=InnerLifeConfig)
    emotion_library: EmotionLibraryConfig = Field(default_factory=EmotionLibraryConfig)
    reflective_learning: ReflectiveLearningConfig = Field(default_factory=ReflectiveLearningConfig)
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

_SOURCE_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def _default_config_dir() -> Path:
    working_tree_config = Path.cwd() / "config"
    if (working_tree_config / "defaults.toml").exists():
        return working_tree_config
    if (_SOURCE_CONFIG_DIR / "defaults.toml").exists():
        return _SOURCE_CONFIG_DIR
    return working_tree_config


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
    """Map HDSC_* vars and their legacy SSA_* aliases into settings."""
    normalized_env = dict(env)
    for key, value in env.items():
        if key.startswith("HDSC_"):
            normalized_env[f"SSA_{key.removeprefix('HDSC_')}"] = value
    env = normalized_env
    overrides: dict[str, Any] = {}

    def set_section(section: str, key: str, value: Any) -> None:
        overrides.setdefault(section, {})[key] = value

    if "SSA_TIMEZONE" in env:
        set_section("app", "timezone", env["SSA_TIMEZONE"])
    if "SSA_LLM_MODEL" in env:
        set_section("llm", "model", env["SSA_LLM_MODEL"])
    if "SSA_LLM_REASONING_MODEL" in env:
        set_section("llm", "reasoning_model", env["SSA_LLM_REASONING_MODEL"])
    if "SSA_LLM_BASE_URL" in env:
        set_section("llm", "base_url", env["SSA_LLM_BASE_URL"])
    if "SSA_LLM_BETA_BASE_URL" in env:
        set_section("llm", "beta_base_url", env["SSA_LLM_BETA_BASE_URL"])
    if "SSA_LLM_ANTHROPIC_BASE_URL" in env:
        set_section("llm", "anthropic_base_url", env["SSA_LLM_ANTHROPIC_BASE_URL"])
    elif "ANTHROPIC_BASE_URL" in env:
        set_section("llm", "anthropic_base_url", env["ANTHROPIC_BASE_URL"])
    if "SSA_LLM_TEMPERATURE" in env:
        set_section("llm", "temperature", float(env["SSA_LLM_TEMPERATURE"]))
    if "SSA_LLM_TOP_P" in env:
        set_section("llm", "top_p", float(env["SSA_LLM_TOP_P"]))
        if "SSA_LLM_TEMPERATURE" not in env:
            set_section("llm", "temperature", None)
    if "SSA_LLM_MAX_TOKENS" in env:
        set_section("llm", "max_tokens", int(env["SSA_LLM_MAX_TOKENS"]))
    if "SSA_LLM_TIMEOUT_SECONDS" in env:
        set_section("llm", "timeout_seconds", int(env["SSA_LLM_TIMEOUT_SECONDS"]))
    if "SSA_LLM_RETRY_COUNT" in env:
        set_section("llm", "retry_count", int(env["SSA_LLM_RETRY_COUNT"]))
    if "SSA_LLM_JSON_RETRY_COUNT" in env:
        set_section("llm", "json_retry_count", int(env["SSA_LLM_JSON_RETRY_COUNT"]))
    if "SSA_LLM_THINKING_MODE" in env:
        set_section("llm", "thinking_mode", env["SSA_LLM_THINKING_MODE"])
    if "SSA_LLM_REASONING_EFFORT" in env:
        set_section("llm", "reasoning_effort", env["SSA_LLM_REASONING_EFFORT"])
    if "SSA_DEEPSEEK_USER_ID" in env:
        set_section("llm", "user_id", env["SSA_DEEPSEEK_USER_ID"])
    if "SSA_EMBEDDING_MODEL" in env:
        set_section("embedding", "model", env["SSA_EMBEDDING_MODEL"])
    if "SSA_EMBEDDING_DIM" in env:
        set_section("embedding", "dim", int(env["SSA_EMBEDDING_DIM"]))
    if "SSA_RETRIEVAL_CANDIDATES" in env:
        set_section("retrieval", "candidates", int(env["SSA_RETRIEVAL_CANDIDATES"]))
    if "SSA_RETRIEVAL_FINAL_K" in env:
        set_section("retrieval", "final_k", int(env["SSA_RETRIEVAL_FINAL_K"]))
    if "SSA_H2_SHADOW_ENABLED" in env:
        set_section("hdsc", "h2_shadow_enabled", _parse_bool(env["SSA_H2_SHADOW_ENABLED"]))
    if "SSA_H2_ACTIVE_CAPACITY" in env:
        set_section("hdsc", "h2_active_capacity", int(env["SSA_H2_ACTIVE_CAPACITY"]))
    if "SSA_H2_CANDIDATE_TOP_K" in env:
        set_section("hdsc", "h2_candidate_top_k", int(env["SSA_H2_CANDIDATE_TOP_K"]))
    if "SSA_H2_CLUSTER_BITS" in env:
        set_section("hdsc", "h2_cluster_bits", int(env["SSA_H2_CLUSTER_BITS"]))
    if "SSA_H2_RETENTION" in env:
        set_section("hdsc", "h2_retention", float(env["SSA_H2_RETENTION"]))
    if "SSA_H2_INJECTION_RATE" in env:
        set_section("hdsc", "h2_injection_rate", float(env["SSA_H2_INJECTION_RATE"]))
    if "SSA_H2_HYSTERESIS" in env:
        set_section("hdsc", "h2_hysteresis", float(env["SSA_H2_HYSTERESIS"]))
    if "SSA_H2_SCORE_LIPSCHITZ_BOUND" in env:
        set_section(
            "hdsc",
            "h2_score_lipschitz_bound",
            float(env["SSA_H2_SCORE_LIPSCHITZ_BOUND"]),
        )
    if "SSA_H2_GAIN_MARGIN" in env:
        set_section("hdsc", "h2_gain_margin", float(env["SSA_H2_GAIN_MARGIN"]))
    if "SSA_H2_MASS_TOLERANCE" in env:
        set_section("hdsc", "h2_mass_tolerance", float(env["SSA_H2_MASS_TOLERANCE"]))
    if "HDSC_ENGRAM_ENABLED" in env:
        set_section("engram", "enabled", _parse_bool(env["HDSC_ENGRAM_ENABLED"]))
    if "HDSC_ENGRAM_MODE" in env:
        set_section("engram", "mode", env["HDSC_ENGRAM_MODE"])
    if "HDSC_ENGRAM_MAX_HOPS" in env:
        set_section("engram", "max_hops", int(env["HDSC_ENGRAM_MAX_HOPS"]))
    if "HDSC_ENGRAM_RESTART_PROBABILITY" in env:
        set_section(
            "engram",
            "restart_probability",
            float(env["HDSC_ENGRAM_RESTART_PROBABILITY"]),
        )
    if "HDSC_ENGRAM_MAX_ACTIVE" in env:
        set_section("engram", "max_active", int(env["HDSC_ENGRAM_MAX_ACTIVE"]))
    if "HDSC_ENGRAM_MAX_RESULTS" in env:
        set_section("engram", "max_results", int(env["HDSC_ENGRAM_MAX_RESULTS"]))
    if "HDSC_ENGRAM_GRAPH_LIMIT" in env:
        set_section("engram", "graph_limit", int(env["HDSC_ENGRAM_GRAPH_LIMIT"]))
    if "SSA_INITIATIVE_DAILY_LIMIT" in env:
        set_section("initiative", "daily_limit", int(env["SSA_INITIATIVE_DAILY_LIMIT"]))
    if "SSA_INITIATIVE_COOLDOWN_MINUTES" in env:
        set_section("initiative", "cooldown_minutes", int(env["SSA_INITIATIVE_COOLDOWN_MINUTES"]))
    if "SSA_BACKGROUND_DAILY_LLM_BUDGET" in env:
        set_section(
            "budget",
            "background_daily_llm_budget",
            int(env["SSA_BACKGROUND_DAILY_LLM_BUDGET"]),
        )
    if "SSA_IDENTITY_REVIEW_INTERVAL_AGENT_EVENTS" in env:
        set_section(
            "identity",
            "review_interval_agent_events",
            int(env["SSA_IDENTITY_REVIEW_INTERVAL_AGENT_EVENTS"]),
        )
    if "SSA_IDENTITY_REVIEW_WINDOW_EVENTS" in env:
        set_section(
            "identity",
            "review_window_events",
            int(env["SSA_IDENTITY_REVIEW_WINDOW_EVENTS"]),
        )
    if "SSA_MIN_OUTCOME_CONFIDENCE" in env:
        set_section(
            "reflective_learning",
            "min_outcome_confidence",
            float(env["SSA_MIN_OUTCOME_CONFIDENCE"]),
        )
    if "SSA_TOOLS_ENABLED" in env:
        set_section("tools", "enabled", _parse_bool(env["SSA_TOOLS_ENABLED"]))
    if "SSA_TOOLS_MAX_ROUNDS" in env:
        set_section("tools", "max_rounds", int(env["SSA_TOOLS_MAX_ROUNDS"]))
    if "SSA_TOOLS_DEFAULT_TIMEOUT_SECONDS" in env:
        set_section(
            "tools",
            "default_timeout_seconds",
            int(env["SSA_TOOLS_DEFAULT_TIMEOUT_SECONDS"]),
        )
    if "SSA_TOOLS_MAX_OUTPUT_CHARS" in env:
        set_section("tools", "max_output_chars", int(env["SSA_TOOLS_MAX_OUTPUT_CHARS"]))
    if "SSA_TOOLS_ALLOW_ELEVATION" in env:
        set_section(
            "tools",
            "allow_elevation",
            _parse_bool(env["SSA_TOOLS_ALLOW_ELEVATION"]),
        )
    if "SSA_WIN32_ENABLED" in env:
        set_section("win32", "enabled", _parse_bool(env["SSA_WIN32_ENABLED"]))
    if "SSA_WIN32_CLIPBOARD_ENABLED" in env:
        set_section(
            "win32",
            "clipboard_enabled",
            _parse_bool(env["SSA_WIN32_CLIPBOARD_ENABLED"]),
        )
    if "SSA_WIN32_MAX_WINDOWS" in env:
        set_section("win32", "max_windows", int(env["SSA_WIN32_MAX_WINDOWS"]))
    if "SSA_WIN32_MAX_PROCESSES" in env:
        set_section("win32", "max_processes", int(env["SSA_WIN32_MAX_PROCESSES"]))
    if "SSA_WIN32_MAX_DIRECTORY_ENTRIES" in env:
        set_section(
            "win32",
            "max_directory_entries",
            int(env["SSA_WIN32_MAX_DIRECTORY_ENTRIES"]),
        )
    if "SSA_WIN32_MAX_CLIPBOARD_CHARS" in env:
        set_section(
            "win32",
            "max_clipboard_chars",
            int(env["SSA_WIN32_MAX_CLIPBOARD_CHARS"]),
        )
    if "SSA_EXTERNAL_TRUTH_ENABLED" in env:
        set_section(
            "external_truth",
            "enabled",
            _parse_bool(env["SSA_EXTERNAL_TRUTH_ENABLED"]),
        )
    if "SSA_EXTERNAL_TRUTH_SOURCES" in env:
        set_section(
            "external_truth",
            "enabled_sources",
            [item.strip() for item in env["SSA_EXTERNAL_TRUTH_SOURCES"].split(",") if item.strip()],
        )
    if "SSA_EXTERNAL_TRUTH_TIMEOUT_SECONDS" in env:
        set_section(
            "external_truth",
            "request_timeout_seconds",
            int(env["SSA_EXTERNAL_TRUTH_TIMEOUT_SECONDS"]),
        )
    if "SSA_EXTERNAL_TRUTH_CACHE_TTL_SECONDS" in env:
        set_section(
            "external_truth",
            "cache_ttl_seconds",
            int(env["SSA_EXTERNAL_TRUTH_CACHE_TTL_SECONDS"]),
        )
    if "SSA_EXTERNAL_TRUTH_MAX_RESPONSE_BYTES" in env:
        set_section(
            "external_truth",
            "max_response_bytes",
            int(env["SSA_EXTERNAL_TRUTH_MAX_RESPONSE_BYTES"]),
        )
    if "SSA_EXTERNAL_TRUTH_MIN_REQUEST_INTERVAL_MS" in env:
        set_section(
            "external_truth",
            "min_request_interval_ms",
            int(env["SSA_EXTERNAL_TRUTH_MIN_REQUEST_INTERVAL_MS"]),
        )
    if "SSA_FIRECRAWL_ENABLED" in env:
        set_section("firecrawl", "enabled", _parse_bool(env["SSA_FIRECRAWL_ENABLED"]))
    if "SSA_FIRECRAWL_BASE_URL" in env:
        set_section("firecrawl", "base_url", env["SSA_FIRECRAWL_BASE_URL"])
    if "SSA_FIRECRAWL_TIMEOUT_SECONDS" in env:
        set_section(
            "firecrawl",
            "request_timeout_seconds",
            int(env["SSA_FIRECRAWL_TIMEOUT_SECONDS"]),
        )
    if "SSA_FIRECRAWL_MAX_RESPONSE_BYTES" in env:
        set_section(
            "firecrawl",
            "max_response_bytes",
            int(env["SSA_FIRECRAWL_MAX_RESPONSE_BYTES"]),
        )
    if "SSA_FIRECRAWL_MAX_RESULT_CONTENT_CHARS" in env:
        set_section(
            "firecrawl",
            "max_result_content_chars",
            int(env["SSA_FIRECRAWL_MAX_RESULT_CONTENT_CHARS"]),
        )
    if "SSA_MULTIMODAL_ENABLED" in env:
        set_section("multimodal", "enabled", _parse_bool(env["SSA_MULTIMODAL_ENABLED"]))
    if "SSA_MULTIMODAL_MODEL" in env:
        set_section("multimodal", "model", env["SSA_MULTIMODAL_MODEL"])
    if "SSA_MULTIMODAL_BASE_URL" in env:
        set_section("multimodal", "base_url", env["SSA_MULTIMODAL_BASE_URL"])
    if "SSA_MULTIMODAL_REASONING_EFFORT" in env:
        set_section("multimodal", "reasoning_effort", env["SSA_MULTIMODAL_REASONING_EFFORT"])
    if "SSA_MULTIMODAL_TIMEOUT_SECONDS" in env:
        set_section("multimodal", "timeout_seconds", int(env["SSA_MULTIMODAL_TIMEOUT_SECONDS"]))
    if "SSA_MULTIMODAL_MAX_FILES" in env:
        set_section("multimodal", "max_files", int(env["SSA_MULTIMODAL_MAX_FILES"]))
    if "SSA_MULTIMODAL_MAX_FILE_BYTES" in env:
        set_section("multimodal", "max_file_bytes", int(env["SSA_MULTIMODAL_MAX_FILE_BYTES"]))
    if "SSA_MULTIMODAL_MAX_TOTAL_BYTES" in env:
        set_section("multimodal", "max_total_bytes", int(env["SSA_MULTIMODAL_MAX_TOTAL_BYTES"]))
    if "HDSC_OPENCODE_ENABLED" in env:
        set_section("opencode", "enabled", _parse_bool(env["HDSC_OPENCODE_ENABLED"]))
    if "HDSC_OPENCODE_BASE_URL" in env:
        set_section("opencode", "base_url", env["HDSC_OPENCODE_BASE_URL"])
    if "HDSC_OPENCODE_PROJECT_DIR" in env:
        set_section("opencode", "project_dir", env["HDSC_OPENCODE_PROJECT_DIR"])
    if "HDSC_OPENCODE_TIMEOUT_SECONDS" in env:
        set_section(
            "opencode", "request_timeout_seconds", int(env["HDSC_OPENCODE_TIMEOUT_SECONDS"])
        )
    if "HDSC_OPENCODE_JOB_TIMEOUT_SECONDS" in env:
        set_section(
            "opencode", "job_timeout_seconds", int(env["HDSC_OPENCODE_JOB_TIMEOUT_SECONDS"])
        )
    if "HDSC_OPENCODE_JOB_RETENTION_SECONDS" in env:
        set_section(
            "opencode",
            "job_retention_seconds",
            int(env["HDSC_OPENCODE_JOB_RETENTION_SECONDS"]),
        )
    if "HDSC_OPENCODE_JOB_STORE_PATH" in env:
        set_section("opencode", "job_store_path", env["HDSC_OPENCODE_JOB_STORE_PATH"])
    if "HDSC_OPENCODE_FIRE_MAX_CONCURRENT" in env:
        set_section(
            "opencode", "fire_max_concurrent", int(env["HDSC_OPENCODE_FIRE_MAX_CONCURRENT"])
        )
    if "HDSC_OPENCODE_POLL_MAX_OUTPUT_CHARS" in env:
        set_section(
            "opencode", "poll_max_output_chars", int(env["HDSC_OPENCODE_POLL_MAX_OUTPUT_CHARS"])
        )
    if "HDSC_OPENCODE_DEFAULT_MODEL" in env:
        set_section("opencode", "default_model", env["HDSC_OPENCODE_DEFAULT_MODEL"])
    if "SSA_EMOTION_LIBRARY_ROOT" in env:
        set_section("emotion_library", "library_root", env["SSA_EMOTION_LIBRARY_ROOT"])
    if "SSA_QUIET_HOURS_START" in env:
        set_section("initiative", "quiet_hours_start", env["SSA_QUIET_HOURS_START"])
    if "SSA_QUIET_HOURS_END" in env:
        set_section("initiative", "quiet_hours_end", env["SSA_QUIET_HOURS_END"])

    return overrides


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


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
    cfg_dir = config_dir or _default_config_dir()
    if environ is not None:
        env_map = environ
    else:
        dotenv_path = cfg_dir.parent / ".env"
        dotenv_local_path = cfg_dir.parent / ".env.local"
        file_env = {
            key: value for key, value in dotenv_values(dotenv_path).items() if value is not None
        }
        local_file_env = {
            key: value
            for key, value in dotenv_values(dotenv_local_path).items()
            if value is not None
        }
        file_env = {**file_env, **local_file_env}
        env_map = {**file_env, **dict(os.environ)}

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
    "EmotionLibraryConfig",
    "Environment",
    "ExternalTruthConfig",
    "FirecrawlConfig",
    "GoalConfig",
    "HDSCConfig",
    "IdentityConfig",
    "InitiativeConfig",
    "LLMConfig",
    "MultimodalConfig",
    "OfflineAgencyConfig",
    "PerceptionConfig",
    "ReasoningEffort",
    "RelationshipConfig",
    "RetrievalConfig",
    "Secrets",
    "Settings",
    "SettingsError",
    "StateConfig",
    "ThinkingMode",
    "ToolConfig",
    "WarpedResonanceMode",
    "Win32Config",
    "WorldConfig",
    "load_settings",
]
