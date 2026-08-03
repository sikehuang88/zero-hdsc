"""Typed contracts for the autonomous tool kernel."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator

FirecrawlSource = Literal["web", "news", "images"]


def _default_firecrawl_sources() -> list[FirecrawlSource]:
    return ["web"]


class ImageGenerationRoute(BaseModel):
    """Per-turn image provider route supplied by the local frontend registry."""

    protocol: Literal["openai-chat", "openai-responses", "anthropic-messages"]
    base_url: str
    api_key: SecretStr
    model: str

    @field_validator("base_url", "model")
    @classmethod
    def _route_value_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("image generation route values must not be blank")
        return value.strip().rstrip("/")


class ToolAutonomyContext(BaseModel):
    """Internal state available while the model decides and executes a tool."""

    correlation_id: str
    conversation_id: str
    energy: float = Field(ge=0.0, le=1.0)
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    trust: float = Field(ge=0.0, le=1.0)
    tension: float = Field(ge=0.0, le=1.0)
    situation_mode: str
    situation_confidence: float = Field(ge=0.0, le=1.0)
    interaction_mode: Literal["chat", "coding"] = "chat"
    workspace_root: str | None = None
    image_generation_route: ImageGenerationRoute | None = None


class ToolOutcome(BaseModel):
    ok: bool
    output: str = ""
    error: str | None = None
    exit_code: int | None = None
    elevated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionResult(ToolOutcome):
    call_id: str
    tool_name: str
    duration_ms: int = Field(ge=0)
    output_chars: int = Field(default=0, ge=0)
    truncated: bool = False


class PowerShellArguments(BaseModel):
    command: str
    working_directory: str | None = None
    timeout_seconds: int = Field(default=60, ge=1, le=600)
    elevated: bool = False

    @field_validator("command")
    @classmethod
    def _command_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("command must not be empty")
        return value


class ReadFileArguments(BaseModel):
    path: str
    offset_bytes: int = Field(default=0, ge=0)
    max_bytes: int = Field(default=1_048_576, ge=1, le=8_388_608)
    encoding: str = "utf-8"
    as_base64: bool = False

    @field_validator("path", "encoding")
    @classmethod
    def _read_text_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value


class WriteFileArguments(BaseModel):
    path: str
    content: str
    mode: Literal["overwrite", "append", "create"] = "overwrite"
    encoding: str = "utf-8"
    content_is_base64: bool = False
    create_parents: bool = True

    @field_validator("path", "encoding")
    @classmethod
    def _write_text_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value


class CodingProjectTreeArguments(BaseModel):
    path: str = "."
    max_depth: int = Field(default=3, ge=1, le=8)
    max_entries: int = Field(default=240, ge=1, le=2_000)
    include_hidden: bool = False


class CodingSearchArguments(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    path: str = "."
    glob: str | None = None
    max_results: int = Field(default=80, ge=1, le=500)
    case_sensitive: bool = False

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class CodingGitStatusArguments(BaseModel):
    path: str = "."
    include_untracked: bool = True


class WebSearchArguments(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=6, ge=1, le=12)

    @field_validator("query")
    @classmethod
    def _search_query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("search query must not be blank")
        return value.strip()


class WebOpenArguments(BaseModel):
    url: str = Field(min_length=1, max_length=4_096)

    @field_validator("url")
    @classmethod
    def _web_url_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("URL must not be blank")
        return value.strip()


class WebReadArguments(BaseModel):
    tab_id: str | None = Field(default=None, max_length=200)
    max_chars: int = Field(default=16_000, ge=500, le=32_000)


class WebTabsArguments(BaseModel):
    pass


class WebCloseArguments(BaseModel):
    tab_id: str | None = Field(default=None, max_length=200)


class FirecrawlSearchArguments(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=8, ge=1, le=20)
    sources: list[FirecrawlSource] = Field(
        default_factory=_default_firecrawl_sources,
        min_length=1,
        max_length=3,
    )
    categories: list[Literal["github", "research", "pdf"]] = Field(
        default_factory=list,
        max_length=3,
    )
    recency: Literal["h", "d", "w", "m", "y"] | None = None
    location: str | None = Field(default=None, max_length=200)
    country: str = "US"
    scrape_results: bool = False

    @field_validator("query")
    @classmethod
    def _firecrawl_query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("search query must not be blank")
        return value.strip()

    @field_validator("sources", "categories")
    @classmethod
    def _firecrawl_filters_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("search filters must be unique")
        return value

    @field_validator("country")
    @classmethod
    def _firecrawl_country_code(cls, value: str) -> str:
        normalized = value.strip().upper()
        if len(normalized) != 2 or not normalized.isalpha():
            raise ValueError("country must be a two-letter ISO code")
        return normalized

    @field_validator("location")
    @classmethod
    def _firecrawl_optional_location(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class FirecrawlScrapeArguments(BaseModel):
    url: str = Field(min_length=1, max_length=4_096)
    only_main_content: bool = True
    question: str | None = Field(default=None, max_length=2_000)
    wait_for_ms: int | None = Field(default=None, ge=0, le=30_000)
    max_age_ms: int | None = Field(default=None, ge=0, le=604_800_000)
    max_chars: int = Field(default=24_000, ge=1_000, le=32_000)

    @field_validator("url")
    @classmethod
    def _firecrawl_url_absolute(cls, value: str) -> str:
        from urllib.parse import urlsplit

        normalized = value.strip()
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("firecrawl_scrape requires an absolute HTTP(S) URL")
        return normalized

    @field_validator("question")
    @classmethod
    def _firecrawl_optional_question(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class SodaMusicNoArguments(BaseModel):
    pass


class SodaMusicControlArguments(BaseModel):
    action: Literal[
        "play",
        "pause",
        "toggle",
        "next",
        "previous",
        "volume_up",
        "volume_down",
        "mute",
    ]


class SodaMusicSearchPlayArguments(BaseModel):
    query: str = Field(min_length=1, max_length=200)

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()


class AskGPTArguments(BaseModel):
    prompt: str = Field(min_length=1, max_length=32_000)
    instructions: str = Field(default="", max_length=4_000)
    paths: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value.strip()

    @field_validator("instructions")
    @classmethod
    def _instructions_normalized(cls, value: str) -> str:
        return value.strip()

    @field_validator("paths")
    @classmethod
    def _paths_not_blank(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("attachment paths must not be blank")
        if len(value) != len(set(value)):
            raise ValueError("attachment paths must be unique")
        return value


class ImageGenerationArguments(BaseModel):
    prompt: str = Field(min_length=1, max_length=4_000)
    size: Literal["1024x1024", "1536x1024", "1024x1536"] = "1024x1024"
    quality: Literal["auto", "low", "medium", "high"] = "auto"

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("image prompt must not be blank")
        return value.strip()


class Win32NoArguments(BaseModel):
    pass


class Win32WindowListArguments(BaseModel):
    limit: int = Field(default=32, ge=1, le=512)
    include_untitled: bool = False


class Win32ProcessListArguments(BaseModel):
    limit: int = Field(default=64, ge=1, le=4_096)
    include_paths: bool = True


class Win32DirectoryListArguments(BaseModel):
    path: str
    pattern: str = "*"
    limit: int = Field(default=128, ge=1, le=10_000)
    include_hidden: bool = False

    @field_validator("path", "pattern")
    @classmethod
    def _directory_values_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("directory path and pattern must not be blank")
        return value.strip()


class Win32ClipboardArguments(BaseModel):
    max_chars: int = Field(default=4_000, ge=128, le=1_000_000)


__all__ = [
    "AskGPTArguments",
    "CodingGitStatusArguments",
    "CodingProjectTreeArguments",
    "CodingSearchArguments",
    "FirecrawlScrapeArguments",
    "FirecrawlSearchArguments",
    "ImageGenerationArguments",
    "ImageGenerationRoute",
    "PowerShellArguments",
    "ReadFileArguments",
    "SodaMusicControlArguments",
    "SodaMusicNoArguments",
    "SodaMusicSearchPlayArguments",
    "ToolAutonomyContext",
    "ToolExecutionResult",
    "ToolOutcome",
    "WebCloseArguments",
    "WebOpenArguments",
    "WebReadArguments",
    "WebSearchArguments",
    "WebTabsArguments",
    "Win32ClipboardArguments",
    "Win32DirectoryListArguments",
    "Win32NoArguments",
    "Win32ProcessListArguments",
    "Win32WindowListArguments",
    "WriteFileArguments",
]
