"""LLM adapter — wraps LiteLLM behind a stable, typed interface.

Pipeline §13.3 / §13.5:
- `ChatMessage`: single message in a chat conversation.
- `LLMRequest` / `LLMResponse`: typed request/response models.
- `LLMAdapter` protocol: `async complete(request) -> response`.
- LiteLLM adapter maps provider exceptions to unified error types.
- M04 does NOT implement business retry — that's the caller's job.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_validator, model_validator

from ssa.adapters.llm_errors import (
    LLMAuthenticationError,
    LLMInsufficientBalanceError,
    LLMInvalidRequestError,
    LLMInvalidResponseError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from ssa.config import ReasoningEffort, ThinkingMode

logger = logging.getLogger(__name__)


class FunctionCall(BaseModel):
    """Function name and JSON-encoded arguments returned by a model."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """OpenAI-compatible function tool call."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class TopLogprob(BaseModel):
    """One candidate token and its log probability."""

    token: str
    logprob: float
    bytes: list[int] | None = None


class TokenLogprob(TopLogprob):
    """Selected token plus the provider's alternative candidates."""

    top_logprobs: list[TopLogprob] = Field(default_factory=list)


class LLMLogprobs(BaseModel):
    """Log probabilities for answer and reasoning token streams."""

    content: list[TokenLogprob] | None = None
    reasoning_content: list[TokenLogprob] | None = None


class ChatMessage(BaseModel):
    """A single chat message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None
    name: str | None = None
    reasoning_content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    prefix: bool = False

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls,
        content: str | None,
        *,
        reasoning_content: str | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> ChatMessage:
        return cls(
            role="assistant",
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls or [],
        )

    @classmethod
    def tool(cls, content: str, tool_call_id: str) -> ChatMessage:
        return cls(role="tool", content=content, tool_call_id=tool_call_id)

    @model_validator(mode="after")
    def _fields_match_role(self) -> ChatMessage:
        if self.role != "assistant" and (
            self.reasoning_content is not None or self.tool_calls or self.prefix
        ):
            raise ValueError("reasoning_content, tool_calls, and prefix are assistant-only fields")
        if self.role == "tool":
            if not self.tool_call_id:
                raise ValueError("tool messages require tool_call_id")
        elif self.tool_call_id is not None:
            raise ValueError("tool_call_id is valid only for tool messages")
        if self.role in {"system", "user", "tool"} and self.content is None:
            raise ValueError(f"{self.role} messages require content")
        if self.prefix and self.content is None:
            raise ValueError("assistant prefix messages require content")
        return self

    def to_api_dict(self) -> dict[str, Any]:
        """Serialize an OpenAI-compatible message without dropping tool-call types."""
        payload = self.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
        if self.role == "assistant" and self.content is None:
            payload["content"] = None
        if self.tool_calls:
            payload["tool_calls"] = [
                tool_call.model_dump(mode="json", exclude_none=True)
                for tool_call in self.tool_calls
            ]
        return payload


class LLMRequest(BaseModel):
    """A typed LLM completion request (pipeline §13.3)."""

    purpose: str
    messages: list[ChatMessage]
    model: str
    temperature: float | None = 0.8
    top_p: float | None = None
    max_tokens: int = 1024
    prompt_version: str = "v1"
    json_schema: dict[str, Any] | None = None
    seed: int | None = None
    thinking: ThinkingMode | None = None
    reasoning_effort: ReasoningEffort | None = None
    stop: str | list[str] | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    user_id: str | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    timeout_seconds: int | None = None

    @field_validator("purpose", "model", "prompt_version")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value

    @field_validator("messages")
    @classmethod
    def _messages_not_empty(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if not value:
            raise ValueError("messages must not be empty")
        return value

    @field_validator("temperature")
    @classmethod
    def _temperature_in_range(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        return value

    @field_validator("top_p")
    @classmethod
    def _top_p_in_range(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("top_p must be in [0, 1]")
        return value

    @field_validator("max_tokens")
    @classmethod
    def _max_tokens_in_range(cls, value: int) -> int:
        if not 1 <= value <= 393_216:
            raise ValueError("max_tokens must be in [1, 393216]")
        return value

    @field_validator("stop")
    @classmethod
    def _stop_sequences_valid(cls, value: str | list[str] | None) -> str | list[str] | None:
        if isinstance(value, list):
            if len(value) > 16 or any(not item for item in value):
                raise ValueError("stop must contain at most 16 non-empty strings")
        elif value == "":
            raise ValueError("stop must not be empty")
        return value

    @field_validator("tools")
    @classmethod
    def _tool_count_valid(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(value) > 128:
            raise ValueError("DeepSeek supports at most 128 tools")
        return value

    @field_validator("user_id")
    @classmethod
    def _user_id_valid(cls, value: str | None) -> str | None:
        if value is None:
            return None
        import re

        if re.fullmatch(r"[A-Za-z0-9_-]{1,512}", value) is None:
            raise ValueError("user_id must match [A-Za-z0-9_-]{1,512}")
        return value

    @field_validator("top_logprobs")
    @classmethod
    def _top_logprobs_valid(cls, value: int | None) -> int | None:
        if value is not None and not 0 <= value <= 20:
            raise ValueError("top_logprobs must be in [0, 20]")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def _timeout_valid(cls, value: int | None) -> int | None:
        if value is not None and not 1 <= value <= 600:
            raise ValueError("timeout_seconds must be in [1, 600]")
        return value

    @model_validator(mode="after")
    def _parameter_combinations_valid(self) -> LLMRequest:
        if self.temperature is not None and self.top_p is not None:
            raise ValueError("set temperature or top_p, not both")
        if self.top_logprobs is not None and not self.logprobs:
            raise ValueError("top_logprobs requires logprobs=true")
        if self.reasoning_effort is not None and self.thinking == ThinkingMode.DISABLED:
            raise ValueError("reasoning_effort requires thinking mode")
        return self

    @property
    def prompt_hash(self) -> str:
        """Stable hash of the prompt content for audit logging."""
        payload = json.dumps(
            [m.model_dump() for m in self.messages],
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLMResponse(BaseModel):
    """A typed LLM completion response (pipeline §13.3)."""

    text: str
    parsed: dict[str, Any] | None = None
    provider: str
    model: str
    response_id: str = ""
    finish_reason: str = ""
    native_finish_reason: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    logprobs: LLMLogprobs | None = None
    system_fingerprint: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    reasoning_tokens: int = 0
    latency_ms: int = 0

    @property
    def cache_hit_ratio(self) -> float:
        """Fraction of reported prompt-cache tokens served from cache."""
        cache_tokens = self.prompt_cache_hit_tokens + self.prompt_cache_miss_tokens
        if cache_tokens == 0:
            return 0.0
        return self.prompt_cache_hit_tokens / cache_tokens


@dataclass(frozen=True)
class LLMStreamEvent:
    """One provider stream delta, emitted before the unified response is complete."""

    kind: Literal["content_delta", "reasoning_delta", "tool_call_delta"]
    text: str = ""
    tool_call_index: int | None = None
    tool_call_id: str = ""
    tool_name: str = ""
    arguments: str = ""


LLMStreamCallback = Callable[[LLMStreamEvent], Awaitable[None] | None]


@runtime_checkable
class LLMAdapter(Protocol):
    """Stable LLM interface (pipeline §13.3)."""

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


@runtime_checkable
class StreamingLLMAdapter(Protocol):
    """Optional extension implemented by adapters backed by a real token stream."""

    async def stream_complete(
        self,
        request: LLMRequest,
        on_event: LLMStreamCallback,
    ) -> LLMResponse: ...


class LiteLLMAdapter:
    """Production LLM adapter backed by LiteLLM.

    Pipeline §13.5:
    1. Read versioned model config from request.
    2. Check token limit and prompt version.
    3. Call LiteLLM.
    4. Map provider exceptions to unified errors.
    5. Record tokens, latency, model, response hash.
    6. Parse JSON if schema provided.
    7. Return unified LLMResponse.
    """

    def __init__(
        self,
        api_key: str,
        *,
        api_base: str | None = None,
        timeout_seconds: int = 120,
        retry_count: int = 2,
        provider_name: str = "litellm",
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self._api_key = api_key
        self._api_base = api_base
        self._timeout_seconds = timeout_seconds
        self._retry_count = retry_count
        self._provider_name = provider_name
        self._extra_body = dict(extra_body) if extra_body is not None else None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return await self._invoke(request, self._build_kwargs(request))

    async def stream_complete(
        self,
        request: LLMRequest,
        on_event: LLMStreamCallback,
    ) -> LLMResponse:
        return await self._invoke_stream(request, self._build_kwargs(request), on_event)

    def _build_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [message.to_api_dict() for message in request.messages],
            "max_tokens": request.max_tokens,
            "api_key": self._api_key,
            "timeout": request.timeout_seconds or self._timeout_seconds,
            "max_retries": self._retry_count,
        }
        if self._api_base is not None:
            kwargs["base_url"] = self._api_base
        if self._extra_body is not None:
            kwargs["extra_body"] = self._extra_body
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        if request.thinking is not None:
            kwargs["thinking"] = {"type": request.thinking.value}
        if request.reasoning_effort is not None:
            kwargs["reasoning_effort"] = request.reasoning_effort.value
        if request.stop is not None:
            kwargs["stop"] = request.stop
        if request.tools:
            kwargs["tools"] = request.tools
        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice
        if request.logprobs:
            kwargs["logprobs"] = True
        if request.top_logprobs is not None:
            kwargs["top_logprobs"] = request.top_logprobs
        if request.user_id is not None:
            kwargs["user"] = request.user_id
        return kwargs

    async def _invoke(
        self,
        request: LLMRequest,
        kwargs: dict[str, Any],
    ) -> LLMResponse:
        import litellm

        start = time.monotonic()
        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as exc:
            raise self._map_error(exc, request.model, self._provider_name) from exc
        latency_ms = int((time.monotonic() - start) * 1000)

        choices = _value(response, "choices", [])
        if not choices:
            raise LLMInvalidResponseError(
                "LLM returned no choices",
                provider=self._provider_name,
                model=request.model,
            )
        choice = choices[0]
        message = _value(choice, "message", {})
        text = str(_value(message, "content", "") or "")
        reasoning_content = _optional_str(_value(message, "reasoning_content", None))
        try:
            tool_calls = _parse_tool_calls(_value(message, "tool_calls", None))
        except ValueError as exc:
            raise LLMInvalidResponseError(
                f"LLM returned invalid tool calls: {exc}",
                provider=self._provider_name,
                model=request.model,
            ) from exc
        actual_model = str(_value(response, "model", request.model))
        mapped_finish_reason = str(_value(choice, "finish_reason", "") or "")
        choice_provider_fields = _value(choice, "provider_specific_fields", None)
        native_finish_reason = _optional_str(
            _value(choice_provider_fields, "native_finish_reason", None)
        )
        finish_reason = mapped_finish_reason
        if native_finish_reason == "insufficient_system_resource" or not finish_reason:
            finish_reason = native_finish_reason or ""

        logprobs = None
        if finish_reason != "insufficient_system_resource":
            raw_logprobs = _object_dict(_value(choice, "logprobs", None))
            try:
                logprobs = (
                    LLMLogprobs.model_validate(raw_logprobs) if raw_logprobs is not None else None
                )
            except ValueError as exc:
                raise LLMInvalidResponseError(
                    f"LLM returned invalid logprobs: {exc}",
                    provider=self._provider_name,
                    model=request.model,
                ) from exc

        usage = _value(response, "usage", None)
        input_tokens = _int_value(usage, "prompt_tokens")
        output_tokens = _int_value(usage, "completion_tokens")
        total_tokens = _int_value(usage, "total_tokens")
        cache_hit_tokens = _int_value(usage, "prompt_cache_hit_tokens")
        cache_miss_tokens = _int_value(usage, "prompt_cache_miss_tokens")
        completion_details = _value(usage, "completion_tokens_details", None)
        reasoning_tokens = _int_value(completion_details, "reasoning_tokens")

        parsed = None
        if (
            request.json_schema is not None
            and text
            and finish_reason != "insufficient_system_resource"
        ):
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMInvalidResponseError(
                    f"LLM returned invalid JSON: {exc}",
                    provider=self._provider_name,
                    model=request.model,
                ) from exc
            if not isinstance(decoded, dict):
                raise LLMInvalidResponseError(
                    "LLM JSON response must be an object",
                    provider=self._provider_name,
                    model=request.model,
                )
            parsed = decoded

        return LLMResponse(
            text=text,
            parsed=parsed,
            provider=self._provider_name,
            model=actual_model,
            response_id=str(_value(response, "id", "")),
            finish_reason=finish_reason,
            native_finish_reason=native_finish_reason,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            logprobs=logprobs,
            system_fingerprint=_optional_str(_value(response, "system_fingerprint", None)),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            prompt_cache_hit_tokens=cache_hit_tokens,
            prompt_cache_miss_tokens=cache_miss_tokens,
            reasoning_tokens=reasoning_tokens,
            latency_ms=latency_ms,
        )

    async def _invoke_stream(
        self,
        request: LLMRequest,
        kwargs: dict[str, Any],
        on_event: LLMStreamCallback,
    ) -> LLMResponse:
        """Consume LiteLLM's SSE iterator while retaining the complete response contract."""
        import litellm

        stream_kwargs = {
            **kwargs,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        start = time.monotonic()
        try:
            stream = await litellm.acompletion(**stream_kwargs)
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_fragments: dict[int, dict[str, str]] = {}
            response_id = ""
            actual_model = request.model
            finish_reason = ""
            native_finish_reason: str | None = None
            system_fingerprint: str | None = None
            usage: object = None
            content_logprobs: list[Any] = []
            reasoning_logprobs: list[Any] = []

            async for chunk in stream:
                response_id = str(_value(chunk, "id", response_id) or response_id)
                actual_model = str(_value(chunk, "model", actual_model) or actual_model)
                system_fingerprint = _optional_str(
                    _value(chunk, "system_fingerprint", system_fingerprint)
                )
                chunk_usage = _value(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = chunk_usage

                choices = _value(chunk, "choices", [])
                if not choices:
                    continue
                choice = choices[0]
                delta = _value(choice, "delta", {})
                content_delta = str(_value(delta, "content", "") or "")
                reasoning_delta = str(
                    _value(
                        delta,
                        "reasoning_content",
                        _value(delta, "reasoning", ""),
                    )
                    or ""
                )
                if content_delta:
                    text_parts.append(content_delta)
                    await _emit_stream_event(
                        on_event,
                        LLMStreamEvent(kind="content_delta", text=content_delta),
                    )
                if reasoning_delta:
                    reasoning_parts.append(reasoning_delta)
                    await _emit_stream_event(
                        on_event,
                        LLMStreamEvent(kind="reasoning_delta", text=reasoning_delta),
                    )

                raw_tool_deltas = _value(delta, "tool_calls", []) or []
                for position, raw_call in enumerate(raw_tool_deltas):
                    index = int(_value(raw_call, "index", position) or 0)
                    fragment = tool_fragments.setdefault(
                        index,
                        {"id": "", "type": "", "name": "", "arguments": ""},
                    )
                    function = _value(raw_call, "function", {})
                    call_id_delta = str(_value(raw_call, "id", "") or "")
                    call_type_delta = str(_value(raw_call, "type", "") or "")
                    name_delta = str(_value(function, "name", "") or "")
                    arguments_delta = str(_value(function, "arguments", "") or "")
                    fragment["id"] = _merge_stream_identifier(fragment["id"], call_id_delta)
                    fragment["type"] = call_type_delta or fragment["type"]
                    fragment["name"] = _merge_stream_identifier(fragment["name"], name_delta)
                    fragment["arguments"] += arguments_delta
                    await _emit_stream_event(
                        on_event,
                        LLMStreamEvent(
                            kind="tool_call_delta",
                            tool_call_index=index,
                            tool_call_id=call_id_delta,
                            tool_name=name_delta,
                            arguments=arguments_delta,
                        ),
                    )

                mapped_finish_reason = str(_value(choice, "finish_reason", "") or "")
                if mapped_finish_reason:
                    finish_reason = mapped_finish_reason
                provider_fields = _value(choice, "provider_specific_fields", None)
                chunk_native_reason = _optional_str(
                    _value(provider_fields, "native_finish_reason", None)
                )
                if chunk_native_reason is not None:
                    native_finish_reason = chunk_native_reason

                raw_logprobs = _object_dict(_value(choice, "logprobs", None))
                if raw_logprobs is not None:
                    content_logprobs.extend(raw_logprobs.get("content") or [])
                    reasoning_logprobs.extend(raw_logprobs.get("reasoning_content") or [])
        except Exception as exc:
            raise self._map_error(exc, request.model, self._provider_name) from exc

        latency_ms = int((time.monotonic() - start) * 1000)
        if native_finish_reason == "insufficient_system_resource" or not finish_reason:
            finish_reason = native_finish_reason or finish_reason

        raw_calls = [
            {
                "id": fragment["id"],
                "type": fragment["type"] or "function",
                "function": {
                    "name": fragment["name"],
                    "arguments": fragment["arguments"],
                },
            }
            for _, fragment in sorted(tool_fragments.items())
        ]
        try:
            tool_calls = _parse_tool_calls(raw_calls)
        except ValueError as exc:
            raise LLMInvalidResponseError(
                f"LLM returned invalid streamed tool calls: {exc}",
                provider=self._provider_name,
                model=request.model,
            ) from exc

        text = "".join(text_parts)
        parsed = _parse_json_response(request, text, finish_reason, self._provider_name)
        logprobs = None
        if content_logprobs or reasoning_logprobs:
            try:
                logprobs = LLMLogprobs.model_validate(
                    {
                        "content": content_logprobs or None,
                        "reasoning_content": reasoning_logprobs or None,
                    }
                )
            except ValueError as exc:
                raise LLMInvalidResponseError(
                    f"LLM returned invalid streamed logprobs: {exc}",
                    provider=self._provider_name,
                    model=request.model,
                ) from exc

        completion_details = _value(usage, "completion_tokens_details", None)
        return LLMResponse(
            text=text,
            parsed=parsed,
            provider=self._provider_name,
            model=actual_model,
            response_id=response_id,
            finish_reason=finish_reason,
            native_finish_reason=native_finish_reason,
            reasoning_content="".join(reasoning_parts) or None,
            tool_calls=tool_calls,
            logprobs=logprobs,
            system_fingerprint=system_fingerprint,
            input_tokens=_int_value(usage, "prompt_tokens"),
            output_tokens=_int_value(usage, "completion_tokens"),
            total_tokens=_int_value(usage, "total_tokens"),
            prompt_cache_hit_tokens=_int_value(usage, "prompt_cache_hit_tokens"),
            prompt_cache_miss_tokens=_int_value(usage, "prompt_cache_miss_tokens"),
            reasoning_tokens=_int_value(completion_details, "reasoning_tokens"),
            latency_ms=latency_ms,
        )

    @staticmethod
    def _map_error(
        exc: Exception,
        model: str,
        provider: str = "",
    ) -> Exception:
        """Map a litellm/provider exception to a unified error type."""
        exc_name = type(exc).__name__
        exc_str = str(exc).lower()
        status_code = getattr(exc, "status_code", None)

        if status_code == 401 or (
            "authentication" in exc_str
            or "api key" in exc_str
            or "unauthorized" in exc_str
            or exc_name == "AuthenticationError"
        ):
            return LLMAuthenticationError(str(exc), provider=provider, model=model)

        if status_code == 402 or "insufficient balance" in exc_str:
            return LLMInsufficientBalanceError(str(exc), provider=provider, model=model)

        if status_code == 429 or (
            "rate limit" in exc_str or "rate_limit" in exc_str or exc_name == "RateLimitError"
        ):
            retry_after = getattr(exc, "retry_after", None)
            return LLMRateLimitError(
                str(exc), provider=provider, model=model, retry_after_s=retry_after
            )

        if "timeout" in exc_str or "timed out" in exc_str or exc_name == "APITimeoutError":
            return LLMTimeoutError(str(exc), provider=provider, model=model)

        if status_code in {400, 422}:
            return LLMInvalidRequestError(str(exc), provider=provider, model=model)

        if (isinstance(status_code, int) and status_code >= 500) or exc_name in (
            "APIConnectionError",
            "APIError",
            "ServiceUnavailableError",
        ):
            return LLMProviderUnavailableError(str(exc), provider=provider, model=model)

        return LLMInvalidResponseError(str(exc), provider=provider, model=model)


def _value(source: object, key: str, default: Any) -> Any:
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _int_value(source: object, key: str) -> int:
    value = _value(source, key, 0)
    return int(value) if value is not None else 0


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _object_dict(source: object) -> dict[str, Any] | None:
    if source is None:
        return None
    if isinstance(source, BaseModel):
        return source.model_dump(mode="json")
    if isinstance(source, Mapping):
        return {str(key): value for key, value in source.items()}
    model_dump = getattr(source, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    return None


def _parse_tool_calls(raw_calls: object) -> list[ToolCall]:
    if raw_calls is None:
        return []
    if not isinstance(raw_calls, list):
        raise ValueError("tool_calls must be a list")
    calls: list[ToolCall] = []
    for raw_call in raw_calls:
        function = _value(raw_call, "function", {})
        call_id = _value(raw_call, "id", None) or _value(raw_call, "call_id", None)
        call_type = _value(raw_call, "type", None) or "function"
        name = _value(function, "name", None)
        arguments = _value(function, "arguments", None)
        if not isinstance(call_id, str) or not call_id:
            call_id = f"tool-{uuid.uuid4().hex}"
        if call_type != "function":
            raise ValueError("each tool call requires type=function")
        if not isinstance(name, str) or not name:
            raise ValueError("each tool call requires function.name")
        if isinstance(arguments, Mapping):
            arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        if not isinstance(arguments, str):
            raise ValueError("each tool call requires object or JSON-string function.arguments")
        calls.append(
            ToolCall(
                id=call_id,
                function=FunctionCall(name=name, arguments=arguments),
            )
        )
    return calls


def _parse_json_response(
    request: LLMRequest,
    text: str,
    finish_reason: str,
    provider: str,
) -> dict[str, Any] | None:
    if request.json_schema is None or not text or finish_reason == "insufficient_system_resource":
        return None
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMInvalidResponseError(
            f"LLM returned invalid JSON: {exc}",
            provider=provider,
            model=request.model,
        ) from exc
    if not isinstance(decoded, dict):
        raise LLMInvalidResponseError(
            "LLM JSON response must be an object",
            provider=provider,
            model=request.model,
        )
    return decoded


async def _emit_stream_event(
    callback: LLMStreamCallback,
    event: LLMStreamEvent,
) -> None:
    """Keep presentation callback failures outside the model transport path."""
    try:
        result = callback(event)
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.debug("LLM stream callback failed", exc_info=True)


def _merge_stream_identifier(current: str, delta: str) -> str:
    """Merge fields that providers may send once, repeat, or fragment."""
    if not delta:
        return current
    if not current:
        return delta
    if delta == current or current.endswith(delta):
        return current
    return current + delta


class FakeLLMAdapter:
    """Deterministic fake LLM for tests.

    Returns canned responses based on the request purpose. If no canned
    response is found, returns an echo of the last user message.
    """

    def __init__(self) -> None:
        self._responses: dict[str, list[str]] = {}
        self._call_count = 0
        self._calls: list[LLMRequest] = []

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def calls(self) -> list[LLMRequest]:
        return list(self._calls)

    def set_response(self, purpose: str, text: str) -> None:
        self._responses[purpose] = [text]

    def set_responses(self, purpose: str, texts: list[str]) -> None:
        self._responses[purpose] = list(texts)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self._call_count += 1
        self._calls.append(request)

        responses = self._responses.get(request.purpose)
        if responses:
            text = responses.pop(0) if len(responses) > 1 else responses[0]
        else:
            user_msgs = [m for m in request.messages if m.role == "user"]
            text = (user_msgs[-1].content or "") if user_msgs else "(empty)"

        parsed = None
        if request.json_schema is not None:
            with suppress(json.JSONDecodeError):
                parsed = json.loads(text)

        return LLMResponse(
            text=text,
            parsed=parsed,
            provider="fake",
            model=request.model,
            input_tokens=max(
                1,
                sum(len(m.content or "") // 4 for m in request.messages),
            ),
            output_tokens=max(1, len(text) // 4),
            total_tokens=max(
                2,
                sum(len(m.content or "") // 4 for m in request.messages) + len(text) // 4,
            ),
            latency_ms=1,
        )


__all__ = [
    "ChatMessage",
    "FakeLLMAdapter",
    "FunctionCall",
    "LLMAdapter",
    "LLMLogprobs",
    "LLMRequest",
    "LLMResponse",
    "LLMStreamCallback",
    "LLMStreamEvent",
    "LiteLLMAdapter",
    "StreamingLLMAdapter",
    "TokenLogprob",
    "ToolCall",
    "TopLogprob",
]
