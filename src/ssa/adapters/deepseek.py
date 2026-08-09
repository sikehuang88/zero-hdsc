"""DeepSeek V4 adapter with official parameter and response semantics."""

from __future__ import annotations

import re
from typing import Any

from ssa.adapters.llm import LiteLLMAdapter, LLMRequest, LLMResponse, LLMStreamCallback
from ssa.adapters.llm_errors import (
    LLMInvalidRequestError,
    LLMInvalidResponseError,
    LLMProviderUnavailableError,
)
from ssa.config import LLMConfig, Settings, ThinkingMode

DEEPSEEK_V4_FLASH = "deepseek/deepseek-v4-flash"
DEEPSEEK_V4_PRO = "deepseek/deepseek-v4-pro"
# Anthropic-format channel ids (newapi gateways expose /v1/messages with these ids)
ANTHROPIC_V4_FLASH = "anthropic/DeepSeek-V4-Flash"
ANTHROPIC_V4_PRO = "anthropic/DeepSeek-V4-Pro"
DEEPSEEK_V4_MODELS = frozenset(
    {DEEPSEEK_V4_FLASH, DEEPSEEK_V4_PRO, ANTHROPIC_V4_FLASH, ANTHROPIC_V4_PRO}
)

_OFFICIAL_MODEL_IDS = {
    "deepseek-v4-flash": DEEPSEEK_V4_FLASH,
    "deepseek-v4-pro": DEEPSEEK_V4_PRO,
    "DeepSeek-V4-Flash": ANTHROPIC_V4_FLASH,
    "DeepSeek-V4-Pro": ANTHROPIC_V4_PRO,
}


class DeepSeekV4Adapter(LiteLLMAdapter):
    """LiteLLM transport constrained to the current DeepSeek V4 API contract."""

    def __init__(
        self,
        api_key: str,
        config: LLMConfig,
        *,
        anthropic_auth_token: str = "",
    ) -> None:
        super().__init__(
            api_key,
            api_base=config.base_url,
            timeout_seconds=config.timeout_seconds,
            retry_count=config.retry_count,
            provider_name="deepseek",
        )
        self._config = config
        self._anthropic_auth_token = anthropic_auth_token

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self._validate_request(request)
        json_retries = 0
        resource_retries = 0

        while True:
            try:
                response = await self._invoke(request, self._build_deepseek_kwargs(request))
            except LLMInvalidResponseError:
                if request.json_schema is None or json_retries >= self._config.json_retry_count:
                    raise
                json_retries += 1
                continue

            if response.finish_reason == "insufficient_system_resource":
                if resource_retries < self._config.retry_count:
                    resource_retries += 1
                    continue
                raise LLMProviderUnavailableError(
                    "DeepSeek stopped generation because inference resources were insufficient",
                    provider="deepseek",
                    model=request.model,
                )

            if request.json_schema is not None and not response.text.strip():
                if json_retries < self._config.json_retry_count:
                    json_retries += 1
                    continue
                raise LLMInvalidResponseError(
                    "DeepSeek returned empty content in JSON mode",
                    provider="deepseek",
                    model=request.model,
                )
            return response

    async def stream_complete(
        self,
        request: LLMRequest,
        on_event: LLMStreamCallback,
    ) -> LLMResponse:
        """Run the DeepSeek request as SSE while preserving V4 retry semantics."""
        self._validate_request(request)
        json_retries = 0
        resource_retries = 0

        while True:
            try:
                response = await self._invoke_stream(
                    request,
                    self._build_deepseek_kwargs(request),
                    on_event,
                )
            except LLMInvalidResponseError:
                if request.json_schema is None or json_retries >= self._config.json_retry_count:
                    raise
                json_retries += 1
                continue

            if response.finish_reason == "insufficient_system_resource":
                if resource_retries < self._config.retry_count:
                    resource_retries += 1
                    continue
                raise LLMProviderUnavailableError(
                    "DeepSeek stopped generation because inference resources were insufficient",
                    provider="deepseek",
                    model=request.model,
                )

            if request.json_schema is not None and not response.text.strip():
                if json_retries < self._config.json_retry_count:
                    json_retries += 1
                    continue
                raise LLMInvalidResponseError(
                    "DeepSeek returned empty content in JSON mode",
                    provider="deepseek",
                    model=request.model,
                )
            return response

    def _build_deepseek_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        thinking = request.thinking or self._config.thinking_mode
        reasoning_effort = request.reasoning_effort or self._config.reasoning_effort
        normalized_model = _normalize_model(request.model)
        anthropic_route = _is_anthropic_route(normalized_model)

        extra_body: dict[str, Any] = {
            "user_id": request.user_id or self._config.user_id,
        }
        if anthropic_route:
            # Anthropic Messages contract: thinking is native and disabled by default;
            # reasoning_effort has no equivalent there.
            if thinking == ThinkingMode.ENABLED:
                extra_body["thinking"] = {"type": ThinkingMode.ENABLED.value}
        else:
            extra_body["thinking"] = {"type": thinking.value}
            if thinking == ThinkingMode.ENABLED:
                # LiteLLM 1.93 maps reasoning_effort only to the thinking switch.
                # Raw extra_body preserves DeepSeek V4's actual high/max strength.
                extra_body["reasoning_effort"] = reasoning_effort.value

        if anthropic_route:
            effective_api_key = self._anthropic_auth_token or self._api_key
            effective_base_url = self._config.anthropic_base_url or self._config.base_url
        else:
            effective_api_key = self._api_key
            effective_base_url = (
                self._config.beta_base_url
                if _requires_beta_endpoint(request)
                else self._config.base_url
            )

        kwargs: dict[str, Any] = {
            "model": normalized_model,
            "messages": [message.to_api_dict() for message in request.messages],
            "max_tokens": request.max_tokens,
            "api_key": effective_api_key,
            "base_url": effective_base_url,
            "timeout": request.timeout_seconds or self._config.timeout_seconds,
            "max_retries": self._config.retry_count,
            "extra_body": extra_body,
        }

        if thinking == ThinkingMode.DISABLED:
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.top_p is not None:
                kwargs["top_p"] = request.top_p
        if request.json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        if request.stop is not None:
            kwargs["stop"] = request.stop
        tool_choice = request.tool_choice
        send_tools = bool(request.tools)
        if anthropic_route:
            if tool_choice == "auto":
                # The channel returns an empty stream when tool_choice auto
                # is sent explicitly; auto is the upstream default anyway.
                tool_choice = None
            elif tool_choice == "none":
                # Omitting tools forbids calls equivalently.
                send_tools = False
                tool_choice = None
        if send_tools:
            kwargs["tools"] = request.tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if not anthropic_route:
            # Anthropic Messages has no logprobs support.
            if request.logprobs:
                kwargs["logprobs"] = True
            if request.top_logprobs is not None:
                kwargs["top_logprobs"] = request.top_logprobs
        return kwargs

    def _validate_request(self, request: LLMRequest) -> None:
        _normalize_model(request.model)
        dynamic_messages_started = False
        for message in request.messages:
            if message.role == "system":
                if dynamic_messages_started:
                    raise LLMInvalidRequestError(
                        "DeepSeek system messages must form a contiguous leading prefix",
                        provider="deepseek",
                        model=request.model,
                    )
            else:
                dynamic_messages_started = True
        if request.seed is not None:
            raise LLMInvalidRequestError(
                "DeepSeek V4 Chat Completions does not declare a seed parameter",
                provider="deepseek",
                model=request.model,
            )
        if request.json_schema is not None:
            prompt_text = "\n".join(
                message.content or ""
                for message in request.messages
                if message.role in {"system", "user"}
            )
            if "json" not in prompt_text.casefold():
                raise LLMInvalidRequestError(
                    "DeepSeek JSON mode requires the prompt to explicitly request JSON",
                    provider="deepseek",
                    model=request.model,
                )

        strict_flags: list[bool] = []
        function_names: set[str] = set()
        for tool in request.tools:
            if tool.get("type") != "function":
                raise LLMInvalidRequestError(
                    "DeepSeek supports function tools only",
                    provider="deepseek",
                    model=request.model,
                )
            function = tool.get("function")
            if not isinstance(function, dict):
                raise LLMInvalidRequestError(
                    "DeepSeek function tools require a function object",
                    provider="deepseek",
                    model=request.model,
                )
            name = function.get("name")
            if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) is None:
                raise LLMInvalidRequestError(
                    "DeepSeek function names must match [A-Za-z0-9_-]{1,64}",
                    provider="deepseek",
                    model=request.model,
                )
            if name in function_names:
                raise LLMInvalidRequestError(
                    f"DeepSeek function names must be unique; duplicate {name!r}",
                    provider="deepseek",
                    model=request.model,
                )
            if "description" in function and not isinstance(function["description"], str):
                raise LLMInvalidRequestError(
                    "DeepSeek function descriptions must be strings",
                    provider="deepseek",
                    model=request.model,
                )
            if "parameters" in function and not isinstance(function["parameters"], dict):
                raise LLMInvalidRequestError(
                    "DeepSeek function parameters must be a JSON Schema object",
                    provider="deepseek",
                    model=request.model,
                )
            if "strict" in function and not isinstance(function["strict"], bool):
                raise LLMInvalidRequestError(
                    "DeepSeek function strict must be a boolean",
                    provider="deepseek",
                    model=request.model,
                )
            function_names.add(name)
            strict_flags.append(function.get("strict") is True)
        if any(strict_flags) and not all(strict_flags):
            raise LLMInvalidRequestError(
                "DeepSeek strict mode requires every function tool to set strict=true",
                provider="deepseek",
                model=request.model,
            )
        thinking = request.thinking or self._config.thinking_mode
        _validate_tool_choice(request, function_names, thinking)

        prefix_positions = [
            index for index, message in enumerate(request.messages) if message.prefix
        ]
        if prefix_positions and prefix_positions != [len(request.messages) - 1]:
            raise LLMInvalidRequestError(
                "DeepSeek prefix completion requires one final assistant prefix message",
                provider="deepseek",
                model=request.model,
            )

        if thinking == ThinkingMode.DISABLED and request.reasoning_effort is not None:
            raise LLMInvalidRequestError(
                "reasoning_effort requires effective thinking mode to be enabled",
                provider="deepseek",
                model=request.model,
            )
        if thinking == ThinkingMode.ENABLED:
            for message in request.messages:
                if (
                    message.role == "assistant"
                    and message.tool_calls
                    and not message.reasoning_content
                ):
                    raise LLMInvalidRequestError(
                        "Thinking-mode tool turns must preserve assistant reasoning_content",
                        provider="deepseek",
                        model=request.model,
                    )


def _validate_tool_choice(
    request: LLMRequest,
    function_names: set[str],
    thinking: ThinkingMode,
) -> None:
    choice = request.tool_choice
    if choice is None:
        return
    if thinking == ThinkingMode.ENABLED and not (
        isinstance(choice, str) and choice in {"none", "auto"}
    ):
        raise LLMInvalidRequestError(
            "DeepSeek thinking mode supports only auto or none tool_choice",
            provider="deepseek",
            model=request.model,
        )
    if isinstance(choice, str):
        if choice not in {"none", "auto", "required"}:
            raise LLMInvalidRequestError(
                "DeepSeek tool_choice must be none, auto, required, or a named function",
                provider="deepseek",
                model=request.model,
            )
        if choice != "none" and not function_names:
            raise LLMInvalidRequestError(
                "DeepSeek tool_choice requires at least one function tool",
                provider="deepseek",
                model=request.model,
            )
        return

    function = choice.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    if choice.get("type") != "function" or not isinstance(name, str):
        raise LLMInvalidRequestError(
            "DeepSeek named tool_choice requires type=function and function.name",
            provider="deepseek",
            model=request.model,
        )
    if name not in function_names:
        raise LLMInvalidRequestError(
            f"DeepSeek tool_choice references unknown function {name!r}",
            provider="deepseek",
            model=request.model,
        )


def _is_anthropic_route(model: str) -> bool:
    """Whether the normalized model id routes through Anthropic Messages format."""
    return model.startswith("anthropic/")


def _normalize_model(model: str) -> str:
    """Map known aliases to canonical ids; pass custom model ids through unchanged."""
    return _OFFICIAL_MODEL_IDS.get(model, model)


def _requires_beta_endpoint(request: LLMRequest) -> bool:
    if any(message.prefix for message in request.messages):
        return True
    return any(
        isinstance(tool.get("function"), dict) and tool["function"].get("strict") is True
        for tool in request.tools
    )


def build_deepseek_adapter(settings: Settings) -> DeepSeekV4Adapter:
    """Build the production DeepSeek adapter from validated project settings."""
    return DeepSeekV4Adapter(
        settings.secrets.require_llm(),
        settings.llm,
        anthropic_auth_token=settings.secrets.anthropic_auth_token.get_secret_value(),
    )


__all__ = [
    "ANTHROPIC_V4_FLASH",
    "ANTHROPIC_V4_PRO",
    "DEEPSEEK_V4_FLASH",
    "DEEPSEEK_V4_MODELS",
    "DEEPSEEK_V4_PRO",
    "DeepSeekV4Adapter",
    "build_deepseek_adapter",
]
