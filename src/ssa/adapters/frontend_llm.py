"""Hot-swappable LLM routing driven by the local frontend registry."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ssa.adapters.llm import (
    FunctionCall,
    LiteLLMAdapter,
    LLMAdapter,
    LLMRequest,
    LLMResponse,
    LLMStreamCallback,
    LLMStreamEvent,
    StreamingLLMAdapter,
    ToolCall,
)
from ssa.adapters.llm_errors import LLMInvalidResponseError

FrontendProtocol = Literal["openai-chat", "openai-responses", "anthropic-messages"]
_INTERACTIVE_TIMEOUT_SECONDS = 45
_AUXILIARY_TIMEOUT_SECONDS = 2
_RESPONSES_FALLBACK_STATUS_CODES = {400, 404, 405, 415, 422, 500, 501, 502, 503, 504}


class ResponsesAPIError(RuntimeError):
    """Structured Responses API failure used for compatibility routing."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"Responses API returned HTTP {status_code}: {detail}")
        self.status_code = status_code


class UnconfiguredLLMAdapter:
    """Reject inference until the frontend supplies its active provider route."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        del request
        raise RuntimeError("frontend inference provider is not configured")


class SwitchableLLMAdapter:
    """One stable object shared by every service, with an atomic delegate switch."""

    def __init__(self, delegate: LLMAdapter, model: str) -> None:
        self._delegate = delegate
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def configured(self) -> bool:
        """Whether requests have a concrete inference delegate."""
        return not isinstance(self._delegate, UnconfiguredLLMAdapter)

    def configure(self, delegate: LLMAdapter, model: str) -> None:
        self._delegate = delegate
        self._model = model

    def _route_request(self, request: LLMRequest) -> LLMRequest:
        return request.model_copy(
            update={
                "model": self._model,
                "thinking": None,
                "reasoning_effort": None,
                "timeout_seconds": (
                    _INTERACTIVE_TIMEOUT_SECONDS
                    if request.purpose == "interactive_chat"
                    else _AUXILIARY_TIMEOUT_SECONDS
                ),
            }
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return await self._delegate.complete(self._route_request(request))

    async def stream_complete(
        self,
        request: LLMRequest,
        on_event: LLMStreamCallback,
    ) -> LLMResponse:
        routed = self._route_request(request)
        delegate = self._delegate
        if isinstance(delegate, StreamingLLMAdapter):
            return await delegate.stream_complete(routed, on_event)
        response = await delegate.complete(routed)
        if response.text:
            emitted = on_event(LLMStreamEvent(kind="content_delta", text=response.text))
            if inspect.isawaitable(emitted):
                await emitted
        return response


class OpenAIResponsesAdapter:
    """Minimal Responses API adapter retaining messages, tools, and usage."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout_seconds: int = 120,
        *,
        disable_thinking: bool = False,
        chat_fallback: LLMAdapter | None = None,
        chat_fallback_model: str | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Responses API key must not be empty")
        self._api_key = api_key.strip()
        self._url = f"{base_url.rstrip('/')}/responses"
        self._timeout = timeout_seconds
        self._disable_thinking = disable_thinking
        self._chat_fallback = chat_fallback
        self._chat_fallback_model = chat_fallback_model
        self._chat_fallback_active = False

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if self._chat_fallback_active:
            return await self._complete_with_chat_fallback(request)
        started = time.monotonic()
        try:
            payload = await asyncio.to_thread(
                self._post,
                self._payload(request),
                request.timeout_seconds or self._timeout,
            )
        except ResponsesAPIError as exc:
            if (
                self._chat_fallback is None
                or exc.status_code not in _RESPONSES_FALLBACK_STATUS_CODES
            ):
                raise
            response = await self._complete_with_chat_fallback(request)
            self._chat_fallback_active = True
            return response
        output = payload.get("output")
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        if isinstance(payload.get("output_text"), str):
            text_parts.append(payload["output_text"])
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "function_call":
                    arguments = item.get("arguments") or {}
                    if not isinstance(arguments, str):
                        arguments = json.dumps(
                            arguments,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    tool_calls.append(
                        ToolCall(
                            id=str(
                                item.get("call_id")
                                or item.get("id")
                                or f"tool-{uuid.uuid4().hex}"
                            ),
                            function=FunctionCall(
                                name=str(item.get("name") or ""),
                                arguments=arguments,
                            ),
                        )
                    )
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if (
                            isinstance(part, dict)
                            and part.get("type") in {"output_text", "text"}
                            and isinstance(part.get("text"), str)
                        ):
                            text_parts.append(part["text"])
        text = "".join(dict.fromkeys(part for part in text_parts if part))
        raw_usage = payload.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        return LLMResponse(
            text=text,
            provider="frontend-openai-responses",
            model=str(payload.get("model") or request.model),
            response_id=str(payload.get("id") or ""),
            finish_reason="tool_calls" if tool_calls else "stop",
            tool_calls=tool_calls,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    async def _complete_with_chat_fallback(self, request: LLMRequest) -> LLMResponse:
        if self._chat_fallback is None:
            raise RuntimeError("Responses API chat fallback is not configured")
        fallback_request = request
        if self._chat_fallback_model:
            fallback_request = request.model_copy(
                update={"model": self._chat_fallback_model}
            )
        return await self._chat_fallback.complete(fallback_request)

    def _payload(self, request: LLMRequest) -> dict[str, Any]:
        inputs: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "tool":
                inputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": message.content or "",
                    }
                )
                continue
            if message.content:
                inputs.append(
                    {
                        "role": message.role,
                        "content": [{"type": "input_text", "text": message.content}],
                    }
                )
            for call in message.tool_calls:
                inputs.append(
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                )
        payload: dict[str, Any] = {
            "model": request.model,
            "input": inputs,
            "max_output_tokens": request.max_tokens,
            "store": False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if self._disable_thinking:
            payload["thinking"] = {"type": "disabled"}
        if request.tools:
            payload["tools"] = [self._response_tool(tool) for tool in request.tools]
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        return payload

    @staticmethod
    def _response_tool(tool: dict[str, Any]) -> dict[str, Any]:
        raw_function = tool.get("function")
        function: dict[str, Any] = raw_function if isinstance(raw_function, dict) else {}
        return {
            "type": "function",
            "name": function.get("name", ""),
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {}),
            "strict": bool(function.get("strict", False)),
        }

    def _post(self, payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
        request = Request(
            self._url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "HDSC/0.5 frontend-router",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ResponsesAPIError(exc.code, detail) from exc
        except URLError as exc:
            raise RuntimeError(f"Responses API connection failed: {exc.reason}") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("Responses API returned a non-object payload")
        return decoded


class OpenAIChatHTTPAdapter:
    """Small OpenAI Chat adapter used when the Responses route is unavailable."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout_seconds: int = 120,
        *,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self._api_key = api_key.strip()
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._timeout = timeout_seconds
        self._extra_body = dict(extra_body or {})

    async def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.monotonic()
        payload = await asyncio.to_thread(
            self._post,
            self._payload(request),
            request.timeout_seconds or self._timeout,
        )
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LLMInvalidResponseError(
                "LLM returned no choices",
                provider="frontend-openai-chat-http",
                model=request.model,
            )
        choice = choices[0]
        raw_message = choice.get("message")
        message: dict[str, Any] = raw_message if isinstance(raw_message, dict) else {}
        tool_calls = self._tool_calls(message.get("tool_calls"), request.model)
        text = str(message.get("content") or "")
        parsed: dict[str, Any] | None = None
        if request.json_schema is not None and text:
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMInvalidResponseError(
                    f"LLM returned invalid JSON: {exc}",
                    provider="frontend-openai-chat-http",
                    model=request.model,
                ) from exc
            if not isinstance(decoded, dict):
                raise LLMInvalidResponseError(
                    "LLM JSON response must be an object",
                    provider="frontend-openai-chat-http",
                    model=request.model,
                )
            parsed = decoded
        raw_usage = payload.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        completion_details = usage.get("completion_tokens_details")
        details = completion_details if isinstance(completion_details, dict) else {}
        return LLMResponse(
            text=text,
            parsed=parsed,
            provider="frontend-openai-chat-http",
            model=str(payload.get("model") or request.model),
            response_id=str(payload.get("id") or ""),
            finish_reason=str(choice.get("finish_reason") or ("tool_calls" if tool_calls else "stop")),
            reasoning_content=(
                str(message["reasoning_content"])
                if message.get("reasoning_content") is not None
                else None
            ),
            tool_calls=tool_calls,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            reasoning_tokens=int(details.get("reasoning_tokens") or 0),
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    def _payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [message.to_api_dict() for message in request.messages],
            "max_tokens": request.max_tokens,
            **self._extra_body,
        }
        for key in ("temperature", "top_p", "seed", "stop"):
            value = getattr(request, key)
            if value is not None:
                payload[key] = value
        if request.json_schema is not None:
            payload["response_format"] = {"type": "json_object"}
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.user_id is not None:
            payload["user"] = request.user_id
        return payload

    @staticmethod
    def _tool_calls(raw_calls: object, model: str) -> list[ToolCall]:
        if raw_calls is None:
            return []
        if not isinstance(raw_calls, list):
            raise LLMInvalidResponseError(
                "LLM returned invalid tool calls",
                provider="frontend-openai-chat-http",
                model=model,
            )
        calls: list[ToolCall] = []
        for raw in raw_calls:
            if not isinstance(raw, dict) or not isinstance(raw.get("function"), dict):
                raise LLMInvalidResponseError(
                    "LLM returned invalid tool calls",
                    provider="frontend-openai-chat-http",
                    model=model,
                )
            function = raw["function"]
            name = function.get("name")
            arguments = function.get("arguments")
            if not isinstance(name, str) or not isinstance(arguments, str):
                raise LLMInvalidResponseError(
                    "LLM returned invalid tool calls",
                    provider="frontend-openai-chat-http",
                    model=model,
                )
            calls.append(
                ToolCall(
                    id=str(raw.get("id") or f"tool-{uuid.uuid4().hex}"),
                    function=FunctionCall(name=name, arguments=arguments),
                )
            )
        return calls

    def _post(self, payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
        request = Request(
            self._url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "HDSC/0.5 frontend-router",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Chat Completions returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Chat Completions connection failed: {exc.reason}") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("Chat Completions returned a non-object payload")
        return decoded


def build_frontend_llm_adapter(
    *,
    protocol: FrontendProtocol,
    base_url: str,
    api_key: str,
    model: str,
) -> tuple[LLMAdapter, str]:
    base_url = base_url.strip().rstrip("/")
    api_key = api_key.strip()
    model = model.strip()
    if not base_url or not api_key or not model:
        raise ValueError("frontend inference route is incomplete")
    parsed_url = urlparse(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("frontend inference base URL must use HTTP or HTTPS")
    disable_thinking = "doubao" in model.lower()
    if protocol == "openai-responses":
        return (
            OpenAIResponsesAdapter(
                api_key,
                base_url,
                disable_thinking=disable_thinking,
                chat_fallback=OpenAIChatHTTPAdapter(
                    api_key,
                    base_url,
                    extra_body=(
                        {"thinking": {"type": "disabled"}}
                        if disable_thinking
                        else None
                    ),
                ),
                chat_fallback_model=model,
            ),
            model,
        )
    prefix = "anthropic" if protocol == "anthropic-messages" else "openai"
    routed_model = model if "/" in model else f"{prefix}/{model}"
    return (
        LiteLLMAdapter(
            api_key,
            api_base=base_url,
            retry_count=0,
            provider_name=f"frontend-{protocol}",
            extra_body=(
                {"thinking": {"type": "disabled"}}
                if disable_thinking and protocol == "openai-chat"
                else None
            ),
        ),
        routed_model,
    )


__all__ = [
    "FrontendProtocol",
    "OpenAIChatHTTPAdapter",
    "OpenAIResponsesAdapter",
    "ResponsesAPIError",
    "SwitchableLLMAdapter",
    "UnconfiguredLLMAdapter",
    "build_frontend_llm_adapter",
]
