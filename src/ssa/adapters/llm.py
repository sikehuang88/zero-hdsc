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
import json
import time
from contextlib import suppress
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from ssa.adapters.llm_errors import (
    LLMAuthenticationError,
    LLMInvalidResponseError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMTimeoutError,
)


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str
    content: str

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> ChatMessage:
        return cls(role="assistant", content=content)


class LLMRequest(BaseModel):
    """A typed LLM completion request (pipeline §13.3)."""

    purpose: str
    messages: list[ChatMessage]
    model: str
    temperature: float = 0.8
    max_tokens: int = 1024
    prompt_version: str = "v1"
    json_schema: dict[str, Any] | None = None
    seed: int | None = None

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
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


@runtime_checkable
class LLMAdapter(Protocol):
    """Stable LLM interface (pipeline §13.3)."""

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


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

    def __init__(self, api_key: str, *, default_model: str = "deepseek/deepseek-chat") -> None:
        self._api_key = api_key
        self._default_model = default_model

    async def complete(self, request: LLMRequest) -> LLMResponse:
        import litellm

        litellm.api_key = self._api_key

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        start = time.monotonic()
        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as exc:
            raise self._map_error(exc, request.model) from exc
        latency_ms = int((time.monotonic() - start) * 1000)

        choice = response.choices[0]
        text = choice.message.content or ""
        provider = getattr(response, "model", request.model)

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        parsed = None
        if request.json_schema is not None and text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMInvalidResponseError(
                    f"LLM returned invalid JSON: {exc}",
                    provider=provider,
                    model=request.model,
                ) from exc

        return LLMResponse(
            text=text,
            parsed=parsed,
            provider=provider,
            model=request.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _map_error(exc: Exception, model: str) -> Exception:
        """Map a litellm/provider exception to a unified error type."""
        exc_name = type(exc).__name__
        exc_str = str(exc).lower()

        if "timeout" in exc_str or "timed out" in exc_str or exc_name == "APITimeoutError":
            return LLMTimeoutError(str(exc), model=model)

        if "rate limit" in exc_str or "rate_limit" in exc_str or exc_name == "RateLimitError":
            retry_after = getattr(exc, "retry_after", None)
            return LLMRateLimitError(
                str(exc), model=model, retry_after_s=retry_after
            )

        if (
            "authentication" in exc_str
            or "api key" in exc_str
            or "unauthorized" in exc_str
            or exc_name == "AuthenticationError"
        ):
            return LLMAuthenticationError(str(exc), model=model)

        if exc_name in ("APIConnectionError", "APIError", "ServiceUnavailableError"):
            return LLMProviderUnavailableError(str(exc), model=model)

        return LLMInvalidResponseError(str(exc), model=model)


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
            text = user_msgs[-1].content if user_msgs else "(empty)"

        parsed = None
        if request.json_schema is not None:
            with suppress(json.JSONDecodeError):
                parsed = json.loads(text)

        return LLMResponse(
            text=text,
            parsed=parsed,
            provider="fake",
            model=request.model,
            input_tokens=max(1, sum(len(m.content) // 4 for m in request.messages)),
            output_tokens=max(1, len(text) // 4),
            latency_ms=1,
        )


__all__ = [
    "ChatMessage",
    "FakeLLMAdapter",
    "LLMAdapter",
    "LLMRequest",
    "LLMResponse",
    "LiteLLMAdapter",
]
