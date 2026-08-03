"""Tests for LLM adapter — FakeLLMAdapter and error mapping.

Contract tests for LiteLLMAdapter live in tests/contract/.
"""

from __future__ import annotations

import json

import pytest

from ssa.adapters.llm import (
    ChatMessage,
    FakeLLMAdapter,
    FunctionCall,
    LLMAdapter,
    LLMRequest,
    LLMResponse,
    ToolCall,
)
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

# ---------------------------------------------------------------------------
# ChatMessage
# ---------------------------------------------------------------------------


def test_chat_message_system():
    m = ChatMessage.system("you are helpful")
    assert m.role == "system"
    assert m.content == "you are helpful"


def test_chat_message_user():
    m = ChatMessage.user("hello")
    assert m.role == "user"
    assert m.content == "hello"


def test_chat_message_assistant():
    m = ChatMessage.assistant("hi there")
    assert m.role == "assistant"


def test_chat_message_assistant_supports_reasoning_and_tool_calls():
    tool_call = ToolCall(
        id="call_1",
        function=FunctionCall(name="lookup", arguments='{"key":"value"}'),
    )

    message = ChatMessage.assistant(
        None,
        reasoning_content="Need to look this up.",
        tool_calls=[tool_call],
    )

    assert message.content is None
    assert message.reasoning_content == "Need to look this up."
    assert message.tool_calls == [tool_call]
    assert message.tool_calls[0].type == "function"


def test_tool_call_parser_accepts_compatible_object_arguments_and_missing_fields():
    from ssa.adapters.llm import _parse_tool_calls

    calls = _parse_tool_calls(
        [
            {
                "function": {
                    "name": "win32_drives",
                    "arguments": {"include_network": False},
                }
            }
        ]
    )

    assert len(calls) == 1
    assert calls[0].id.startswith("tool-")
    assert calls[0].type == "function"
    assert json.loads(calls[0].function.arguments) == {"include_network": False}


def test_chat_message_tool_carries_call_id():
    message = ChatMessage.tool("result", "call_1")

    assert message.role == "tool"
    assert message.content == "result"
    assert message.tool_call_id == "call_1"


def test_chat_message_assistant_only_fields_reject_wrong_roles():
    with pytest.raises(ValueError, match="assistant-only"):
        ChatMessage(role="user", content="prefix", prefix=True)


@pytest.mark.parametrize(
    "message",
    [
        {"role": "tool", "content": "result"},
        {"role": "user", "content": None},
        {"role": "assistant", "content": None, "prefix": True},
    ],
    ids=["tool-call-id", "user-content", "prefix-content"],
)
def test_chat_message_rejects_invalid_role_fields(message: dict[str, object]):
    with pytest.raises(ValueError):
        ChatMessage.model_validate(message)


# ---------------------------------------------------------------------------
# LLMRequest
# ---------------------------------------------------------------------------


def test_llm_request_defaults():
    req = LLMRequest(
        purpose="test",
        messages=[ChatMessage.user("hello")],
        model="fake-model",
    )
    assert req.temperature == 0.8
    assert req.max_tokens == 1024
    assert req.prompt_version == "v1"
    assert req.json_schema is None
    assert req.seed is None
    assert req.top_p is None
    assert req.thinking is None
    assert req.reasoning_effort is None
    assert req.stop is None
    assert req.tools == []
    assert req.tool_choice is None
    assert req.user_id is None
    assert req.logprobs is False
    assert req.top_logprobs is None
    assert req.timeout_seconds is None


def test_llm_request_accepts_deepseek_v4_controls():
    req = LLMRequest(
        purpose="reasoning",
        messages=[ChatMessage.user("hello")],
        model="deepseek/deepseek-v4-pro",
        temperature=None,
        top_p=0.9,
        thinking=ThinkingMode.ENABLED,
        reasoning_effort=ReasoningEffort.MAX,
        stop=["done"],
        user_id="user_42",
        logprobs=True,
        top_logprobs=5,
        timeout_seconds=300,
    )

    assert req.top_p == 0.9
    assert req.thinking == ThinkingMode.ENABLED
    assert req.reasoning_effort == ReasoningEffort.MAX
    assert req.stop == ["done"]
    assert req.user_id == "user_42"
    assert req.top_logprobs == 5
    assert req.timeout_seconds == 300


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"temperature": 0.5, "top_p": 0.9}, "temperature or top_p"),
        ({"top_logprobs": 1}, "requires logprobs"),
        (
            {
                "thinking": ThinkingMode.DISABLED,
                "reasoning_effort": ReasoningEffort.HIGH,
            },
            "requires thinking mode",
        ),
        ({"user_id": "contains spaces"}, "user_id"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
    ],
)
def test_llm_request_rejects_invalid_v4_controls(
    updates: dict[str, object],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        LLMRequest(
            purpose="test",
            messages=[ChatMessage.user("hello")],
            model="model",
            **updates,
        )


def test_llm_request_prompt_hash_is_stable():
    req1 = LLMRequest(
        purpose="test",
        messages=[ChatMessage.user("hello"), ChatMessage.system("sys")],
        model="m",
    )
    req2 = LLMRequest(
        purpose="test",
        messages=[ChatMessage.user("hello"), ChatMessage.system("sys")],
        model="m",
    )
    assert req1.prompt_hash == req2.prompt_hash


def test_llm_request_prompt_hash_changes_with_content():
    req1 = LLMRequest(
        purpose="test",
        messages=[ChatMessage.user("hello")],
        model="m",
    )
    req2 = LLMRequest(
        purpose="test",
        messages=[ChatMessage.user("world")],
        model="m",
    )
    assert req1.prompt_hash != req2.prompt_hash


# ---------------------------------------------------------------------------
# FakeLLMAdapter
# ---------------------------------------------------------------------------


def test_fake_adapter_is_an_llm_adapter():
    assert isinstance(FakeLLMAdapter(), LLMAdapter)


@pytest.mark.asyncio
async def test_fake_adapter_returns_canned_response():
    adapter = FakeLLMAdapter()
    adapter.set_response("greeting", "Hello there!")
    req = LLMRequest(
        purpose="greeting",
        messages=[ChatMessage.user("hi")],
        model="fake",
    )
    resp = await adapter.complete(req)
    assert resp.text == "Hello there!"
    assert resp.provider == "fake"
    assert resp.model == "fake"


@pytest.mark.asyncio
async def test_fake_adapter_echoes_when_no_canned_response():
    adapter = FakeLLMAdapter()
    req = LLMRequest(
        purpose="unknown",
        messages=[ChatMessage.system("sys"), ChatMessage.user("echo me")],
        model="fake",
    )
    resp = await adapter.complete(req)
    assert resp.text == "echo me"


@pytest.mark.asyncio
async def test_fake_adapter_parses_json():
    adapter = FakeLLMAdapter()
    adapter.set_response("json_test", '{"key": "value"}')
    req = LLMRequest(
        purpose="json_test",
        messages=[ChatMessage.user("give json")],
        model="fake",
        json_schema={"type": "object"},
    )
    resp = await adapter.complete(req)
    assert resp.parsed == {"key": "value"}


@pytest.mark.asyncio
async def test_fake_adapter_records_calls():
    adapter = FakeLLMAdapter()
    adapter.set_response("test", "response")
    req = LLMRequest(
        purpose="test",
        messages=[ChatMessage.user("hello")],
        model="fake",
    )
    await adapter.complete(req)
    assert adapter.call_count == 1
    assert len(adapter.calls) == 1
    assert adapter.calls[0].purpose == "test"


@pytest.mark.asyncio
async def test_fake_adapter_returns_multiple_responses():
    adapter = FakeLLMAdapter()
    adapter.set_responses("seq", ["first", "second", "third"])
    req = LLMRequest(
        purpose="seq",
        messages=[ChatMessage.user("go")],
        model="fake",
    )
    r1 = await adapter.complete(req)
    r2 = await adapter.complete(req)
    r3 = await adapter.complete(req)
    assert r1.text == "first"
    assert r2.text == "second"
    assert r3.text == "third"


@pytest.mark.asyncio
async def test_fake_adapter_response_has_latency():
    adapter = FakeLLMAdapter()
    adapter.set_response("test", "resp")
    req = LLMRequest(
        purpose="test",
        messages=[ChatMessage.user("hi")],
        model="fake",
    )
    resp = await adapter.complete(req)
    assert resp.latency_ms >= 0
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0


# ---------------------------------------------------------------------------
# Error type hierarchy
# ---------------------------------------------------------------------------


def test_error_hierarchy():
    assert issubclass(LLMTimeoutError, Exception)
    assert issubclass(LLMRateLimitError, Exception)
    assert issubclass(LLMAuthenticationError, Exception)
    assert issubclass(LLMInsufficientBalanceError, Exception)
    assert issubclass(LLMInvalidRequestError, Exception)
    assert issubclass(LLMInvalidResponseError, Exception)
    assert issubclass(LLMProviderUnavailableError, Exception)


def test_llm_response_carries_deepseek_usage_and_tool_metadata():
    tool_call = ToolCall(
        id="call_1",
        function=FunctionCall(name="lookup", arguments="{}"),
    )
    response = LLMResponse(
        text="",
        provider="deepseek",
        model="deepseek-v4-pro",
        finish_reason="tool_calls",
        reasoning_content="I should look this up.",
        tool_calls=[tool_call],
        system_fingerprint="fp_v4",
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        prompt_cache_hit_tokens=80,
        prompt_cache_miss_tokens=20,
        reasoning_tokens=12,
    )

    assert response.finish_reason == "tool_calls"
    assert response.reasoning_content == "I should look this up."
    assert response.tool_calls == [tool_call]
    assert response.prompt_cache_hit_tokens == 80
    assert response.prompt_cache_miss_tokens == 20
    assert response.cache_hit_ratio == 0.8
    assert response.reasoning_tokens == 12


def test_llm_response_cache_hit_ratio_is_zero_without_cache_usage():
    response = LLMResponse(text="ok", provider="fake", model="fake-model")

    assert response.cache_hit_ratio == 0.0


def test_llm_rate_limit_error_has_retry_after():
    err = LLMRateLimitError("rate limited", retry_after_s=30.0)
    assert err.retry_after_s == 30.0


def test_llm_error_has_provider_and_model():
    err = LLMTimeoutError("timed out", provider="deepseek", model="deepseek-chat")
    assert err.provider == "deepseek"
    assert err.model == "deepseek-chat"
