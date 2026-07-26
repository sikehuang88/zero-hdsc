"""Tests for LLM adapter — FakeLLMAdapter and error mapping.

Contract tests for LiteLLMAdapter live in tests/contract/.
"""

from __future__ import annotations

import pytest

from ssa.adapters.llm import (
    ChatMessage,
    FakeLLMAdapter,
    LLMAdapter,
    LLMRequest,
)
from ssa.adapters.llm_errors import (
    LLMAuthenticationError,
    LLMInvalidResponseError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMTimeoutError,
)

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
    assert issubclass(LLMInvalidResponseError, Exception)
    assert issubclass(LLMProviderUnavailableError, Exception)


def test_llm_rate_limit_error_has_retry_after():
    err = LLMRateLimitError("rate limited", retry_after_s=30.0)
    assert err.retry_after_s == 30.0


def test_llm_error_has_provider_and_model():
    err = LLMTimeoutError("timed out", provider="deepseek", model="deepseek-chat")
    assert err.provider == "deepseek"
    assert err.model == "deepseek-chat"
