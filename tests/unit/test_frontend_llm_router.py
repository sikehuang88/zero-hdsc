from __future__ import annotations

import pytest

from ssa.adapters.frontend_llm import (
    OpenAIChatHTTPAdapter,
    OpenAIResponsesAdapter,
    ResponsesAPIError,
    SwitchableLLMAdapter,
    build_frontend_llm_adapter,
)
from ssa.adapters.llm import ChatMessage, FakeLLMAdapter, LLMRequest


@pytest.mark.asyncio
async def test_anthropic_deepseek_route_preserves_tool_definitions() -> None:
    delegate = FakeLLMAdapter()
    router = SwitchableLLMAdapter(delegate, "anthropic/deepseek-v4-flash")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "coding_search",
                "description": "Search code",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    request = LLMRequest(
        purpose="interactive_chat",
        messages=[ChatMessage.user("Inspect the project")],
        model="placeholder",
        tools=tools,
        tool_choice="auto",
    )

    await router.complete(request)

    routed = delegate.calls[0]
    assert routed.model == "anthropic/deepseek-v4-flash"
    assert routed.tools == tools
    assert routed.tool_choice == "auto"
    assert routed.thinking is None
    assert routed.timeout_seconds == 45


@pytest.mark.asyncio
async def test_responses_gateway_error_falls_back_to_chat_and_stays_there() -> None:
    fallback = FakeLLMAdapter()
    fallback.set_response("interactive_chat", "LUNA_LINK_OK")
    adapter = OpenAIResponsesAdapter(
        "secret",
        "https://gateway.example/v1",
        chat_fallback=fallback,
        chat_fallback_model="openai/gpt-5.6-luna",
    )
    response_calls = 0

    def fail_responses(*args: object) -> dict[str, object]:
        nonlocal response_calls
        response_calls += 1
        raise ResponsesAPIError(502, "upstream unavailable")

    adapter._post = fail_responses  # type: ignore[method-assign]
    request = LLMRequest(
        purpose="interactive_chat",
        messages=[ChatMessage.user("probe")],
        model="gpt-5.6-luna",
    )

    first = await adapter.complete(request)
    second = await adapter.complete(request)

    assert first.text == "LUNA_LINK_OK"
    assert second.text == "LUNA_LINK_OK"
    assert response_calls == 1
    assert [call.model for call in fallback.calls] == [
        "openai/gpt-5.6-luna",
        "openai/gpt-5.6-luna",
    ]


@pytest.mark.asyncio
async def test_responses_auth_error_does_not_fall_back_to_chat() -> None:
    fallback = FakeLLMAdapter()
    adapter = OpenAIResponsesAdapter(
        "secret",
        "https://gateway.example/v1",
        chat_fallback=fallback,
        chat_fallback_model="openai/gpt-5.6-luna",
    )

    def fail_responses(*args: object) -> dict[str, object]:
        raise ResponsesAPIError(401, "invalid key")

    adapter._post = fail_responses  # type: ignore[method-assign]
    request = LLMRequest(
        purpose="interactive_chat",
        messages=[ChatMessage.user("probe")],
        model="gpt-5.6-luna",
    )

    with pytest.raises(ResponsesAPIError, match="HTTP 401"):
        await adapter.complete(request)
    assert fallback.calls == []


@pytest.mark.asyncio
async def test_responses_adapter_normalizes_object_tool_arguments_and_missing_call_id() -> None:
    adapter = OpenAIResponsesAdapter("secret", "https://gateway.example/v1")

    def respond(*args: object) -> dict[str, object]:
        return {
            "id": "response-1",
            "model": "gpt-5.6-luna",
            "output": [
                {
                    "type": "function_call",
                    "name": "win32_drives",
                    "arguments": {"include_network": False},
                }
            ],
        }

    adapter._post = respond  # type: ignore[method-assign]
    response = await adapter.complete(
        LLMRequest(
            purpose="interactive_chat",
            messages=[ChatMessage.user("inspect drives")],
            model="gpt-5.6-luna",
        )
    )

    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].id.startswith("tool-")
    assert response.tool_calls[0].function.name == "win32_drives"
    assert response.tool_calls[0].function.arguments == '{"include_network":false}'


@pytest.mark.asyncio
async def test_raw_chat_fallback_preserves_tools_without_litellm() -> None:
    adapter = OpenAIChatHTTPAdapter("secret", "https://gateway.example/v1")
    captured: dict[str, object] = {}

    def respond(payload: dict[str, object], timeout: int) -> dict[str, object]:
        captured.update(payload)
        assert timeout == 17
        return {
            "id": "chat-1",
            "model": "gpt-5.6-luna",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "win32_drives",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        }

    adapter._post = respond  # type: ignore[method-assign]
    response = await adapter.complete(
        LLMRequest(
            purpose="interactive_chat",
            messages=[ChatMessage.user("检查磁盘")],
            model="gpt-5.6-luna",
            timeout_seconds=17,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "win32_drives",
                        "description": "Inspect drives",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            tool_choice="auto",
        )
    )

    assert captured["model"] == "gpt-5.6-luna"
    assert captured["tool_choice"] == "auto"
    assert response.tool_calls[0].function.name == "win32_drives"
    assert response.total_tokens == 7


def test_responses_builder_configures_raw_chat_fallback() -> None:
    adapter, model = build_frontend_llm_adapter(
        protocol="openai-responses",
        base_url="https://gateway.example/v1",
        api_key="secret",
        model="gpt-5.6-luna",
    )

    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert isinstance(adapter._chat_fallback, OpenAIChatHTTPAdapter)
    assert adapter._chat_fallback._url == "https://gateway.example/v1/chat/completions"
    assert model == "gpt-5.6-luna"
