"""Contract tests for the DeepSeek V4 adapter boundary."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from ssa.adapters.deepseek import (
    DEEPSEEK_V4_FLASH,
    DEEPSEEK_V4_PRO,
    DeepSeekV4Adapter,
    build_deepseek_adapter,
)
from ssa.adapters.llm import ChatMessage, FunctionCall, LLMRequest, LLMStreamEvent, ToolCall
from ssa.adapters.llm_errors import (
    LLMAuthenticationError,
    LLMInsufficientBalanceError,
    LLMInvalidRequestError,
    LLMInvalidResponseError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
)
from ssa.config import LLMConfig, ReasoningEffort, Secrets, Settings, ThinkingMode

_BASE_URL = "https://api.deepseek.com"
_BETA_BASE_URL = "https://api.deepseek.com/beta"


class _HTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"DeepSeek HTTP {status_code}")
        self.status_code = status_code


def _response(
    content: str | None = "ok",
    *,
    finish_reason: str = "stop",
    native_finish_reason: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: object | None = None,
    logprobs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    choice: dict[str, Any] = {
        "index": 0,
        "message": message,
        "finish_reason": finish_reason,
    }
    if native_finish_reason is not None:
        choice["provider_specific_fields"] = {"native_finish_reason": native_finish_reason}
    if logprobs is not None:
        choice["logprobs"] = logprobs
    return {
        "id": "response_v4",
        "model": "deepseek-v4-pro",
        "system_fingerprint": "fp_v4",
        "choices": [choice],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 12},
        },
    }


def _install_completion(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[dict[str, Any] | Exception],
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    queued = list(responses)

    async def acompletion(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if not queued:
            raise AssertionError("unexpected extra LiteLLM call")
        result = queued.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(acompletion=acompletion))
    return calls


class _AsyncChunks:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> _AsyncChunks:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _install_stream(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def acompletion(**kwargs: Any) -> _AsyncChunks:
        calls.append(kwargs)
        return _AsyncChunks(list(chunks))

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(acompletion=acompletion))
    return calls


def _adapter(**overrides: Any) -> DeepSeekV4Adapter:
    config = LLMConfig(
        base_url=_BASE_URL,
        beta_base_url=_BETA_BASE_URL,
        **overrides,
    )
    return DeepSeekV4Adapter("test-key", config)


def _request(**overrides: Any) -> LLMRequest:
    values: dict[str, Any] = {
        "purpose": "deepseek_contract",
        "messages": [ChatMessage.user("Hello")],
        "model": DEEPSEEK_V4_FLASH,
    }
    values.update(overrides)
    return LLMRequest(**values)


@pytest.mark.asyncio
async def test_settings_factory_builds_configured_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_completion(monkeypatch, [_response()])
    settings = Settings(
        llm=LLMConfig(base_url="https://proxy.example/v1"),
        secrets=Secrets.from_env({"DEEPSEEK_API_KEY": "factory-key"}),
    )

    await build_deepseek_adapter(settings).complete(_request())

    assert calls[0]["api_key"] == "factory-key"
    assert calls[0]["base_url"] == "https://proxy.example/v1"


@pytest.mark.asyncio
async def test_flash_non_thinking_request_uses_official_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_completion(monkeypatch, [_response()])
    adapter = _adapter(thinking_mode=ThinkingMode.DISABLED)

    response = await adapter.complete(_request(temperature=0.3, max_tokens=2048, user_id="user_42"))

    assert response.provider == "deepseek"
    assert calls == [
        {
            "model": DEEPSEEK_V4_FLASH,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 2048,
            "api_key": "test-key",
            "base_url": _BASE_URL,
            "timeout": 120,
            "max_retries": 2,
            "extra_body": {
                "thinking": {"type": "disabled"},
                "user_id": "user_42",
            },
            "temperature": 0.3,
        }
    ]


@pytest.mark.asyncio
async def test_pro_thinking_request_preserves_max_effort_and_drops_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_completion(monkeypatch, [_response(reasoning_content="reasoning")])
    adapter = _adapter()

    await adapter.complete(
        _request(
            model="deepseek-v4-pro",
            temperature=1.7,
            thinking=ThinkingMode.ENABLED,
            reasoning_effort=ReasoningEffort.MAX,
        )
    )

    sent = calls[0]
    assert sent["model"] == DEEPSEEK_V4_PRO
    assert sent["extra_body"] == {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
        "user_id": "hdsc-primary",
    }
    assert "temperature" not in sent
    assert "top_p" not in sent


@pytest.mark.asyncio
async def test_response_parses_reasoning_tools_and_cache_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"query":"ssa"}'},
        }
    ]
    _install_completion(
        monkeypatch,
        [
            _response(
                None,
                finish_reason="tool_calls",
                reasoning_content="I should call lookup.",
                tool_calls=tool_calls,
            )
        ],
    )

    response = await _adapter().complete(
        _request(
            model=DEEPSEEK_V4_PRO,
            temperature=None,
            thinking=ThinkingMode.ENABLED,
        )
    )

    assert response.text == ""
    assert response.finish_reason == "tool_calls"
    assert response.reasoning_content == "I should call lookup."
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].function.name == "lookup"
    assert response.prompt_cache_hit_tokens == 80
    assert response.prompt_cache_miss_tokens == 20
    assert response.reasoning_tokens == 12
    assert response.system_fingerprint == "fp_v4"


@pytest.mark.asyncio
async def test_sse_stream_aggregates_content_reasoning_usage_and_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        {
            "id": "stream_1",
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "delta": {"reasoning_content": "private ", "content": "Hello"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "stream_1",
            "model": "deepseek-v4-flash",
            "system_fingerprint": "fp_stream",
            "choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}],
        },
        {
            "id": "stream_1",
            "model": "deepseek-v4-flash",
            "choices": [],
            "usage": {
                "prompt_tokens": 40,
                "completion_tokens": 9,
                "total_tokens": 49,
                "prompt_cache_hit_tokens": 30,
                "prompt_cache_miss_tokens": 10,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        },
    ]
    calls = _install_stream(monkeypatch, chunks)
    events: list[LLMStreamEvent] = []

    response = await _adapter(thinking_mode=ThinkingMode.DISABLED).stream_complete(
        _request(thinking=ThinkingMode.DISABLED),
        events.append,
    )

    assert response.text == "Hello world"
    assert response.reasoning_content == "private "
    assert response.finish_reason == "stop"
    assert response.input_tokens == 40
    assert response.output_tokens == 9
    assert response.prompt_cache_hit_tokens == 30
    assert response.prompt_cache_miss_tokens == 10
    assert response.reasoning_tokens == 2
    assert [event.kind for event in events] == [
        "content_delta",
        "reasoning_delta",
        "content_delta",
    ]
    assert calls[0]["stream"] is True
    assert calls[0]["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_sse_stream_assembles_fragmented_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        {
            "id": "stream_tool",
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read_", "arguments": '{"pa'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "stream_tool",
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "type": "function",
                                "function": {"name": "file", "arguments": 'th":"x"}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
        },
    ]
    _install_stream(monkeypatch, chunks)
    events: list[LLMStreamEvent] = []

    response = await _adapter(thinking_mode=ThinkingMode.DISABLED).stream_complete(
        _request(thinking=ThinkingMode.DISABLED),
        events.append,
    )

    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].function.name == "read_file"
    assert response.tool_calls[0].function.arguments == '{"path":"x"}'
    assert [event.tool_name for event in events] == ["read_", "file"]


@pytest.mark.asyncio
async def test_sse_callback_failure_does_not_break_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stream(
        monkeypatch,
        [
            {
                "id": "stream_callback",
                "model": "deepseek-v4-flash",
                "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}],
            }
        ],
    )

    def fail_callback(_event: LLMStreamEvent) -> None:
        raise RuntimeError("presentation failed")

    response = await _adapter(thinking_mode=ThinkingMode.DISABLED).stream_complete(
        _request(thinking=ThinkingMode.DISABLED),
        fail_callback,
    )

    assert response.text == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_calls",
    [
        {"id": "call_1"},
        [{"type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
        [{"id": "call_1", "function": {"name": "lookup", "arguments": "{}"}}],
        [{"id": "call_1", "type": "function", "function": {"arguments": "{}"}}],
        [{"id": "call_1", "type": "function", "function": {"name": "lookup"}}],
    ],
    ids=["not-list", "missing-id", "missing-type", "missing-name", "missing-arguments"],
)
async def test_malformed_tool_call_response_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tool_calls: object,
) -> None:
    _install_completion(
        monkeypatch,
        [_response(content=None, finish_reason="tool_calls", tool_calls=tool_calls)],
    )

    with pytest.raises(LLMInvalidResponseError, match="tool calls"):
        await _adapter().complete(_request())


@pytest.mark.asyncio
async def test_response_preserves_content_and_reasoning_logprobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logprobs = {
        "content": [
            {
                "token": "answer",
                "logprob": -0.1,
                "bytes": [97],
                "top_logprobs": [{"token": "reply", "logprob": -0.4, "bytes": [114]}],
            }
        ],
        "reasoning_content": [
            {
                "token": "think",
                "logprob": -0.2,
                "bytes": [116],
                "top_logprobs": [],
            }
        ],
    }
    _install_completion(monkeypatch, [_response(logprobs=logprobs)])

    response = await _adapter().complete(_request(logprobs=True, top_logprobs=1))

    assert response.logprobs is not None
    assert response.logprobs.content is not None
    assert response.logprobs.content[0].token == "answer"
    assert response.logprobs.content[0].top_logprobs[0].token == "reply"
    assert response.logprobs.reasoning_content is not None
    assert response.logprobs.reasoning_content[0].token == "think"


@pytest.mark.asyncio
async def test_invalid_logprobs_map_to_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_completion(
        monkeypatch,
        [
            _response(
                logprobs={
                    "content": [{"token": "x", "logprob": "invalid"}],
                }
            )
        ],
    )

    with pytest.raises(LLMInvalidResponseError, match="logprobs"):
        await _adapter().complete(_request(logprobs=True))


@pytest.mark.asyncio
async def test_json_mode_retries_an_empty_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_completion(monkeypatch, [_response(""), _response('{"ok":true}')])
    adapter = _adapter(json_retry_count=1)

    response = await adapter.complete(
        _request(
            messages=[ChatMessage.user("Return a JSON object")],
            json_schema={"type": "object"},
        )
    )

    assert response.parsed == {"ok": True}
    assert len(calls) == 2
    assert calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_json_mode_retries_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_completion(
        monkeypatch,
        [_response("not-json"), _response('{"ok":true}')],
    )

    response = await _adapter(json_retry_count=1).complete(
        _request(
            messages=[ChatMessage.system("Output JSON"), ChatMessage.user("go")],
            json_schema={"type": "object"},
        )
    )

    assert response.parsed == {"ok": True}
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_insufficient_system_resource_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_completion(
        monkeypatch,
        [
            _response("", finish_reason="insufficient_system_resource"),
            _response("ok"),
        ],
    )

    response = await _adapter(retry_count=1).complete(_request())

    assert response.text == "ok"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_native_insufficient_system_resource_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_completion(
        monkeypatch,
        [
            _response(
                "",
                finish_reason="stop",
                native_finish_reason="insufficient_system_resource",
            ),
            _response("ok"),
        ],
    )

    response = await _adapter(retry_count=1).complete(_request())

    assert response.text == "ok"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_native_resource_stop_precedes_json_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_completion(
        monkeypatch,
        [
            _response(
                '{"partial":',
                finish_reason="stop",
                native_finish_reason="insufficient_system_resource",
            ),
            _response('{"ok":true}'),
        ],
    )

    response = await _adapter(retry_count=1, json_retry_count=0).complete(
        _request(
            messages=[ChatMessage.user("Return JSON")],
            json_schema={"type": "object"},
        )
    )

    assert response.parsed == {"ok": True}
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_insufficient_system_resource_exhaustion_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_completion(
        monkeypatch,
        [
            _response("", finish_reason="insufficient_system_resource"),
            _response("", finish_reason="insufficient_system_resource"),
        ],
    )

    with pytest.raises(LLMProviderUnavailableError, match="resources were insufficient"):
        await _adapter(retry_count=1).complete(_request())

    assert len(calls) == 2


@pytest.mark.asyncio
async def test_strict_tools_use_beta_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_completion(monkeypatch, [_response()])
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }
    ]

    await _adapter().complete(_request(tools=tools))

    assert calls[0]["base_url"] == _BETA_BASE_URL
    assert calls[0]["tools"] == tools


@pytest.mark.asyncio
async def test_mixed_strict_tools_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_completion(monkeypatch, [_response()])
    tools = [
        {"type": "function", "function": {"name": "strict", "strict": True}},
        {"type": "function", "function": {"name": "loose"}},
    ]

    with pytest.raises(LLMInvalidRequestError, match="every function tool"):
        await _adapter().complete(_request(tools=tools))

    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tools",
    [
        [{"type": "custom", "function": {"name": "lookup"}}],
        [{"type": "function"}],
        [{"type": "function", "function": {"name": "has space"}}],
        [
            {"type": "function", "function": {"name": "lookup"}},
            {"type": "function", "function": {"name": "lookup"}},
        ],
        [{"type": "function", "function": {"name": "lookup", "description": 1}}],
        [{"type": "function", "function": {"name": "lookup", "parameters": []}}],
        [{"type": "function", "function": {"name": "lookup", "strict": "true"}}],
    ],
    ids=[
        "type",
        "function",
        "name",
        "duplicate",
        "description-type",
        "parameters-type",
        "strict-type",
    ],
)
async def test_invalid_function_tool_definitions_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tools: list[dict[str, Any]],
) -> None:
    calls = _install_completion(monkeypatch, [_response()])

    with pytest.raises(LLMInvalidRequestError):
        await _adapter().complete(_request(tools=tools))

    assert calls == []


@pytest.mark.asyncio
async def test_named_tool_choice_is_validated_and_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_completion(monkeypatch, [_response()])
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    choice = {"type": "function", "function": {"name": "lookup"}}

    await _adapter().complete(_request(tools=tools, tool_choice=choice))

    assert calls[0]["tool_choice"] == choice


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_choice",
    ["required", {"type": "function", "function": {"name": "lookup"}}],
    ids=["required", "named"],
)
async def test_thinking_mode_rejects_forced_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
    tool_choice: str | dict[str, Any],
) -> None:
    calls = _install_completion(monkeypatch, [_response()])
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    with pytest.raises(LLMInvalidRequestError, match="thinking mode"):
        await _adapter().complete(
            _request(
                model=DEEPSEEK_V4_PRO,
                temperature=None,
                thinking=ThinkingMode.ENABLED,
                tools=tools,
                tool_choice=tool_choice,
            )
        )

    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_choice", ["auto", "none"])
async def test_thinking_mode_accepts_auto_and_none_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
    tool_choice: str,
) -> None:
    calls = _install_completion(monkeypatch, [_response()])
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    await _adapter().complete(
        _request(
            model=DEEPSEEK_V4_PRO,
            temperature=None,
            thinking=ThinkingMode.ENABLED,
            tools=tools,
            tool_choice=tool_choice,
        )
    )

    assert calls[0]["tool_choice"] == tool_choice


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_choice",
    [
        "invalid",
        "required",
        {"type": "function", "function": {"name": "missing"}},
    ],
    ids=["unknown-mode", "required-without-tools", "unknown-function"],
)
async def test_invalid_tool_choice_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tool_choice: str | dict[str, Any],
) -> None:
    calls = _install_completion(monkeypatch, [_response()])

    with pytest.raises(LLMInvalidRequestError, match="tool_choice"):
        await _adapter().complete(_request(tool_choice=tool_choice))

    assert calls == []


@pytest.mark.asyncio
async def test_thinking_tool_turn_requires_reasoning_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_completion(monkeypatch, [_response()])
    tool_call = ToolCall(
        id="call_1",
        function=FunctionCall(name="lookup", arguments="{}"),
    )
    messages = [
        ChatMessage.user("look this up"),
        ChatMessage.assistant(None, tool_calls=[tool_call]),
        ChatMessage.tool("result", "call_1"),
    ]

    with pytest.raises(LLMInvalidRequestError, match="reasoning_content"):
        await _adapter().complete(
            _request(
                model=DEEPSEEK_V4_PRO,
                messages=messages,
                temperature=None,
                thinking=ThinkingMode.ENABLED,
            )
        )

    assert calls == []


@pytest.mark.asyncio
async def test_assistant_tool_call_serialization_keeps_function_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_completion(monkeypatch, [_response()])
    tool_call = ToolCall(
        id="call_1",
        function=FunctionCall(name="lookup", arguments="{}"),
    )
    messages = [
        ChatMessage.user("look this up"),
        ChatMessage.assistant(
            None,
            reasoning_content="I should call lookup.",
            tool_calls=[tool_call],
        ),
        ChatMessage.tool("result", "call_1"),
    ]

    await _adapter().complete(
        _request(
            model=DEEPSEEK_V4_PRO,
            messages=messages,
            temperature=None,
            thinking=ThinkingMode.ENABLED,
        )
    )

    sent_tool_call = calls[0]["messages"][1]["tool_calls"][0]
    assert calls[0]["messages"][1]["content"] is None
    assert sent_tool_call["type"] == "function"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "non_prompt_message",
    [
        ChatMessage.assistant("Return JSON"),
        ChatMessage.tool("Return JSON", "call_1"),
    ],
    ids=["assistant", "tool"],
)
async def test_json_keyword_must_be_in_system_or_user_prompt(
    monkeypatch: pytest.MonkeyPatch,
    non_prompt_message: ChatMessage,
) -> None:
    calls = _install_completion(monkeypatch, [_response('{"ok":true}')])
    request = _request(
        messages=[ChatMessage.user("Return the result"), non_prompt_message],
        json_schema={"type": "object"},
    )

    with pytest.raises(LLMInvalidRequestError, match="prompt"):
        await _adapter().complete(request)

    assert calls == []


@pytest.mark.asyncio
async def test_leading_system_prefix_order_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_completion(monkeypatch, [_response()])
    messages = [
        ChatMessage.system("stable persona"),
        ChatMessage.system("stable output contract"),
        ChatMessage.user("new event"),
    ]

    await _adapter().complete(_request(messages=messages))

    assert calls[0]["messages"] == [
        {"role": "system", "content": "stable persona"},
        {"role": "system", "content": "stable output contract"},
        {"role": "user", "content": "new event"},
    ]


@pytest.mark.asyncio
async def test_system_message_after_dynamic_context_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_completion(monkeypatch, [_response()])
    messages = [
        ChatMessage.system("stable persona"),
        ChatMessage.user("new event"),
        ChatMessage.system("late instruction"),
    ]

    with pytest.raises(LLMInvalidRequestError, match="leading prefix"):
        await _adapter().complete(_request(messages=messages))

    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages",
    [
        [
            ChatMessage.user("start"),
            ChatMessage(role="assistant", content="prefix", prefix=True),
            ChatMessage.user("continue"),
        ],
    ],
    ids=["not-last"],
)
async def test_prefix_must_be_on_the_final_assistant_message(
    monkeypatch: pytest.MonkeyPatch,
    messages: list[ChatMessage],
) -> None:
    calls = _install_completion(monkeypatch, [_response()])

    with pytest.raises(LLMInvalidRequestError, match="prefix"):
        await _adapter().complete(_request(messages=messages))

    assert calls == []


@pytest.mark.asyncio
async def test_valid_prefix_completion_uses_beta_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_completion(monkeypatch, [_response()])
    messages = [
        ChatMessage.user("Write Python"),
        ChatMessage(role="assistant", content="```python\n", prefix=True),
    ]

    await _adapter().complete(_request(messages=messages))

    assert calls[0]["base_url"] == _BETA_BASE_URL
    assert calls[0]["messages"][-1]["prefix"] is True


@pytest.mark.asyncio
async def test_request_effort_is_not_silently_ignored_when_effective_thinking_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_completion(monkeypatch, [_response()])
    adapter = _adapter(thinking_mode=ThinkingMode.DISABLED)
    request = _request(reasoning_effort=ReasoningEffort.MAX)

    with pytest.raises(LLMInvalidRequestError, match="reasoning_effort"):
        await adapter.complete(request)

    assert calls == []


@pytest.mark.asyncio
async def test_undeclared_seed_parameter_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_completion(monkeypatch, [_response()])

    with pytest.raises(LLMInvalidRequestError, match="seed"):
        await _adapter().complete(_request(seed=42))

    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "deepseek-chat",
        "deepseek-reasoner",
        "deepseek/deepseek-chat",
        "deepseek/deepseek-reasoner",
    ],
)
async def test_deprecated_model_aliases_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    calls = _install_completion(monkeypatch, [_response()])

    with pytest.raises(LLMInvalidRequestError, match="expired"):
        await _adapter().complete(_request(model=model))

    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (400, LLMInvalidRequestError),
        (401, LLMAuthenticationError),
        (402, LLMInsufficientBalanceError),
        (422, LLMInvalidRequestError),
        (429, LLMRateLimitError),
        (500, LLMProviderUnavailableError),
        (503, LLMProviderUnavailableError),
    ],
)
async def test_http_statuses_map_to_stable_errors(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    error_type: type[Exception],
) -> None:
    _install_completion(monkeypatch, [_HTTPError(status_code)])

    with pytest.raises(error_type) as exc_info:
        await _adapter().complete(_request())

    assert exc_info.value.provider == "deepseek"
    assert exc_info.value.model == DEEPSEEK_V4_FLASH


@pytest.mark.asyncio
async def test_json_retry_exhaustion_raises_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_completion(monkeypatch, [_response(""), _response("")])

    with pytest.raises(LLMInvalidResponseError, match="empty content"):
        await _adapter(json_retry_count=1).complete(
            _request(
                messages=[ChatMessage.user("Return JSON")],
                json_schema={"type": "object"},
            )
        )

    assert len(calls) == 2
