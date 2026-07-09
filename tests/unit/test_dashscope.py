"""Tests for the DashScope chat model adapter."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx
import pytest
from openai import APIConnectionError, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from enterprise_ai_assistant.config import Settings
from enterprise_ai_assistant.llm import (
    DashScopeChatModel,
    LLMConfigurationError,
    LLMProviderError,
    create_dashscope_chat_model,
)
from enterprise_ai_assistant.models import AgentMessage, ChatMessage, ToolSpec


class FakeCompletions:
    """Record requests and return configured responses."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def create(self, **request: Any) -> Any:
        """Return or raise the next configured response."""

        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    """Minimal shape consumed by the DashScope adapter."""

    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = type("FakeChat", (), {"completions": completions})()


class FakeStream:
    """Asynchronous stream of OpenAI-compatible chunks."""

    def __init__(self, *chunks: ChatCompletionChunk) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        for chunk in self._chunks:
            yield chunk


def completion(content: str = "你好") -> ChatCompletion:
    """Build a realistic OpenAI-compatible complete response."""

    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": content, "role": "assistant"},
                }
            ],
            "created": 1,
            "model": "qwen-plus",
            "object": "chat.completion",
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            },
        }
    )


def chunk(content: str, finish_reason: str | None = None) -> ChatCompletionChunk:
    """Build a realistic OpenAI-compatible streaming chunk."""

    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "delta": {"content": content},
                    "finish_reason": finish_reason,
                    "index": 0,
                }
            ],
            "created": 1,
            "model": "qwen-plus",
            "object": "chat.completion.chunk",
        }
    )


def tool_completion(arguments: str = '{"location":"北京"}') -> ChatCompletion:
    """Build a native Function Calling response."""

    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-tool",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "message": {
                        "content": "需要查询实时天气。",
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "weather",
                                    "arguments": arguments,
                                },
                            }
                        ],
                    },
                }
            ],
            "created": 1,
            "model": "qwen-plus",
            "object": "chat.completion",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
    )


def build_model(
    completions: FakeCompletions,
    *,
    max_retries: int = 0,
) -> DashScopeChatModel:
    """Build an adapter with a fake provider client."""

    return DashScopeChatModel(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="qwen-plus",
        max_retries=max_retries,
        client=cast(AsyncOpenAI, FakeClient(completions)),
    )


def test_chat_maps_response_and_usage() -> None:
    """Complete provider responses should map to domain models."""

    completions = FakeCompletions(completion())
    model = build_model(completions)

    response = asyncio.run(
        model.chat(
            [ChatMessage(role="user", content="你好")],
            temperature=0.2,
            max_tokens=100,
            response_format="json_object",
        )
    )

    assert response.content == "你好"
    assert response.model == "qwen-plus"
    assert response.finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.total_tokens == 5
    assert completions.requests[0]["temperature"] == 0.2
    assert completions.requests[0]["max_tokens"] == 100
    assert completions.requests[0]["response_format"] == {"type": "json_object"}
    assert completions.requests[0]["messages"] == [{"role": "user", "content": "你好"}]


def test_stream_chat_yields_incremental_chunks() -> None:
    """Streaming responses should remain incremental and provider independent."""

    completions = FakeCompletions(FakeStream(chunk("你"), chunk("好", "stop")))
    model = build_model(completions)

    async def collect() -> list[str]:
        return [
            item.content
            async for item in model.stream_chat(
                [ChatMessage(role="user", content="你好")]
            )
        ]

    assert asyncio.run(collect()) == ["你", "好"]
    assert completions.requests[0]["stream"] is True
    assert completions.requests[0]["stream_options"] == {"include_usage": True}


def test_chat_retries_transient_connection_errors() -> None:
    """Transient provider failures should be retried up to the configured bound."""

    connection_error = APIConnectionError(
        request=httpx.Request("POST", "https://example.com/v1/chat/completions")
    )
    completions = FakeCompletions(connection_error, completion("重试成功"))
    model = build_model(completions, max_retries=1)

    response = asyncio.run(model.chat([ChatMessage(role="user", content="请重试")]))

    assert response.content == "重试成功"
    assert len(completions.requests) == 2


def test_chat_wraps_exhausted_provider_errors() -> None:
    """Provider exceptions should not leak through the application boundary."""

    connection_error = APIConnectionError(
        request=httpx.Request("POST", "https://example.com/v1/chat/completions")
    )
    model = build_model(FakeCompletions(connection_error))

    with pytest.raises(LLMProviderError, match="APIConnectionError"):
        asyncio.run(model.chat([ChatMessage(role="user", content="你好")]))


def test_chat_rejects_empty_conversation() -> None:
    """At least one message is required for every provider request."""

    model = build_model(FakeCompletions(completion()))

    with pytest.raises(ValueError, match="at least one"):
        asyncio.run(model.chat([]))


def test_chat_with_tools_maps_calls_and_openai_messages() -> None:
    """Function Calling should remain provider-independent at the Agent layer."""

    completions = FakeCompletions(tool_completion())
    model = build_model(completions)
    spec = ToolSpec(
        name="weather",
        description="Get weather.",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}},
        },
    )
    messages = [
        AgentMessage(role="user", content="北京天气?"),
        AgentMessage(
            role="assistant",
            content="先查询。",
            tool_calls=(),
        ),
        AgentMessage(
            role="tool",
            content='{"temperature":30}',
            tool_call_id="previous-call",
        ),
    ]

    result = asyncio.run(model.chat_with_tools(messages, [spec]))

    assert result.message.tool_calls[0].arguments == {"location": "北京"}
    assert result.prompt_tokens == 10
    request = completions.requests[0]
    assert request["parallel_tool_calls"] is False
    assert request["extra_body"] == {"enable_thinking": False}
    assert request["tools"][0]["function"]["name"] == "weather"
    assert request["messages"][-1]["tool_call_id"] == "previous-call"


def test_chat_with_tools_rejects_invalid_argument_json() -> None:
    """Malformed provider arguments must not reach a Tool."""

    model = build_model(FakeCompletions(tool_completion("not-json")))

    with pytest.raises(LLMProviderError, match="invalid Agent"):
        asyncio.run(
            model.chat_with_tools(
                [AgentMessage(role="user", content="天气?")],
                [],
            )
        )


def test_factory_requires_dashscope_api_key(monkeypatch: object) -> None:
    """The production adapter must never start without a provider key."""

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)  # type: ignore[attr-defined]
    with pytest.raises(LLMConfigurationError, match="DASHSCOPE_API_KEY"):
        create_dashscope_chat_model(Settings(_env_file=None))
