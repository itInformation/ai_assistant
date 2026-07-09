"""Tests for the bounded Tavily search adapter."""

import asyncio
import json

import httpx
import pytest

from enterprise_ai_assistant.tools import (
    TavilySearchTool,
    ToolConfigurationError,
)


def test_search_maps_and_truncates_results() -> None:
    """Only compact search fields should cross the tool boundary."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["include_raw_content"] is False
        return httpx.Response(
            200,
            request=request,
            json={
                "results": [
                    {
                        "title": "示例",
                        "url": "https://example.com",
                        "content": "一段很长的摘要",
                        "score": 0.9,
                    }
                ],
                "response_time": 0.12,
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tool = TavilySearchTool(
        api_key="test-key",
        max_content_chars=4,
        client=client,
    )

    result = asyncio.run(tool.invoke({"query": "企业 AI", "max_results": 1}))
    asyncio.run(client.aclose())

    assert result.data["results"][0]["content"] == "一段很长"
    assert result.metadata["response_time"] == 0.12


def test_search_requires_api_key_before_provider_call() -> None:
    """A missing secret should produce a configuration error."""

    tool = TavilySearchTool(api_key="")

    with pytest.raises(ToolConfigurationError, match="TAVILY_API_KEY"):
        asyncio.run(tool.invoke({"query": "企业 AI"}))
    asyncio.run(tool.close())
