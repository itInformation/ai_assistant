"""Tavily-backed web search tool with bounded result content."""

from collections.abc import Mapping
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_exponential

from enterprise_ai_assistant.core import get_logger
from enterprise_ai_assistant.models import JSONValue, ToolResult, ToolSpec
from enterprise_ai_assistant.tools._validation import validate_arguments
from enterprise_ai_assistant.tools.exceptions import (
    ToolConfigurationError,
    ToolProviderError,
)

_SEARCH_URL = "https://api.tavily.com/search"


class SearchInput(BaseModel):
    """Validated arguments accepted by web search."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=500)
    max_results: int = Field(default=5, ge=1, le=10)
    search_depth: Literal["basic", "advanced"] = "basic"


class _RetryableSearchError(Exception):
    """Internal marker for retryable provider responses."""


class TavilySearchTool:
    """Search the public web through Tavily's Agent-oriented API."""

    spec = ToolSpec(
        name="search",
        description="搜索公开网页并返回标题、URL 和受限长度的摘要。",
        parameters=SearchInput.model_json_schema(),
    )

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float = 10.0,
        max_retries: int = 2,
        max_content_chars: int = 1_500,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if timeout_seconds <= 0 or max_retries < 0 or max_content_chars <= 0:
            raise ValueError("invalid search timeout, retry, or content limit")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._max_content_chars = max_content_chars
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient()
        self._logger = get_logger(tool="search", provider="tavily")

    async def invoke(
        self,
        arguments: Mapping[str, JSONValue],
    ) -> ToolResult:
        """Search the web and remove oversized provider fields."""

        if not self._api_key.strip():
            raise ToolConfigurationError("TAVILY_API_KEY is required")
        request = validate_arguments(SearchInput, arguments)
        try:
            payload = await self._post_with_retry(request)
            raw_results = payload.get("results")
            if not isinstance(raw_results, list):
                raise ValueError("search results must be a list")
            results = [
                {
                    "title": str(item["title"]),
                    "url": str(item["url"]),
                    "content": str(item.get("content") or "")[
                        : self._max_content_chars
                    ],
                    "score": float(item["score"]),
                }
                for item in raw_results[: request.max_results]
            ]
        except (
            httpx.HTTPError,
            _RetryableSearchError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            self._logger.error("search_request_failed", error_type=type(exc).__name__)
            raise ToolProviderError(
                f"search provider failed: {type(exc).__name__}"
            ) from exc
        return ToolResult(
            tool_name=self.spec.name,
            content=f"找到 {len(results)} 条与“{request.query}”相关的网页结果。",
            data={"query": request.query, "results": results},
            metadata={
                "provider": "Tavily",
                "response_time": self._optional_number(payload.get("response_time")),
            },
        )

    async def close(self) -> None:
        """Close the internally owned HTTP client."""

        if self._owns_client:
            await self._client.aclose()

    async def _post_with_retry(self, request: SearchInput) -> dict[str, Any]:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.TransportError, _RetryableSearchError)
            ),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                response = await self._client.post(
                    _SEARCH_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "query": request.query,
                        "max_results": request.max_results,
                        "search_depth": request.search_depth,
                        "include_answer": False,
                        "include_raw_content": False,
                    },
                    timeout=self._timeout_seconds,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise _RetryableSearchError(
                        f"retryable HTTP status {response.status_code}"
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("search response must be an object")
                return payload
        raise AssertionError("retry loop completed without a result")

    @staticmethod
    def _optional_number(value: Any) -> JSONValue:
        return float(value) if value is not None else None
