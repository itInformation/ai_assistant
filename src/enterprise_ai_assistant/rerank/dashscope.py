"""DashScope qwen3-rerank adapter using the compatible HTTP API."""

from collections.abc import Sequence
from typing import Any

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_exponential

from enterprise_ai_assistant.config import Settings
from enterprise_ai_assistant.core import get_logger
from enterprise_ai_assistant.models import (
    RerankItem,
    RerankResponse,
    RetrievalCandidate,
)
from enterprise_ai_assistant.rerank.exceptions import (
    RerankerConfigurationError,
    RerankerProviderError,
)

_MAX_DOCUMENTS = 500
_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query."
)


class _RetryableRerankError(Exception):
    """Internal marker for retryable HTTP status responses."""


class DashScopeReranker:
    """Rerank text candidates with Alibaba Cloud qwen3-rerank."""

    def __init__(
        self,
        *,
        api_key: str,
        url: str,
        model: str = "qwen3-rerank",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        instruction: str = _DEFAULT_INSTRUCTION,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a bounded asynchronous reranker adapter."""

        if not api_key.strip():
            raise RerankerConfigurationError("DASHSCOPE_API_KEY is required")
        if not url.startswith(("https://", "http://")):
            raise RerankerConfigurationError("rerank URL must use HTTP or HTTPS")
        if timeout_seconds <= 0 or max_retries < 0:
            raise RerankerConfigurationError(
                "timeout must be positive and retries non-negative"
            )
        if not instruction.strip():
            raise RerankerConfigurationError("rerank instruction must not be empty")

        self._api_key = api_key
        self._url = url
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._instruction = instruction
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient()
        self._logger = get_logger(provider="dashscope", model=model)

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        top_n: int,
    ) -> RerankResponse:
        """Rerank candidates and preserve their original metadata."""

        self._validate_input(query, candidates, top_n)
        try:
            payload = await self._post_with_retry(query, candidates, top_n)
        except (
            httpx.HTTPError,
            _RetryableRerankError,
            ValueError,
        ) as exc:
            self._logger.error(
                "rerank_request_failed",
                error_type=type(exc).__name__,
            )
            raise RerankerProviderError(
                f"DashScope rerank failed: {type(exc).__name__}"
            ) from exc

        try:
            results = payload.get("results")
            if not isinstance(results, list):
                raise ValueError("results must be a list")
            seen_indexes: set[int] = set()
            items: list[RerankItem] = []
            for result in results:
                index = int(result["index"])
                if index in seen_indexes or not 0 <= index < len(candidates):
                    raise ValueError("invalid candidate indexes")
                seen_indexes.add(index)
                items.append(
                    RerankItem(
                        candidate=candidates[index],
                        score=float(result["relevance_score"]),
                    )
                )
            usage = payload.get("usage") or {}
            total_tokens = (
                int(usage["total_tokens"])
                if usage.get("total_tokens") is not None
                else None
            )
        except (KeyError, TypeError, ValueError) as exc:
            detail = (
                "invalid candidate indexes"
                if "indexes" in str(exc)
                else "invalid response schema"
            )
            raise RerankerProviderError(f"DashScope rerank returned {detail}") from exc
        return RerankResponse(
            items=tuple(items),
            model=str(payload.get("model") or self._model),
            request_id=payload.get("id"),
            total_tokens=total_tokens,
        )

    async def close(self) -> None:
        """Close an internally owned HTTP client."""

        if self._owns_client:
            await self._client.aclose()

    async def _post_with_retry(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        top_n: int,
    ) -> dict[str, Any]:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(
                (
                    httpx.TimeoutException,
                    httpx.TransportError,
                    _RetryableRerankError,
                )
            ),
            reraise=True,
            before_sleep=self._log_retry,
        )
        async for attempt in retrying:
            with attempt:
                response = await self._client.post(
                    self._url,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "query": query,
                        "documents": [candidate.content for candidate in candidates],
                        "top_n": top_n,
                        "instruct": self._instruction,
                    },
                    timeout=self._timeout_seconds,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise _RetryableRerankError(
                        f"retryable HTTP status {response.status_code}"
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("rerank response must be a JSON object")
                return payload
        raise AssertionError("retry loop completed without a result")

    def _log_retry(self, retry_state: Any) -> None:
        self._logger.warning(
            "rerank_request_retry",
            attempt=retry_state.attempt_number,
            error_type=type(retry_state.outcome.exception()).__name__,
        )

    @staticmethod
    def _validate_input(
        query: str,
        candidates: Sequence[RetrievalCandidate],
        top_n: int,
    ) -> None:
        if not query.strip():
            raise ValueError("rerank query must not be empty")
        if not candidates:
            raise ValueError("at least one rerank candidate is required")
        if len(candidates) > _MAX_DOCUMENTS:
            raise ValueError(f"qwen3-rerank accepts at most {_MAX_DOCUMENTS} documents")
        if not 1 <= top_n <= min(20, len(candidates)):
            raise ValueError("top_n must be between 1 and candidate count")


def create_dashscope_reranker(settings: Settings) -> DashScopeReranker:
    """Build the configured qwen3-rerank adapter."""

    return DashScopeReranker(
        api_key=settings.dashscope_api_key or "",
        url=settings.dashscope_rerank_url,
        model=settings.dashscope_rerank_model,
        timeout_seconds=settings.dashscope_timeout_seconds,
        max_retries=settings.dashscope_max_retries,
    )
