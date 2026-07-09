"""DashScope text embedding adapter using its OpenAI-compatible API."""

from collections.abc import Sequence
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from openai.types.create_embedding_response import CreateEmbeddingResponse
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_exponential

from enterprise_ai_assistant.config import Settings
from enterprise_ai_assistant.core import get_logger
from enterprise_ai_assistant.embedding.exceptions import (
    EmbeddingConfigurationError,
    EmbeddingProviderError,
)
from enterprise_ai_assistant.models import EmbeddingResponse, EmbeddingUsage

_RETRYABLE_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
_V3_DIMENSIONS = frozenset({64, 128, 256, 512, 768, 1024})
_V4_DIMENSIONS = _V3_DIMENSIONS | {1536, 2048}
_MAX_SYNC_BATCH_SIZE = 10


class DashScopeEmbeddingModel:
    """Text embedding model backed by Alibaba Cloud Model Studio."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        dimensions: int = 1024,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        client: AsyncOpenAI | None = None,
    ) -> None:
        """Create an embedding adapter with validated provider limits."""

        if not api_key.strip():
            raise EmbeddingConfigurationError("DASHSCOPE_API_KEY is required")
        if timeout_seconds <= 0:
            raise EmbeddingConfigurationError(
                "timeout_seconds must be greater than zero"
            )
        if max_retries < 0:
            raise EmbeddingConfigurationError("max_retries must not be negative")
        allowed_dimensions = (
            _V4_DIMENSIONS if model == "text-embedding-v4" else _V3_DIMENSIONS
        )
        if dimensions not in allowed_dimensions:
            raise EmbeddingConfigurationError(
                f"dimensions {dimensions} are not supported by {model}"
            )

        self._model = model
        self._dimensions = dimensions
        self._max_retries = max_retries
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            # Tenacity is the single retry owner across all provider adapters.
            max_retries=0,
        )
        self._logger = get_logger(provider="dashscope", model=model)

    async def embed(self, texts: Sequence[str]) -> EmbeddingResponse:
        """Convert up to ten non-empty texts into ordered dense vectors."""

        self._validate_texts(texts)
        try:
            response = await self._create_with_retry(texts)
        except APIStatusError as exc:
            raise self._provider_error(exc) from exc
        except _RETRYABLE_ERRORS as exc:
            raise self._provider_error(exc) from exc

        if not isinstance(response, CreateEmbeddingResponse):
            raise EmbeddingProviderError(
                "DashScope returned an unexpected embedding response type"
            )

        # Provider indexes are authoritative. Sorting protects callers from an
        # out-of-order transport response while preserving input alignment.
        ordered_data = sorted(response.data, key=lambda item: item.index)
        expected_indexes = list(range(len(texts)))
        if [item.index for item in ordered_data] != expected_indexes:
            raise EmbeddingProviderError(
                "DashScope embedding indexes do not match the input order"
            )
        vectors = tuple(
            tuple(float(value) for value in item.embedding) for item in ordered_data
        )
        if any(len(vector) != self._dimensions for vector in vectors):
            raise EmbeddingProviderError(
                "DashScope returned an unexpected embedding dimension"
            )

        return EmbeddingResponse(
            vectors=vectors,
            model=response.model,
            request_id=response.id,
            usage=EmbeddingUsage(
                prompt_tokens=response.usage.prompt_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )

    async def _create_with_retry(
        self,
        texts: Sequence[str],
    ) -> CreateEmbeddingResponse:
        retrying = AsyncRetrying(
            # max_retries counts retries after the initial request.
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(_RETRYABLE_ERRORS),
            reraise=True,
            before_sleep=self._log_retry,
        )
        async for attempt in retrying:
            with attempt:
                return await self._client.embeddings.create(
                    model=self._model,
                    input=list(texts),
                    dimensions=self._dimensions,
                    encoding_format="float",
                )
        raise AssertionError("retry loop completed without a result")

    def _log_retry(self, retry_state: Any) -> None:
        self._logger.warning(
            "embedding_request_retry",
            attempt=retry_state.attempt_number,
            error_type=type(retry_state.outcome.exception()).__name__,
        )

    def _provider_error(self, error: Exception) -> EmbeddingProviderError:
        self._logger.error(
            "embedding_request_failed",
            error_type=type(error).__name__,
        )
        return EmbeddingProviderError(
            f"DashScope embedding request failed: {type(error).__name__}"
        )

    @staticmethod
    def _validate_texts(texts: Sequence[str]) -> None:
        if not texts:
            raise ValueError("at least one text is required")
        if len(texts) > _MAX_SYNC_BATCH_SIZE:
            raise ValueError(
                f"at most {_MAX_SYNC_BATCH_SIZE} texts are allowed per request"
            )
        if any(not text.strip() for text in texts):
            raise ValueError("embedding texts must not be empty")


def create_dashscope_embedding_model(settings: Settings) -> DashScopeEmbeddingModel:
    """Build the default DashScope embedding adapter from settings."""

    return DashScopeEmbeddingModel(
        api_key=settings.dashscope_api_key or "",
        base_url=settings.dashscope_base_url,
        model=settings.dashscope_embedding_model,
        dimensions=settings.dashscope_embedding_dimensions,
        timeout_seconds=settings.dashscope_timeout_seconds,
        max_retries=settings.dashscope_max_retries,
    )
