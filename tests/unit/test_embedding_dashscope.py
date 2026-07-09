"""Tests for the DashScope text embedding adapter."""

import asyncio
from typing import Any, cast

import httpx
import pytest
from openai import APIConnectionError, AsyncOpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse

from enterprise_ai_assistant.config import Settings
from enterprise_ai_assistant.embedding import (
    DashScopeEmbeddingModel,
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    create_dashscope_embedding_model,
)


class FakeEmbeddings:
    """Record requests and return configured embedding responses."""

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
    """Minimal OpenAI-compatible client shape used by the adapter."""

    def __init__(self, embeddings: FakeEmbeddings) -> None:
        self.embeddings = embeddings


def embedding_response(
    *,
    indexes: tuple[int, ...] = (0, 1),
    dimension: int = 64,
) -> CreateEmbeddingResponse:
    """Build a realistic OpenAI-compatible embedding response."""

    return CreateEmbeddingResponse.model_validate(
        {
            "id": "embed-1",
            "data": [
                {
                    "embedding": [float(index)] * dimension,
                    "index": index,
                    "object": "embedding",
                }
                for index in indexes
            ],
            "model": "text-embedding-v3",
            "object": "list",
            "usage": {"prompt_tokens": 6, "total_tokens": 6},
        }
    )


def build_model(
    embeddings: FakeEmbeddings,
    *,
    dimensions: int = 64,
    max_retries: int = 0,
) -> DashScopeEmbeddingModel:
    """Build an adapter with a deterministic fake provider client."""

    return DashScopeEmbeddingModel(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="text-embedding-v3",
        dimensions=dimensions,
        max_retries=max_retries,
        client=cast(AsyncOpenAI, FakeClient(embeddings)),
    )


def test_embed_maps_vectors_in_input_order() -> None:
    """Provider indexes should restore input order and preserve usage."""

    embeddings = FakeEmbeddings(embedding_response(indexes=(1, 0)))
    model = build_model(embeddings)

    response = asyncio.run(model.embed(["知识库", "文档检索"]))

    assert response.dimension == 64
    assert response.vectors[0][0] == 0
    assert response.vectors[1][0] == 1
    assert response.usage is not None
    assert response.usage.total_tokens == 6
    assert embeddings.requests[0] == {
        "model": "text-embedding-v3",
        "input": ["知识库", "文档检索"],
        "dimensions": 64,
        "encoding_format": "float",
    }


def test_embed_retries_transient_connection_errors() -> None:
    """Transient failures should use the same bounded retry policy as chat."""

    connection_error = APIConnectionError(
        request=httpx.Request("POST", "https://example.com/v1/embeddings")
    )
    embeddings = FakeEmbeddings(connection_error, embedding_response())
    model = build_model(embeddings, max_retries=1)

    response = asyncio.run(model.embed(["知识库", "文档检索"]))

    assert response.dimension == 64
    assert len(embeddings.requests) == 2


def test_embed_wraps_exhausted_provider_errors() -> None:
    """Provider SDK errors should not leak through the embedding port."""

    connection_error = APIConnectionError(
        request=httpx.Request("POST", "https://example.com/v1/embeddings")
    )
    model = build_model(FakeEmbeddings(connection_error))

    with pytest.raises(EmbeddingProviderError, match="APIConnectionError"):
        asyncio.run(model.embed(["知识库"]))


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        ([], "at least one"),
        ([""], "must not be empty"),
        (["文本"] * 11, "at most 10"),
    ],
)
def test_embed_validates_provider_input_limits(
    texts: list[str],
    expected: str,
) -> None:
    """Invalid batches should fail locally without consuming provider quota."""

    model = build_model(FakeEmbeddings(embedding_response()))

    with pytest.raises(ValueError, match=expected):
        asyncio.run(model.embed(texts))


def test_embed_rejects_mismatched_indexes() -> None:
    """Missing provider indexes should not silently corrupt text alignment."""

    model = build_model(FakeEmbeddings(embedding_response(indexes=(0, 2))))

    with pytest.raises(EmbeddingProviderError, match="indexes"):
        asyncio.run(model.embed(["知识库", "文档检索"]))


def test_embed_rejects_unexpected_dimension() -> None:
    """Unexpected vector dimensions should fail before vector-store insertion."""

    model = build_model(
        FakeEmbeddings(embedding_response(dimension=128)),
        dimensions=64,
    )

    with pytest.raises(EmbeddingProviderError, match="dimension"):
        asyncio.run(model.embed(["知识库", "文档检索"]))


def test_configuration_validates_model_dimensions() -> None:
    """Unsupported dimensions should fail during adapter construction."""

    with pytest.raises(EmbeddingConfigurationError, match="not supported"):
        DashScopeEmbeddingModel(
            api_key="test-key",
            base_url="https://example.com/v1",
            model="text-embedding-v3",
            dimensions=1536,
        )


def test_factory_requires_dashscope_api_key(monkeypatch: object) -> None:
    """The production adapter should never start without a provider key."""

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)  # type: ignore[attr-defined]
    with pytest.raises(EmbeddingConfigurationError, match="DASHSCOPE_API_KEY"):
        create_dashscope_embedding_model(Settings(_env_file=None))
