"""Tests for the DashScope qwen3-rerank adapter."""

import asyncio

import httpx
import pytest

from enterprise_ai_assistant.config import Settings
from enterprise_ai_assistant.models import RetrievalCandidate
from enterprise_ai_assistant.rerank import (
    DashScopeReranker,
    RerankerConfigurationError,
    RerankerProviderError,
    create_dashscope_reranker,
)


def candidate(index: int) -> RetrievalCandidate:
    """Build one retrieval candidate."""

    return RetrievalCandidate(
        chunk_id=f"chunk-{index}",
        document_id="doc-1",
        content=f"候选内容 {index}",
        source="manual.md",
        chunk_index=index,
        vector_score=0.8 - index * 0.1,
        metadata={},
    )


def test_reranker_maps_indexes_and_scores() -> None:
    """Provider indexes should map scores back to complete candidates."""

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(
            200,
            json={
                "id": "rerank-1",
                "model": "qwen3-rerank",
                "results": [
                    {"index": 1, "relevance_score": 0.95},
                    {"index": 0, "relevance_score": 0.7},
                ],
                "usage": {"total_tokens": 12},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reranker = DashScopeReranker(
        api_key="test-key",
        url="https://example.com/reranks",
        client=client,
        max_retries=0,
    )

    response = asyncio.run(
        reranker.rerank("问题", [candidate(0), candidate(1)], top_n=2)
    )

    assert [item.candidate.chunk_id for item in response.items] == [
        "chunk-1",
        "chunk-0",
    ]
    assert response.items[0].score == 0.95
    assert response.request_id == "rerank-1"
    assert response.total_tokens == 12
    request_body = captured["request"].content.decode()
    assert '"model":"qwen3-rerank"' in request_body
    assert '"top_n":2' in request_body


def test_reranker_retries_server_errors() -> None:
    """Retryable HTTP failures should use the configured bounded policy."""

    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, json={"message": "busy"})
        return httpx.Response(
            200,
            json={
                "results": [{"index": 0, "relevance_score": 0.8}],
                "model": "qwen3-rerank",
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reranker = DashScopeReranker(
        api_key="test-key",
        url="https://example.com/reranks",
        client=client,
        max_retries=1,
    )

    result = asyncio.run(reranker.rerank("问题", [candidate(0)], top_n=1))

    assert result.items[0].score == 0.8
    assert attempts == 2


def test_reranker_wraps_invalid_provider_indexes() -> None:
    """Out-of-range indexes should not silently attach scores to wrong chunks."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [{"index": 4, "relevance_score": 0.8}],
                "model": "qwen3-rerank",
            },
        )

    reranker = DashScopeReranker(
        api_key="test-key",
        url="https://example.com/reranks",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )

    with pytest.raises(RerankerProviderError, match="invalid candidate indexes"):
        asyncio.run(reranker.rerank("问题", [candidate(0)], top_n=1))


@pytest.mark.parametrize(
    ("query", "candidates", "top_n", "expected"),
    [
        ("", [candidate(0)], 1, "query"),
        ("问题", [], 1, "candidate"),
        ("问题", [candidate(0)], 2, "top_n"),
    ],
)
def test_reranker_validates_input(
    query: str,
    candidates: list[RetrievalCandidate],
    top_n: int,
    expected: str,
) -> None:
    """Invalid rerank requests should fail before consuming provider quota."""

    reranker = DashScopeReranker(
        api_key="test-key",
        url="https://example.com/reranks",
        client=httpx.AsyncClient(),
    )

    with pytest.raises(ValueError, match=expected):
        asyncio.run(reranker.rerank(query, candidates, top_n=top_n))


def test_factory_requires_api_key(monkeypatch: object) -> None:
    """Production reranker construction should require provider credentials."""

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)  # type: ignore[attr-defined]

    with pytest.raises(RerankerConfigurationError, match="DASHSCOPE_API_KEY"):
        create_dashscope_reranker(Settings(_env_file=None))
