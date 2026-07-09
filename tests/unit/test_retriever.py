"""Tests for query embedding and vector retrieval."""

import asyncio
from collections.abc import Sequence

from enterprise_ai_assistant.models import (
    EmbeddingResponse,
    VectorDeleteResult,
    VectorInsertResult,
    VectorRecord,
    VectorSearchFilter,
    VectorSearchResult,
)
from enterprise_ai_assistant.rag import VectorRetriever


class FakeEmbeddingModel:
    """Return a stable query vector."""

    async def embed(self, texts: Sequence[str]) -> EmbeddingResponse:
        """Return one vector for the query."""

        return EmbeddingResponse(vectors=((1.0, 0.0),), model="fake")


class FakeVectorStore:
    """Return one representative vector hit."""

    def __init__(self) -> None:
        self.top_k: int | None = None
        self.search_filter: VectorSearchFilter | None = None

    async def ensure_collection(self) -> None:
        """Satisfy the vector-store protocol."""

    async def insert(
        self,
        records: Sequence[VectorRecord],
    ) -> VectorInsertResult:
        """Satisfy the vector-store protocol."""

        return VectorInsertResult(0, ())

    async def delete(self, ids: Sequence[str]) -> VectorDeleteResult:
        """Satisfy the vector-store protocol."""

        return VectorDeleteResult(0)

    async def delete_by_document_id(
        self,
        document_id: str,
    ) -> VectorDeleteResult:
        """Satisfy the vector-store protocol."""

        return VectorDeleteResult(0)

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        search_filter: VectorSearchFilter | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Record search options and return a hit."""

        self.top_k = top_k
        self.search_filter = search_filter
        return (
            VectorSearchResult(
                id="chunk-1",
                score=0.9,
                document_id="doc-1",
                content="退款三个工作日到账",
                source="manual.md",
                chunk_index=2,
                metadata={"page_number": 3},
            ),
        )

    async def close(self) -> None:
        """Satisfy the vector-store protocol."""


def test_retriever_maps_vector_hits_to_candidates() -> None:
    """Retriever should preserve source metadata for rerank and citation."""

    store = FakeVectorStore()
    retriever = VectorRetriever(
        embedding_model=FakeEmbeddingModel(),
        vector_store=store,
        default_top_k=20,
    )
    search_filter = VectorSearchFilter(source="manual.md")

    candidates = asyncio.run(
        retriever.retrieve(
            "退款多久到账?",
            top_k=10,
            search_filter=search_filter,
        )
    )

    assert candidates[0].chunk_id == "chunk-1"
    assert candidates[0].vector_score == 0.9
    assert candidates[0].metadata["page_number"] == 3
    assert store.top_k == 10
    assert store.search_filter == search_filter
