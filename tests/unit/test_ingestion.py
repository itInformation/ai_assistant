"""Tests for document loading, chunking, embedding, and indexing."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

from enterprise_ai_assistant.models import (
    EmbeddingResponse,
    VectorDeleteResult,
    VectorInsertResult,
    VectorRecord,
    VectorSearchFilter,
    VectorSearchResult,
)
from enterprise_ai_assistant.rag import (
    IngestionService,
    TextChunker,
    create_document_loader_registry,
)


class FakeEmbeddingModel:
    """Return deterministic two-dimensional vectors."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.batch_sizes: list[int] = []

    async def embed(self, texts: Sequence[str]) -> EmbeddingResponse:
        """Record batch order and return one vector per text."""

        self.events.append("embed")
        self.batch_sizes.append(len(texts))
        return EmbeddingResponse(
            vectors=tuple((1.0, float(index)) for index, _ in enumerate(texts)),
            model="fake",
        )


class FakeVectorStore:
    """Record document replacement and insert batches."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.document_id: str | None = None
        self.records: list[VectorRecord] = []
        self.insert_batch_sizes: list[int] = []

    async def ensure_collection(self) -> None:
        """Record collection readiness."""

        self.events.append("ensure")

    async def insert(
        self,
        records: Sequence[VectorRecord],
    ) -> VectorInsertResult:
        """Record inserted records."""

        self.events.append("insert")
        self.records.extend(records)
        self.insert_batch_sizes.append(len(records))
        return VectorInsertResult(
            inserted_count=len(records),
            primary_keys=tuple(record.id for record in records),
        )

    async def delete(self, ids: Sequence[str]) -> VectorDeleteResult:
        """Satisfy the shared vector-store protocol."""

        return VectorDeleteResult(deleted_count=len(ids))

    async def delete_by_document_id(
        self,
        document_id: str,
    ) -> VectorDeleteResult:
        """Record document-level replacement."""

        self.events.append("delete")
        self.document_id = document_id
        return VectorDeleteResult(deleted_count=2)

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        search_filter: VectorSearchFilter | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Satisfy the shared vector-store protocol."""

        return ()

    async def close(self) -> None:
        """Satisfy the shared vector-store protocol."""


def test_ingestion_batches_embedding_then_replaces_document(
    tmp_path: Path,
) -> None:
    """All embeddings should finish before old chunks are deleted."""

    path = tmp_path / "manual.md"
    path.write_text("企业知识。" * 80, encoding="utf-8")
    events: list[str] = []
    embedder = FakeEmbeddingModel(events)
    store = FakeVectorStore(events)
    service = IngestionService(
        loaders=create_document_loader_registry(),
        chunker=TextChunker(chunk_size=100, overlap=20),
        embedding_model=embedder,
        vector_store=store,
        embedding_batch_size=2,
        insert_batch_size=3,
    )

    result = asyncio.run(service.ingest(path))

    assert result.chunk_count > 3
    assert result.replaced_count == 2
    assert result.inserted_count == result.chunk_count
    assert all(size <= 2 for size in embedder.batch_sizes)
    assert all(size <= 3 for size in store.insert_batch_sizes)
    assert events.index("delete") > max(
        index for index, event in enumerate(events) if event == "embed"
    )
    assert store.records[0].metadata["file_type"] == "md"
    assert store.document_id == result.document_id
