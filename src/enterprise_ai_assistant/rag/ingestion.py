"""Application service for loading, chunking, embedding, and indexing files."""

from pathlib import Path

from enterprise_ai_assistant.embedding import EmbeddingModel
from enterprise_ai_assistant.models import (
    IngestionResult,
    VectorRecord,
)
from enterprise_ai_assistant.rag.chunking import TextChunker
from enterprise_ai_assistant.rag.loaders import DocumentLoaderRegistry
from enterprise_ai_assistant.vectorstore import VectorStore


class IngestionService:
    """Build and replace the searchable index for one source document."""

    def __init__(
        self,
        *,
        loaders: DocumentLoaderRegistry,
        chunker: TextChunker,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        embedding_batch_size: int = 10,
        insert_batch_size: int = 100,
    ) -> None:
        """Create a bounded ingestion pipeline."""

        if not 1 <= embedding_batch_size <= 10:
            raise ValueError("embedding_batch_size must be between 1 and 10")
        if insert_batch_size <= 0:
            raise ValueError("insert_batch_size must be positive")
        self._loaders = loaders
        self._chunker = chunker
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._embedding_batch_size = embedding_batch_size
        self._insert_batch_size = insert_batch_size

    async def ingest(self, path: Path) -> IngestionResult:
        """Parse and atomically prepare replacement chunks before indexing."""

        document = self._loaders.load(path)
        chunks = self._chunker.split(document)

        # Finish all external embedding calls before deleting the old index.
        # This narrows the non-transactional replacement window in Milvus.
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(chunks), self._embedding_batch_size):
            batch = chunks[start : start + self._embedding_batch_size]
            response = await self._embedding_model.embed(
                [chunk.content for chunk in batch]
            )
            if len(response.vectors) != len(batch):
                raise ValueError("embedding count does not match chunk count")
            vectors.extend(response.vectors)

        records = [
            VectorRecord(
                id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                embedding=vectors[index],
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                metadata=dict(chunk.metadata),
            )
            for index, chunk in enumerate(chunks)
        ]

        await self._vector_store.ensure_collection()
        replaced = await self._vector_store.delete_by_document_id(document.id)
        inserted_count = 0
        for start in range(0, len(records), self._insert_batch_size):
            result = await self._vector_store.insert(
                records[start : start + self._insert_batch_size]
            )
            inserted_count += result.inserted_count
        return IngestionResult(
            document_id=document.id,
            source=document.source,
            chunk_count=len(chunks),
            replaced_count=replaced.deleted_count,
            inserted_count=inserted_count,
        )
