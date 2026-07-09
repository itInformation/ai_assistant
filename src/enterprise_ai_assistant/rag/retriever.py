"""Vector retrieval service."""

from enterprise_ai_assistant.embedding import EmbeddingModel
from enterprise_ai_assistant.models import (
    RetrievalCandidate,
    VectorSearchFilter,
)
from enterprise_ai_assistant.vectorstore import VectorStore


class VectorRetriever:
    """Embed a query and recall high-coverage Milvus candidates."""

    def __init__(
        self,
        *,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        default_top_k: int = 20,
    ) -> None:
        """Create a retriever with a bounded default recall count."""

        if not 1 <= default_top_k <= 100:
            raise ValueError("default_top_k must be between 1 and 100")
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._default_top_k = default_top_k

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        search_filter: VectorSearchFilter | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        """Recall semantically similar chunks in vector-score order."""

        if not query.strip():
            raise ValueError("retrieval query must not be empty")
        requested_top_k = top_k or self._default_top_k
        embedding = await self._embedding_model.embed([query])
        hits = await self._vector_store.search(
            embedding.vectors[0],
            top_k=requested_top_k,
            search_filter=search_filter,
        )
        return tuple(
            RetrievalCandidate(
                chunk_id=hit.id,
                document_id=hit.document_id,
                content=hit.content,
                source=hit.source,
                chunk_index=hit.chunk_index,
                vector_score=hit.score,
                metadata=hit.metadata,
            )
            for hit in hits
        )
