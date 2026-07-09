"""Provider-independent vector-store interface."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from enterprise_ai_assistant.models import (
    VectorDeleteResult,
    VectorInsertResult,
    VectorRecord,
    VectorSearchFilter,
    VectorSearchResult,
)


@runtime_checkable
class VectorStore(Protocol):
    """Port implemented by local or remote vector databases."""

    async def ensure_collection(self) -> None:
        """Create the collection when absent and validate it when present."""

        ...

    async def insert(
        self,
        records: Sequence[VectorRecord],
    ) -> VectorInsertResult:
        """Insert document chunk vectors."""

        ...

    async def delete(self, ids: Sequence[str]) -> VectorDeleteResult:
        """Delete records by their application-owned primary keys."""

        ...

    async def delete_by_document_id(
        self,
        document_id: str,
    ) -> VectorDeleteResult:
        """Delete all chunks belonging to one logical document."""

        ...

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        search_filter: VectorSearchFilter | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Return the nearest document chunks ordered by similarity."""

        ...

    async def close(self) -> None:
        """Release vector-store client resources."""

        ...
