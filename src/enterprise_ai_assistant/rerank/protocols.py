"""Provider-independent reranker interface."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from enterprise_ai_assistant.models import RerankResponse, RetrievalCandidate


@runtime_checkable
class Reranker(Protocol):
    """Port implemented by semantic or hybrid reranking providers."""

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        top_n: int,
    ) -> RerankResponse:
        """Rank candidates by their ability to answer the query."""

        ...

    async def close(self) -> None:
        """Release provider client resources."""

        ...
