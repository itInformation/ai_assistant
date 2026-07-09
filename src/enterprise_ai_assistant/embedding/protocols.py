"""Provider-independent embedding interfaces."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from enterprise_ai_assistant.models import EmbeddingResponse


@runtime_checkable
class EmbeddingModel(Protocol):
    """Port implemented by text embedding providers."""

    async def embed(
        self,
        texts: Sequence[str],
    ) -> EmbeddingResponse:
        """Convert ordered texts into ordered dense vectors."""

        ...
