"""Provider-independent models for text embeddings."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingUsage:
    """Token usage reported by an embedding provider."""

    prompt_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    """Ordered dense vectors and metadata returned by an embedding model."""

    vectors: tuple[tuple[float, ...], ...]
    model: str
    request_id: str | None = None
    usage: EmbeddingUsage | None = None

    def __post_init__(self) -> None:
        """Ensure downstream vector stores receive a rectangular matrix."""

        if not self.vectors:
            raise ValueError("embedding response must contain at least one vector")
        dimension = len(self.vectors[0])
        if dimension == 0:
            raise ValueError("embedding vectors must not be empty")
        if any(len(vector) != dimension for vector in self.vectors):
            raise ValueError("embedding vectors must have the same dimension")

    @property
    def dimension(self) -> int:
        """Return the common vector dimension."""

        return len(self.vectors[0])
