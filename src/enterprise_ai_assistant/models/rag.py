"""Provider-independent retrieval, rerank, and RAG result models."""

from dataclasses import dataclass

from enterprise_ai_assistant.models.llm import TokenUsage
from enterprise_ai_assistant.models.vectorstore import JSONValue


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """A chunk recalled from vector search."""

    chunk_id: str
    document_id: str
    content: str
    source: str
    chunk_index: int
    vector_score: float
    metadata: dict[str, JSONValue]


@dataclass(frozen=True, slots=True)
class RerankItem:
    """A retrieval candidate with a cross-encoder relevance score."""

    candidate: RetrievalCandidate
    score: float


@dataclass(frozen=True, slots=True)
class RerankResponse:
    """Ordered rerank results and provider metadata."""

    items: tuple[RerankItem, ...]
    model: str
    request_id: str | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Summary of one document ingestion operation."""

    document_id: str
    source: str
    chunk_count: int
    replaced_count: int
    inserted_count: int


@dataclass(frozen=True, slots=True)
class RagSource:
    """One source chunk supplied to the answer model."""

    reference: int
    chunk_id: str
    source: str
    chunk_index: int
    score: float
    content: str
    metadata: dict[str, JSONValue]


@dataclass(frozen=True, slots=True)
class RagAnswer:
    """Final grounded answer with inspectable source chunks."""

    answer: str
    sources: tuple[RagSource, ...]
    retrieved_count: int
    model: str
    usage: TokenUsage | None = None
