"""Tests for retrieval, reranking, prompting, and grounded answers."""

import asyncio
from collections.abc import AsyncIterator, Sequence

from enterprise_ai_assistant.models import (
    ChatChunk,
    ChatMessage,
    ChatResponse,
    RagAnswer,
    RerankItem,
    RerankResponse,
    ResponseFormat,
    RetrievalCandidate,
    VectorSearchFilter,
)
from enterprise_ai_assistant.rag import RagService


def candidate(index: int, content: str) -> RetrievalCandidate:
    """Build one retrieved candidate."""

    return RetrievalCandidate(
        chunk_id=f"chunk-{index}",
        document_id="doc-1",
        content=content,
        source="/knowledge/manual.pdf",
        chunk_index=index,
        vector_score=0.8,
        metadata={"page_number": index + 1},
    )


class FakeRetriever:
    """Return configured candidates."""

    def __init__(self, candidates: tuple[RetrievalCandidate, ...]) -> None:
        self.candidates = candidates
        self.top_k: int | None = None

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        search_filter: VectorSearchFilter | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        """Record recall size and return candidates."""

        self.top_k = top_k
        return self.candidates


class FakeReranker:
    """Reverse candidates to prove rerank order is respected."""

    def __init__(self) -> None:
        self.called = False
        self.top_n: int | None = None

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        top_n: int,
    ) -> RerankResponse:
        """Return candidates in reverse order."""

        self.called = True
        self.top_n = top_n
        items = tuple(
            RerankItem(candidate=item, score=0.9 - index * 0.1)
            for index, item in enumerate(reversed(candidates))
        )
        return RerankResponse(items=items[:top_n], model="fake-rerank")

    async def close(self) -> None:
        """Satisfy the reranker protocol."""


class FakeChatModel:
    """Record the grounded prompt and return a cited answer."""

    def __init__(self) -> None:
        self.messages: Sequence[ChatMessage] = ()
        self.called = False

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: ResponseFormat = "text",
    ) -> ChatResponse:
        """Record messages and return a stable answer."""

        self.called = True
        self.messages = messages
        return ChatResponse(
            content="退款三个工作日到账。[来源1]",
            model="fake-chat",
        )

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: ResponseFormat = "text",
    ) -> AsyncIterator[ChatChunk]:
        """Satisfy the ChatModel protocol."""

        if False:
            yield ChatChunk(content="", model="fake")


def build_service(
    candidates: tuple[RetrievalCandidate, ...],
) -> tuple[RagService, FakeRetriever, FakeReranker, FakeChatModel]:
    """Build the RAG service with deterministic test doubles."""

    retriever = FakeRetriever(candidates)
    reranker = FakeReranker()
    chat_model = FakeChatModel()
    service = RagService(
        retriever=retriever,  # type: ignore[arg-type]
        reranker=reranker,
        chat_model=chat_model,
        initial_top_k=20,
        final_top_k=2,
        max_context_chars=1_000,
    )
    return service, retriever, reranker, chat_model


def test_rag_service_uses_reranked_context_and_sources() -> None:
    """The LLM context and returned citations should share rerank order."""

    service, retriever, reranker, chat_model = build_service(
        (
            candidate(0, "普通说明"),
            candidate(1, "退款三个工作日到账"),
        )
    )

    answer = asyncio.run(service.answer("退款多久到账?"))

    assert answer.answer.endswith("[来源1]")
    assert answer.sources[0].chunk_id == "chunk-1"
    assert answer.sources[0].score == 0.9
    assert answer.retrieved_count == 2
    assert answer.model == "fake-chat"
    assert retriever.top_k == 20
    assert reranker.top_n == 2
    assert "退款三个工作日到账" in chat_model.messages[-1].content
    assert 'id="来源1"' in chat_model.messages[-1].content
    assert "不可信数据" in chat_model.messages[0].content


def test_rag_service_refuses_without_context() -> None:
    """No retrieval hit should produce a deterministic answer without LLM cost."""

    service, _, reranker, chat_model = build_service(())

    answer = asyncio.run(service.answer("未知问题"))

    assert answer == RagAnswer(
        answer="根据现有知识库无法确定。",
        sources=(),
        retrieved_count=0,
        model="no-context",
    )
    assert reranker.called is False
    assert chat_model.called is False
