"""Offline end-to-end integration test for the complete RAG pipeline."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from enterprise_ai_assistant.models import (
    ChatChunk,
    ChatMessage,
    ChatResponse,
    EmbeddingResponse,
    RerankItem,
    RerankResponse,
    ResponseFormat,
    RetrievalCandidate,
)
from enterprise_ai_assistant.rag import (
    IngestionService,
    RagService,
    TextChunker,
    VectorRetriever,
    create_document_loader_registry,
)
from enterprise_ai_assistant.vectorstore import MilvusVectorStore


class KeywordEmbeddingModel:
    """Map refund and photography text into separate semantic directions."""

    async def embed(self, texts: Sequence[str]) -> EmbeddingResponse:
        """Return normalized keyword vectors."""

        vectors = tuple((1.0, 0.0) if "退款" in text else (0.0, 1.0) for text in texts)
        return EmbeddingResponse(vectors=vectors, model="keyword-test")


class KeywordReranker:
    """Prefer candidates containing the query's refund keyword."""

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        top_n: int,
    ) -> RerankResponse:
        """Return deterministic semantic relevance order."""

        ordered = sorted(
            candidates,
            key=lambda item: "退款" in item.content,
            reverse=True,
        )
        return RerankResponse(
            items=tuple(
                RerankItem(candidate=item, score=1.0 - index * 0.1)
                for index, item in enumerate(ordered[:top_n])
            ),
            model="keyword-reranker",
        )

    async def close(self) -> None:
        """Satisfy the reranker protocol."""


class GroundedChatModel:
    """Return a stable answer after verifying source context is present."""

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: ResponseFormat = "text",
    ) -> ChatResponse:
        """Return a cited answer from the supplied context."""

        assert "三个工作日" in messages[-1].content
        return ChatResponse(
            content="退款将在三个工作日内到账。[来源1]",
            model="grounded-test",
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
            yield ChatChunk(content="", model="grounded-test")


def test_markdown_to_milvus_to_grounded_answer(tmp_path: Path) -> None:
    """A Markdown source should travel through every Phase 6 pipeline stage."""

    async def exercise() -> None:
        path = tmp_path / "enterprise-policy.md"
        path.write_text(
            "# 退款政策\n\n"
            + "企业客户退款将在三个工作日内原路到账。"
            + "如遇节假日可能顺延。"
            + "\n\n# 摄影设备\n\n"
            + "Sony A7R V 适合高像素商业摄影。",
            encoding="utf-8",
        )
        store = MilvusVectorStore(
            uri=str(tmp_path / "rag.db"),
            collection_name="knowledge_chunks",
            dimension=2,
        )
        embedder = KeywordEmbeddingModel()
        ingestion = IngestionService(
            loaders=create_document_loader_registry(),
            chunker=TextChunker(chunk_size=100, overlap=20),
            embedding_model=embedder,
            vector_store=store,
        )
        try:
            indexed = await ingestion.ingest(path)
            assert indexed.inserted_count == indexed.chunk_count

            service = RagService(
                retriever=VectorRetriever(
                    embedding_model=embedder,
                    vector_store=store,
                    default_top_k=5,
                ),
                reranker=KeywordReranker(),
                chat_model=GroundedChatModel(),
                initial_top_k=5,
                final_top_k=2,
            )
            answer = await service.answer("企业客户退款多久到账?")

            assert answer.answer.endswith("[来源1]")
            assert answer.sources[0].source == str(path.absolute())
            assert "三个工作日" in answer.sources[0].content
            assert answer.retrieved_count >= 1
        finally:
            await store.close()

    asyncio.run(exercise())
