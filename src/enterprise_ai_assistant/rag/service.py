"""End-to-end retrieval, rerank, and grounded answer service."""

from html import escape
from pathlib import Path

from enterprise_ai_assistant.llm import ChatModel
from enterprise_ai_assistant.models import (
    RagAnswer,
    RagSource,
    RerankItem,
    VectorSearchFilter,
)
from enterprise_ai_assistant.prompt import PromptService
from enterprise_ai_assistant.rag.prompts import build_rag_answer_prompt
from enterprise_ai_assistant.rag.retriever import VectorRetriever
from enterprise_ai_assistant.rerank import Reranker


class RagService:
    """Answer questions using retrieval, reranking, and grounded generation."""

    def __init__(
        self,
        *,
        retriever: VectorRetriever,
        reranker: Reranker,
        chat_model: ChatModel,
        initial_top_k: int = 20,
        final_top_k: int = 5,
        max_context_chars: int = 8_000,
        max_answer_tokens: int = 1_200,
    ) -> None:
        """Create the RAG pipeline with explicit recall and context budgets."""

        if not 1 <= final_top_k <= initial_top_k <= 100:
            raise ValueError("RAG TopK must satisfy 1 <= final <= initial <= 100")
        if max_context_chars <= 0 or max_answer_tokens <= 0:
            raise ValueError("RAG context and answer budgets must be positive")
        self._retriever = retriever
        self._reranker = reranker
        self._prompt_service = PromptService(chat_model)
        self._initial_top_k = initial_top_k
        self._final_top_k = final_top_k
        self._max_context_chars = max_context_chars
        self._max_answer_tokens = max_answer_tokens
        self._answer_prompt = build_rag_answer_prompt()

    async def answer(
        self,
        question: str,
        *,
        search_filter: VectorSearchFilter | None = None,
    ) -> RagAnswer:
        """Return a grounded answer and the exact chunks supplied to the LLM."""

        if not question.strip():
            raise ValueError("RAG question must not be empty")
        candidates = await self._retriever.retrieve(
            question,
            top_k=self._initial_top_k,
            search_filter=search_filter,
        )
        if not candidates:
            return RagAnswer(
                answer="根据现有知识库无法确定。",
                sources=(),
                retrieved_count=0,
                model="no-context",
            )

        reranked = await self._reranker.rerank(
            question,
            candidates,
            top_n=min(self._final_top_k, len(candidates)),
        )
        sources = self._build_sources(reranked.items)
        if not sources:
            return RagAnswer(
                answer="根据现有知识库无法确定。",
                sources=(),
                retrieved_count=len(candidates),
                model="no-context",
            )
        context = self._build_context(sources)
        response = await self._prompt_service.run(
            self._answer_prompt,
            {"context": context, "question": question},
            temperature=0.1,
            max_tokens=self._max_answer_tokens,
        )
        return RagAnswer(
            answer=response.content,
            sources=sources,
            retrieved_count=len(candidates),
            model=response.model,
            usage=response.usage,
        )

    def _build_sources(
        self,
        items: tuple[RerankItem, ...],
    ) -> tuple[RagSource, ...]:
        sources: list[RagSource] = []
        used_chars = 0
        for item in items:
            content = item.candidate.content
            remaining = self._max_context_chars - used_chars
            if remaining <= 0:
                break
            if len(content) > remaining:
                content = content[:remaining]
            sources.append(
                RagSource(
                    reference=len(sources) + 1,
                    chunk_id=item.candidate.chunk_id,
                    source=item.candidate.source,
                    chunk_index=item.candidate.chunk_index,
                    score=item.score,
                    content=content,
                    metadata=item.candidate.metadata,
                )
            )
            used_chars += len(content)
        return tuple(sources)

    @staticmethod
    def _build_context(sources: tuple[RagSource, ...]) -> str:
        blocks = []
        for source in sources:
            page_number = source.metadata.get("page_number")
            location = Path(source.source).name
            if page_number is not None:
                location = f"{location}, 第{page_number}页"
            blocks.append(
                f'<source id="来源{source.reference}" '
                f'location="{escape(location, quote=True)}">\n'
                f"{escape(source.content)}\n</source>"
            )
        return "\n\n".join(blocks)
