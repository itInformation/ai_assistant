"""Lazy service container used by the FastAPI application."""

from __future__ import annotations

from dataclasses import dataclass

from enterprise_ai_assistant.agent import ConversationMemory, ReActAgent
from enterprise_ai_assistant.config import Settings
from enterprise_ai_assistant.embedding import (
    EmbeddingModel,
    create_dashscope_embedding_model,
)
from enterprise_ai_assistant.llm import (
    ChatModel,
    ToolCallingModel,
    create_dashscope_chat_model,
)
from enterprise_ai_assistant.rag import RagService, VectorRetriever
from enterprise_ai_assistant.rerank import Reranker, create_dashscope_reranker
from enterprise_ai_assistant.tools import ToolRegistry, create_tool_registry
from enterprise_ai_assistant.vectorstore import VectorStore, create_milvus_vector_store
from enterprise_ai_assistant.workflow import (
    AdvancedLangGraphWorkflow,
    BasicLangGraphWorkflow,
)


@dataclass(slots=True)
class ApiServiceOverrides:
    """Optional test doubles or externally managed services."""

    chat_model: ChatModel | None = None
    rag_service: RagService | None = None
    agent: ReActAgent | None = None
    basic_workflow: BasicLangGraphWorkflow | None = None
    advanced_workflow: AdvancedLangGraphWorkflow | None = None


class ApiContainer:
    """Create and cache real application services for API endpoints."""

    def __init__(
        self,
        settings: Settings,
        *,
        overrides: ApiServiceOverrides | None = None,
    ) -> None:
        self._settings = settings
        self._overrides = overrides or ApiServiceOverrides()
        self._chat_model: ChatModel | ToolCallingModel | None = None
        self._embedding_model: EmbeddingModel | None = None
        self._vector_store: VectorStore | None = None
        self._reranker: Reranker | None = None
        self._tool_registry: ToolRegistry | None = None
        self._rag_service: RagService | None = None
        self._agent: ReActAgent | None = None
        self._basic_workflow: BasicLangGraphWorkflow | None = None
        self._advanced_workflow: AdvancedLangGraphWorkflow | None = None

    def chat_model(self) -> ChatModel:
        """Return the configured chat model."""

        if self._overrides.chat_model is not None:
            return self._overrides.chat_model
        if self._chat_model is None:
            self._chat_model = create_dashscope_chat_model(self._settings)
        return self._chat_model

    def rag_service(self) -> RagService:
        """Return the RAG service used by `/api/v1/rag`."""

        if self._overrides.rag_service is not None:
            return self._overrides.rag_service
        if self._rag_service is None:
            self._rag_service = RagService(
                retriever=self._retriever(),
                reranker=self._reranker_or_create(),
                chat_model=self.chat_model(),
                initial_top_k=self._settings.rag_initial_top_k,
                final_top_k=self._settings.rag_final_top_k,
                max_context_chars=self._settings.rag_max_context_chars,
                max_answer_tokens=self._settings.rag_max_answer_tokens,
            )
        return self._rag_service

    def agent(self) -> ReActAgent:
        """Return the ReAct agent used by `/api/v1/agent`."""

        if self._overrides.agent is not None:
            return self._overrides.agent
        if self._agent is None:
            self._agent = ReActAgent(
                model=self._tool_calling_model(),
                tools=self._tools(),
                memory=ConversationMemory(
                    max_turns=self._settings.agent_memory_turns,
                    max_chars=self._settings.agent_memory_max_chars,
                ),
                max_tool_calls=self._settings.agent_max_tool_calls,
                max_observation_chars=self._settings.agent_max_observation_chars,
                max_answer_tokens=self._settings.agent_max_answer_tokens,
            )
        return self._agent

    def workflow(self, mode: str) -> BasicLangGraphWorkflow | AdvancedLangGraphWorkflow:
        """Return a compiled LangGraph workflow."""

        if mode == "advanced":
            if self._overrides.advanced_workflow is not None:
                return self._overrides.advanced_workflow
            if self._advanced_workflow is None:
                self._advanced_workflow = AdvancedLangGraphWorkflow(
                    chat_model=self.chat_model(),
                    retriever=self._retriever(),
                    tools=self._tools(),
                    retrieval_top_k=self._settings.workflow_retrieval_top_k,
                    max_answer_tokens=self._settings.workflow_max_answer_tokens,
                )
            return self._advanced_workflow
        if self._overrides.basic_workflow is not None:
            return self._overrides.basic_workflow
        if self._basic_workflow is None:
            self._basic_workflow = BasicLangGraphWorkflow(
                chat_model=self.chat_model(),
                retriever=self._retriever(),
                tools=self._tools(),
                retrieval_top_k=self._settings.workflow_retrieval_top_k,
                max_answer_tokens=self._settings.workflow_max_answer_tokens,
            )
        return self._basic_workflow

    async def close(self) -> None:
        """Close infrastructure resources owned by the container."""

        if self._tool_registry is not None:
            await self._tool_registry.close()
        if self._reranker is not None:
            await self._reranker.close()
        if self._vector_store is not None:
            await self._vector_store.close()

    def _tool_calling_model(self) -> ToolCallingModel:
        model = self.chat_model()
        if not hasattr(model, "chat_with_tools"):
            # The real DashScope adapter implements both protocols. Tests that
            # inject an agent never reach this branch.
            raise TypeError("configured chat model does not support tool calling")
        return model  # type: ignore[return-value]

    def _retriever(self) -> VectorRetriever:
        return VectorRetriever(
            embedding_model=self._embedding(),
            vector_store=self._vector_store_or_create(),
            default_top_k=self._settings.rag_initial_top_k,
        )

    def _embedding(self) -> EmbeddingModel:
        if self._embedding_model is None:
            self._embedding_model = create_dashscope_embedding_model(self._settings)
        return self._embedding_model

    def _vector_store_or_create(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = create_milvus_vector_store(self._settings)
        return self._vector_store

    def _reranker_or_create(self) -> Reranker:
        if self._reranker is None:
            self._reranker = create_dashscope_reranker(self._settings)
        return self._reranker

    def _tools(self) -> ToolRegistry:
        if self._tool_registry is None:
            self._tool_registry = create_tool_registry(self._settings)
        return self._tool_registry
