"""Tests for the local command-line entry point."""

import asyncio
import json
import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

from enterprise_ai_assistant import cli
from enterprise_ai_assistant.config import Settings
from enterprise_ai_assistant.models import (
    AgentMessage,
    AgentModelResponse,
    AgentToolCall,
    ChatChunk,
    ChatMessage,
    ChatResponse,
    EmbeddingResponse,
    EmbeddingUsage,
    JSONValue,
    RerankItem,
    RerankResponse,
    ResponseFormat,
    RetrievalCandidate,
    ToolResult,
    ToolSpec,
    VectorDeleteResult,
    VectorInsertResult,
    VectorRecord,
    VectorSearchFilter,
    VectorSearchResult,
)
from enterprise_ai_assistant.tools import ToolRegistry


class FakeChatModel:
    """Deterministic chat model used by CLI tests."""

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: ResponseFormat = "text",
    ) -> ChatResponse:
        """Return a stable complete response."""

        if response_format == "json_object":
            return ChatResponse(
                content=(
                    '{"category":"售后","urgency":"中",' '"reason":"用户询问退款进度"}'
                ),
                model="fake",
            )
        return ChatResponse(content=f"回复: {messages[0].content}", model="fake")

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: ResponseFormat = "text",
    ) -> AsyncIterator[ChatChunk]:
        """Yield a stable response in two chunks."""

        for content in ("流式", "回复"):
            yield ChatChunk(content=content, model="fake")


class FakeEmbeddingModel:
    """Deterministic embedding model used by CLI tests."""

    async def embed(self, texts: Sequence[str]) -> EmbeddingResponse:
        """Return one small vector per input text."""

        vectors = (
            (1.0, 0.0),
            (0.8, 0.6),
            (0.0, 1.0),
            (0.9, 0.1),
        )[: len(texts)]
        return EmbeddingResponse(
            vectors=vectors,
            model="fake-embedding",
            usage=EmbeddingUsage(prompt_tokens=4, total_tokens=4),
        )


class FakeVectorStore:
    """In-memory vector store used by the Phase 5 CLI test."""

    def __init__(self) -> None:
        self.records: Sequence[VectorRecord] = ()
        self.deleted_ids: Sequence[str] = ()
        self.deleted_document_id: str | None = None

    async def ensure_collection(self) -> None:
        """Satisfy collection initialization."""

    async def insert(
        self,
        records: Sequence[VectorRecord],
    ) -> VectorInsertResult:
        """Record inserted entities."""

        self.records = records
        return VectorInsertResult(
            inserted_count=len(records),
            primary_keys=tuple(record.id for record in records),
        )

    async def delete(self, ids: Sequence[str]) -> VectorDeleteResult:
        """Record deleted primary keys."""

        self.deleted_ids = ids
        return VectorDeleteResult(deleted_count=len(ids))

    async def delete_by_document_id(
        self,
        document_id: str,
    ) -> VectorDeleteResult:
        """Record document-level replacement deletion."""

        self.deleted_document_id = document_id
        return VectorDeleteResult(deleted_count=0)

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        search_filter: VectorSearchFilter | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Return one deterministic search result."""

        record = self.records[min(1, len(self.records) - 1)]
        return (
            VectorSearchResult(
                id=record.id,
                score=0.9,
                document_id=record.document_id,
                content=record.content,
                source=record.source,
                chunk_index=record.chunk_index,
                metadata=record.metadata,
            ),
        )

    async def close(self) -> None:
        """Satisfy the vector-store lifecycle."""


class FakeReranker:
    """Return candidates in their current order."""

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        top_n: int,
    ) -> RerankResponse:
        """Assign deterministic relevance scores."""

        return RerankResponse(
            items=tuple(
                RerankItem(candidate=item, score=0.9) for item in candidates[:top_n]
            ),
            model="fake-rerank",
        )

    async def close(self) -> None:
        """Satisfy the reranker lifecycle."""


class FakeTool:
    """Deterministic tool used by CLI tests."""

    spec = ToolSpec(
        name="weather",
        description="Fake weather.",
        parameters={"type": "object", "properties": {}},
    )

    async def invoke(
        self,
        arguments: Mapping[str, JSONValue],
    ) -> ToolResult:
        return ToolResult("weather", "晴", dict(arguments))

    async def close(self) -> None:
        """Satisfy the tool lifecycle."""


class FakeAgentModel:
    """Return one tool call followed by a final answer."""

    def __init__(self) -> None:
        self.turn = 0

    async def chat_with_tools(
        self,
        messages: Sequence[AgentMessage],
        tools: Sequence[ToolSpec],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AgentModelResponse:
        self.turn += 1
        if self.turn == 1:
            return AgentModelResponse(
                message=AgentMessage(
                    role="assistant",
                    content="需要查询天气。",
                    tool_calls=(
                        AgentToolCall(
                            id="call-1",
                            name="weather",
                            arguments={"location": "北京"},
                        ),
                    ),
                ),
                model="fake-agent",
            )
        return AgentModelResponse(
            message=AgentMessage(role="assistant", content="北京天气晴。"),
            model="fake-agent",
        )


def test_main_prints_structured_application_info(
    monkeypatch: object,
    capsys: object,
) -> None:
    """CLI should initialize the app and finish with a JSON result."""

    settings = Settings(app_name="Knowledge Assistant", app_env="testing")
    monkeypatch.setattr(cli, "get_settings", lambda: settings)  # type: ignore[attr-defined]

    cli.main([])

    output_lines = capsys.readouterr().out.strip().splitlines()  # type: ignore[attr-defined]
    result = json.loads(output_lines[-1])
    assert result == {
        "name": "Knowledge Assistant",
        "environment": "testing",
        "status": "ready",
        "version": "0.1.0",
    }


def test_run_chat_demo_prints_complete_response(capsys: object) -> None:
    """Non-streaming chat should print the complete model response."""

    asyncio.run(
        cli.run_chat_demo(
            Settings(_env_file=None),
            "你好",
            stream=False,
            model=FakeChatModel(),
        )
    )

    assert capsys.readouterr().out == "回复: 你好\n"  # type: ignore[attr-defined]


def test_run_chat_demo_prints_streaming_chunks(capsys: object) -> None:
    """Streaming chat should concatenate incremental chunks."""

    asyncio.run(
        cli.run_chat_demo(
            Settings(_env_file=None),
            "你好",
            stream=True,
            model=FakeChatModel(),
        )
    )

    assert capsys.readouterr().out == "流式回复\n"  # type: ignore[attr-defined]


def test_main_runs_chat_command(
    monkeypatch: object,
    capsys: object,
) -> None:
    """The chat subcommand should resolve and invoke the configured adapter."""

    settings = Settings(app_env="testing")
    monkeypatch.setattr(cli, "get_settings", lambda: settings)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli,
        "create_dashscope_chat_model",
        lambda current_settings: FakeChatModel(),
    )

    cli.main(["chat", "你好"])

    assert capsys.readouterr().out.endswith("回复: 你好\n")  # type: ignore[attr-defined]


def test_run_prompt_demo_supports_offline_debug(capsys: object) -> None:
    """Prompt debug should expose rendered messages without calling a model."""

    asyncio.run(
        cli.run_prompt_demo(
            Settings(_env_file=None),
            "退款什么时候到账?",
            debug=True,
        )
    )

    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["name"] == "support-ticket-classification"
    assert output["version"] == 1
    assert output["message_count"] == 6


def test_run_prompt_demo_prints_validated_json(capsys: object) -> None:
    """Prompt demo should print the schema-validated business result."""

    asyncio.run(
        cli.run_prompt_demo(
            Settings(_env_file=None),
            "退款什么时候到账?",
            debug=False,
            model=FakeChatModel(),
        )
    )

    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["category"] == "售后"
    assert output["urgency"] == "中"


def test_run_embedding_demo_prints_vector_summary(capsys: object) -> None:
    """Embedding demo should summarize vectors without dumping full arrays."""

    asyncio.run(
        cli.run_embedding_demo(
            Settings(_env_file=None),
            ["企业知识库", "公司文档检索"],
            model=FakeEmbeddingModel(),
        )
    )

    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["model"] == "fake-embedding"
    assert output["count"] == 2
    assert output["dimension"] == 2
    assert output["cosine_similarity"] == 0.8
    assert output["vectors"][0]["preview"] == [1.0, 0.0]


def test_run_vectorstore_demo_covers_insert_search_delete(
    capsys: object,
) -> None:
    """Milvus Demo should clean up only the records it inserted."""

    store = FakeVectorStore()
    asyncio.run(
        cli.run_vectorstore_demo(
            Settings(
                _env_file=None,
                dashscope_embedding_dimensions=2,
            ),
            "如何检索企业文档?",
            embedding_model=FakeEmbeddingModel(),
            vector_store=store,
        )
    )

    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["inserted_count"] == 3
    assert output["deleted_count"] == 3
    assert output["results"][0]["score"] == 0.9
    assert tuple(store.deleted_ids) == tuple(record.id for record in store.records)


def test_run_ingestion_demo_indexes_markdown(
    tmp_path: Path,
    capsys: object,
) -> None:
    """Ingestion CLI should parse, embed, and persist a source document."""

    path = tmp_path / "manual.md"
    path.write_text("退款将在三个工作日内到账。", encoding="utf-8")
    store = FakeVectorStore()
    asyncio.run(
        cli.run_ingestion_demo(
            Settings(
                _env_file=None,
                dashscope_embedding_dimensions=2,
            ),
            path,
            embedding_model=FakeEmbeddingModel(),
            vector_store=store,
        )
    )

    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["chunk_count"] == 1
    assert output["inserted_count"] == 1
    assert store.records[0].content == "退款将在三个工作日内到账。"


def test_run_rag_demo_returns_sources(capsys: object) -> None:
    """RAG CLI should return a grounded answer and source metadata."""

    store = FakeVectorStore()
    store.records = (
        VectorRecord(
            id="doc-1-0",
            document_id="doc-1",
            content="退款将在三个工作日内到账。",
            embedding=(1.0, 0.0),
            source="manual.md",
            chunk_index=0,
            metadata={"page_number": 1},
        ),
    )
    asyncio.run(
        cli.run_rag_demo(
            Settings(
                _env_file=None,
                dashscope_embedding_dimensions=2,
                rag_initial_top_k=1,
                rag_final_top_k=1,
            ),
            "退款多久到账?",
            embedding_model=FakeEmbeddingModel(),
            vector_store=store,
            reranker=FakeReranker(),
            chat_model=FakeChatModel(),
        )
    )

    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["retrieved_count"] == 1
    assert output["sources"][0]["source"] == "manual.md"


def test_run_tool_demo_dispatches_json_arguments(capsys: object) -> None:
    """Tool Demo should parse JSON and invoke the Registry allow-list."""

    asyncio.run(
        cli.run_tool_demo(
            Settings(_env_file=None),
            "weather",
            '{"location":"北京"}',
            registry=ToolRegistry([FakeTool()]),
        )
    )

    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["tool_name"] == "weather"
    assert output["data"] == {"location": "北京"}


def test_list_tools_prints_json_schema(capsys: object) -> None:
    """Tool discovery should expose the registered contract."""

    cli.list_tools(
        Settings(_env_file=None),
        registry=ToolRegistry([FakeTool()]),
    )

    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output[0]["name"] == "weather"
    assert output[0]["parameters"]["type"] == "object"


def test_run_agent_demo_prints_react_trace(capsys: object) -> None:
    """Agent Demo should print required Thought, Action, and Observation."""

    asyncio.run(
        cli.run_agent_demo(
            Settings(_env_file=None),
            ["北京天气?"],
            model=FakeAgentModel(),
            registry=ToolRegistry([FakeTool()]),
        )
    )

    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "Thought: 需要查询天气。" in output
    assert 'Action: {"tool": "weather"' in output
    assert "Observation:" in output
    assert "Final Answer: 北京天气晴。" in output


def test_run_workflow_demo_prints_langgraph_trace(capsys: object) -> None:
    """Workflow Demo should print checkpoint, node summaries, and final answer."""

    store = FakeVectorStore()
    store.records = (
        VectorRecord(
            id="doc-1-0",
            document_id="doc-1",
            content="北京办公室访客需要提前预约。",
            embedding=(1.0, 0.0),
            source="policy.md",
            chunk_index=0,
            metadata={},
        ),
    )

    class WorkflowChatModel(FakeChatModel):
        async def chat(
            self,
            messages: Sequence[ChatMessage],
            *,
            temperature: float | None = None,
            max_tokens: int | None = None,
            response_format: ResponseFormat = "text",
        ) -> ChatResponse:
            if response_format == "json_object" and "Planner" in messages[0].content:
                return ChatResponse(
                    content=(
                        '{"goal":"回答出行建议",'
                        '"retrieval_query":"北京 出行",'
                        '"tool_name":"weather",'
                        '"tool_arguments":{"location":"北京"}}'
                    ),
                    model="fake",
                )
            if response_format == "json_object" and "Reviewer" in messages[0].content:
                return ChatResponse(
                    content='{"passed":true,"feedback":"证据足够。"}',
                    model="fake",
                )
            return ChatResponse(content="北京适合散步。", model="fake")

    asyncio.run(
        cli.run_workflow_demo(
            Settings(
                _env_file=None,
                dashscope_embedding_dimensions=2,
                workflow_retrieval_top_k=1,
            ),
            "basic",
            "北京适合散步吗?",
            chat_model=WorkflowChatModel(),
            embedding_model=FakeEmbeddingModel(),
            vector_store=store,
            registry=ToolRegistry([FakeTool()]),
        )
    )

    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "Checkpoint: workflow-" in output
    assert "Node: planner" in output
    assert "Node: answer" in output
    assert "Final Answer: 北京适合散步。" in output


def test_run_api_server_passes_settings_to_uvicorn(monkeypatch: object) -> None:
    """Serve command should delegate host and port to Uvicorn."""

    calls: list[dict[str, object]] = []

    def fake_run(app_path: str, **kwargs: object) -> None:
        calls.append({"app_path": app_path, **kwargs})

    monkeypatch.setitem(  # type: ignore[attr-defined]
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=fake_run),
    )

    cli.run_api_server(
        Settings(_env_file=None, api_host="0.0.0.0", api_port=9000),
        reload=True,
    )

    assert calls == [
        {
            "app_path": "enterprise_ai_assistant.api.app:app",
            "host": "0.0.0.0",
            "port": 9000,
            "reload": True,
        }
    ]
