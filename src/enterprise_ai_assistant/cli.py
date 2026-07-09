"""Command-line entry point for local development."""

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict
from math import sqrt
from pathlib import Path
from uuid import uuid4

from enterprise_ai_assistant.agent import ConversationMemory, ReActAgent
from enterprise_ai_assistant.app import run_demo
from enterprise_ai_assistant.config import Settings, get_settings
from enterprise_ai_assistant.core import configure_logging
from enterprise_ai_assistant.embedding import (
    EmbeddingModel,
    create_dashscope_embedding_model,
)
from enterprise_ai_assistant.llm import (
    ChatModel,
    ToolCallingModel,
    create_dashscope_chat_model,
)
from enterprise_ai_assistant.models import (
    ChatMessage,
    ToolResult,
    VectorRecord,
    VectorSearchFilter,
)
from enterprise_ai_assistant.prompt import (
    PromptRegistry,
    PromptService,
    StructuredOutputParser,
)
from enterprise_ai_assistant.prompt.examples import (
    SupportTicketClassification,
    build_support_classification_prompt,
)
from enterprise_ai_assistant.rag import (
    IngestionService,
    RagService,
    TextChunker,
    VectorRetriever,
    create_document_loader_registry,
)
from enterprise_ai_assistant.rerank import (
    Reranker,
    create_dashscope_reranker,
)
from enterprise_ai_assistant.tools import ToolRegistry, create_tool_registry
from enterprise_ai_assistant.vectorstore import (
    VectorStore,
    create_milvus_vector_store,
)
from enterprise_ai_assistant.workflow import (
    AdvancedLangGraphWorkflow,
    BasicLangGraphWorkflow,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the local command-line interface."""

    parser = argparse.ArgumentParser(description="企业级 AI 知识助手")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("health", help="显示本地应用健康信息")
    chat_parser = subparsers.add_parser("chat", help="调用 DashScope 进行最小聊天")
    chat_parser.add_argument("prompt", help="发送给模型的用户消息")
    chat_parser.add_argument(
        "--stream",
        action="store_true",
        help="以增量流式方式输出回复",
    )
    prompt_parser = subparsers.add_parser(
        "prompt",
        help="运行结构化 Prompt 分类 Demo",
    )
    prompt_parser.add_argument("question", help="需要分类的客服问题")
    prompt_parser.add_argument(
        "--debug",
        action="store_true",
        help="只渲染并检查 Prompt, 不调用模型",
    )
    embedding_parser = subparsers.add_parser(
        "embed",
        help="调用 DashScope 生成文本向量",
    )
    embedding_parser.add_argument(
        "texts",
        nargs="+",
        help="需要向量化的一到十段文本",
    )
    vector_parser = subparsers.add_parser(
        "vector-demo",
        help="运行 Milvus Lite 增删查 Demo",
    )
    vector_parser.add_argument(
        "query",
        nargs="?",
        default="如何使用向量数据库检索企业文档?",
        help="用于相似度检索的问题",
    )
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="解析并索引 PDF、DOCX 或 Markdown 文档",
    )
    ingest_parser.add_argument("path", type=Path, help="需要摄取的文档路径")
    rag_parser = subparsers.add_parser(
        "rag",
        help="检索、精排并回答企业知识问题",
    )
    rag_parser.add_argument("question", help="需要从知识库回答的问题")
    subparsers.add_parser("tools", help="列出已注册工具及其输入 Schema")
    tool_parser = subparsers.add_parser(
        "tool",
        help="通过 Registry 调用指定工具",
    )
    tool_parser.add_argument("name", choices=("weather", "search", "database"))
    tool_parser.add_argument(
        "arguments",
        help='JSON 对象, 例如 \'{"location":"北京"}\'',
    )
    agent_parser = subparsers.add_parser(
        "agent",
        help="运行带 Memory 和 Tool Calling 的 ReAct Agent",
    )
    agent_parser.add_argument(
        "questions",
        nargs="+",
        help="一个或多个连续问题; 多个问题共享本次进程内 Memory",
    )
    workflow_parser = subparsers.add_parser(
        "workflow",
        help="运行 Phase 9 LangGraph 工作流 Demo",
    )
    workflow_parser.add_argument(
        "mode",
        choices=("basic", "advanced"),
        help="basic=Planner/Retriever/Tool/Reviewer/Answer; advanced=多 Agent 编排",
    )
    workflow_parser.add_argument("question", help="需要工作流回答的问题")
    serve_parser = subparsers.add_parser(
        "serve",
        help="启动 Phase 10 FastAPI 服务",
    )
    serve_parser.add_argument("--host", default=None, help="监听地址")
    serve_parser.add_argument("--port", type=int, default=None, help="监听端口")
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="开发模式下自动重载",
    )
    return parser


async def run_chat_demo(
    settings: Settings,
    prompt: str,
    *,
    stream: bool,
    model: ChatModel | None = None,
) -> None:
    """Run the Phase 2 DashScope chat demo."""

    chat_model = model or create_dashscope_chat_model(settings)
    messages = [ChatMessage(role="user", content=prompt)]
    if stream:
        async for chunk in chat_model.stream_chat(messages):
            print(chunk.content, end="", flush=True)
        print()
        return

    response = await chat_model.chat(messages)
    print(response.content)


async def run_prompt_demo(
    settings: Settings,
    question: str,
    *,
    debug: bool,
    model: ChatModel | None = None,
) -> None:
    """Run the Phase 3 versioned and structured prompt demo."""

    registry = PromptRegistry()
    registry.register(build_support_classification_prompt())
    template = registry.get("support-ticket-classification")
    variables = {"question": question}
    if debug:
        print(
            json.dumps(
                asdict(template.debug(**variables)),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    chat_model = model or create_dashscope_chat_model(settings)
    parser = StructuredOutputParser(SupportTicketClassification)
    result = await PromptService(chat_model).run_structured(
        template,
        variables,
        parser,
    )
    print(result.value.model_dump_json(indent=2))


async def run_embedding_demo(
    settings: Settings,
    texts: Sequence[str],
    *,
    model: EmbeddingModel | None = None,
) -> None:
    """Run the Phase 4 DashScope embedding demo."""

    embedding_model = model or create_dashscope_embedding_model(settings)
    response = await embedding_model.embed(texts)
    output: dict[str, object] = {
        "model": response.model,
        "count": len(response.vectors),
        "dimension": response.dimension,
        "usage": asdict(response.usage) if response.usage else None,
        # A short preview proves vectors were generated without dumping 1024
        # floating-point values per text into logs or terminal history.
        "vectors": [
            {
                "index": index,
                "norm": _vector_norm(vector),
                "preview": vector[:8],
            }
            for index, vector in enumerate(response.vectors)
        ],
    }
    if len(response.vectors) == 2:
        output["cosine_similarity"] = _cosine_similarity(
            response.vectors[0],
            response.vectors[1],
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


def _vector_norm(vector: Sequence[float]) -> float:
    """Calculate the Euclidean norm of one dense vector."""

    return sqrt(sum(value * value for value in vector))


def _cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Calculate cosine similarity for two equal-length dense vectors."""

    left_norm = _vector_norm(left)
    right_norm = _vector_norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )


async def run_vectorstore_demo(
    settings: Settings,
    query: str,
    *,
    embedding_model: EmbeddingModel | None = None,
    vector_store: VectorStore | None = None,
) -> None:
    """Run the Phase 5 Milvus Lite collection, insert, search, and delete demo."""

    texts = [
        "RAG 通过检索企业文档为大模型补充外部知识。",
        "Milvus 是支持向量相似度搜索和元数据过滤的向量数据库。",
        "Sony A7R V 是一台高像素全画幅微单相机。",
    ]
    embedder = embedding_model or create_dashscope_embedding_model(settings)
    store = vector_store or create_milvus_vector_store(settings)
    owns_store = vector_store is None
    embedding_response = await embedder.embed([*texts, query])
    document_id = f"phase5-demo-{uuid4().hex}"
    records = [
        VectorRecord(
            id=f"{document_id}-{index}",
            document_id=document_id,
            content=text,
            embedding=embedding_response.vectors[index],
            source="phase5-demo",
            chunk_index=index,
            metadata={"topic": "photography" if index == 2 else "ai"},
        )
        for index, text in enumerate(texts)
    ]
    inserted_ids: tuple[str, ...] = ()
    try:
        await store.ensure_collection()
        insert_result = await store.insert(records)
        inserted_ids = insert_result.primary_keys
        hits = await store.search(
            embedding_response.vectors[-1],
            top_k=2,
            search_filter=VectorSearchFilter(source="phase5-demo"),
        )
        output = {
            "collection": settings.milvus_collection_name,
            "inserted_count": insert_result.inserted_count,
            "results": [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "content": hit.content,
                    "metadata": hit.metadata,
                }
                for hit in hits
            ],
        }
    finally:
        if inserted_ids:
            delete_result = await store.delete(inserted_ids)
            output["deleted_count"] = delete_result.deleted_count
        if owns_store:
            await store.close()
    print(json.dumps(output, ensure_ascii=False, indent=2))


async def run_ingestion_demo(
    settings: Settings,
    path: Path,
    *,
    embedding_model: EmbeddingModel | None = None,
    vector_store: VectorStore | None = None,
) -> None:
    """Run the Phase 6 document ingestion pipeline."""

    embedder = embedding_model or create_dashscope_embedding_model(settings)
    store = vector_store or create_milvus_vector_store(settings)
    owns_store = vector_store is None
    service = IngestionService(
        loaders=create_document_loader_registry(
            max_file_size_mb=settings.rag_max_file_size_mb
        ),
        chunker=TextChunker(
            chunk_size=settings.rag_chunk_size,
            overlap=settings.rag_chunk_overlap,
        ),
        embedding_model=embedder,
        vector_store=store,
        embedding_batch_size=settings.rag_embedding_batch_size,
        insert_batch_size=settings.rag_insert_batch_size,
    )
    try:
        result = await service.ingest(path)
    finally:
        if owns_store:
            await store.close()
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


async def run_rag_demo(
    settings: Settings,
    question: str,
    *,
    embedding_model: EmbeddingModel | None = None,
    vector_store: VectorStore | None = None,
    reranker: Reranker | None = None,
    chat_model: ChatModel | None = None,
) -> None:
    """Run the complete Phase 6 retrieval and grounded answer pipeline."""

    embedder = embedding_model or create_dashscope_embedding_model(settings)
    store = vector_store or create_milvus_vector_store(settings)
    ranker = reranker or create_dashscope_reranker(settings)
    model = chat_model or create_dashscope_chat_model(settings)
    owns_store = vector_store is None
    owns_reranker = reranker is None
    service = RagService(
        retriever=VectorRetriever(
            embedding_model=embedder,
            vector_store=store,
            default_top_k=settings.rag_initial_top_k,
        ),
        reranker=ranker,
        chat_model=model,
        initial_top_k=settings.rag_initial_top_k,
        final_top_k=settings.rag_final_top_k,
        max_context_chars=settings.rag_max_context_chars,
        max_answer_tokens=settings.rag_max_answer_tokens,
    )
    try:
        result = await service.answer(question)
    finally:
        if owns_reranker:
            await ranker.close()
        if owns_store:
            await store.close()
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


async def run_tool_demo(
    settings: Settings,
    name: str,
    arguments_json: str,
    *,
    registry: ToolRegistry | None = None,
) -> None:
    """Invoke one Phase 7 tool through the shared registry."""

    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        raise ValueError("tool arguments must be valid JSON") from exc
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be a JSON object")

    tool_registry = registry or create_tool_registry(settings)
    try:
        result = await tool_registry.invoke(name, arguments)
    finally:
        if registry is None:
            await tool_registry.close()
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


def list_tools(
    settings: Settings,
    *,
    registry: ToolRegistry | None = None,
) -> None:
    """Print registered tool contracts without invoking external providers."""

    tool_registry = registry or create_tool_registry(settings)
    print(
        json.dumps(
            [asdict(spec) for spec in tool_registry.specs()],
            ensure_ascii=False,
            indent=2,
        )
    )
    if registry is None:
        asyncio.run(tool_registry.close())


async def run_agent_demo(
    settings: Settings,
    questions: Sequence[str],
    *,
    model: ToolCallingModel | None = None,
    registry: ToolRegistry | None = None,
) -> None:
    """Run Phase 8 ReAct and print its auditable execution trace."""

    agent_model = model or create_dashscope_chat_model(settings)
    tool_registry = registry or create_tool_registry(settings)
    agent = ReActAgent(
        model=agent_model,
        tools=tool_registry,
        memory=ConversationMemory(
            max_turns=settings.agent_memory_turns,
            max_chars=settings.agent_memory_max_chars,
        ),
        max_tool_calls=settings.agent_max_tool_calls,
        max_observation_chars=settings.agent_max_observation_chars,
        max_answer_tokens=settings.agent_max_answer_tokens,
    )
    try:
        for question in questions:
            result = await agent.answer(question)
            for trace in result.traces:
                print(f"Thought: {trace.thought}")
                print(
                    "Action: "
                    + json.dumps(
                        {
                            "tool": trace.action.name,
                            "arguments": trace.action.arguments,
                        },
                        ensure_ascii=False,
                    )
                )
                observation = trace.observation
                if isinstance(observation, ToolResult):
                    observation_value: object = {
                        "summary": observation.content,
                        "data": observation.data,
                    }
                else:
                    observation_value = observation
                print(
                    "Observation: " + json.dumps(observation_value, ensure_ascii=False)
                )
            print(f"Final Answer: {result.answer}")
    finally:
        if registry is None:
            await tool_registry.close()


async def run_workflow_demo(
    settings: Settings,
    mode: str,
    question: str,
    *,
    chat_model: ChatModel | None = None,
    embedding_model: EmbeddingModel | None = None,
    vector_store: VectorStore | None = None,
    registry: ToolRegistry | None = None,
) -> None:
    """Run Phase 9 LangGraph workflow and print every auditable node step."""

    model = chat_model or create_dashscope_chat_model(settings)
    embedder = embedding_model or create_dashscope_embedding_model(settings)
    store = vector_store or create_milvus_vector_store(settings)
    tool_registry = registry or create_tool_registry(settings)
    owns_store = vector_store is None
    owns_registry = registry is None
    workflow_cls = (
        AdvancedLangGraphWorkflow if mode == "advanced" else BasicLangGraphWorkflow
    )
    workflow = workflow_cls(
        chat_model=model,
        retriever=VectorRetriever(
            embedding_model=embedder,
            vector_store=store,
            default_top_k=settings.workflow_retrieval_top_k,
        ),
        tools=tool_registry,
        retrieval_top_k=settings.workflow_retrieval_top_k,
        max_answer_tokens=settings.workflow_max_answer_tokens,
    )
    try:
        result = await workflow.run(question)
    finally:
        if owns_registry:
            await tool_registry.close()
        if owns_store:
            await store.close()

    print(f"Checkpoint: {result.checkpoint_thread_id}")
    for step in result.steps:
        print(f"Node: {step.node}")
        print(f"Summary: {step.summary}")
    print(f"Final Answer: {result.answer}")


def run_api_server(
    settings: Settings,
    *,
    host: str | None = None,
    port: int | None = None,
    reload: bool = False,
) -> None:
    """Start the Phase 10 FastAPI application with Swagger enabled."""

    import uvicorn

    uvicorn.run(
        "enterprise_ai_assistant.api.app:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Load configuration and run the requested local demo."""

    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(
        settings.log_level,
        json_logs=settings.app_env in {"staging", "production"},
    )
    if args.command == "chat":
        asyncio.run(
            run_chat_demo(
                settings,
                args.prompt,
                stream=args.stream,
            )
        )
        return
    if args.command == "prompt":
        asyncio.run(
            run_prompt_demo(
                settings,
                args.question,
                debug=args.debug,
            )
        )
        return
    if args.command == "embed":
        asyncio.run(
            run_embedding_demo(
                settings,
                args.texts,
            )
        )
        return
    if args.command == "vector-demo":
        asyncio.run(
            run_vectorstore_demo(
                settings,
                args.query,
            )
        )
        return
    if args.command == "ingest":
        asyncio.run(
            run_ingestion_demo(
                settings,
                args.path,
            )
        )
        return
    if args.command == "rag":
        asyncio.run(
            run_rag_demo(
                settings,
                args.question,
            )
        )
        return
    if args.command == "tools":
        list_tools(settings)
        return
    if args.command == "tool":
        asyncio.run(
            run_tool_demo(
                settings,
                args.name,
                args.arguments,
            )
        )
        return
    if args.command == "agent":
        asyncio.run(
            run_agent_demo(
                settings,
                args.questions,
            )
        )
        return
    if args.command == "workflow":
        asyncio.run(run_workflow_demo(settings, args.mode, args.question))
        return
    if args.command == "serve":
        run_api_server(
            settings,
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return

    result = run_demo(settings)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
