"""Command-line entry point for local development."""

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict
from math import sqrt
from uuid import uuid4

from enterprise_ai_assistant.app import run_demo
from enterprise_ai_assistant.config import Settings, get_settings
from enterprise_ai_assistant.core import configure_logging
from enterprise_ai_assistant.embedding import (
    EmbeddingModel,
    create_dashscope_embedding_model,
)
from enterprise_ai_assistant.llm import ChatModel, create_dashscope_chat_model
from enterprise_ai_assistant.models import (
    ChatMessage,
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
from enterprise_ai_assistant.vectorstore import (
    VectorStore,
    create_milvus_vector_store,
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

    result = run_demo(settings)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
