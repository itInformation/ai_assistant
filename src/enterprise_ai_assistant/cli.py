"""Command-line entry point for local development."""

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict

from enterprise_ai_assistant.app import run_demo
from enterprise_ai_assistant.config import Settings, get_settings
from enterprise_ai_assistant.core import configure_logging
from enterprise_ai_assistant.llm import ChatModel, create_dashscope_chat_model
from enterprise_ai_assistant.models import ChatMessage
from enterprise_ai_assistant.prompt import (
    PromptRegistry,
    PromptService,
    StructuredOutputParser,
)
from enterprise_ai_assistant.prompt.examples import (
    SupportTicketClassification,
    build_support_classification_prompt,
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


def main(argv: Sequence[str] | None = None) -> None:
    """Load configuration and run a health or chat demo."""

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

    result = run_demo(settings)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
