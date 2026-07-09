"""Tests for the local command-line entry point."""

import asyncio
import json
from collections.abc import AsyncIterator, Sequence

from enterprise_ai_assistant import cli
from enterprise_ai_assistant.config import Settings
from enterprise_ai_assistant.models import ChatChunk, ChatMessage, ChatResponse


class FakeChatModel:
    """Deterministic chat model used by CLI tests."""

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Return a stable complete response."""

        return ChatResponse(content=f"回复: {messages[0].content}", model="fake")

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatChunk]:
        """Yield a stable response in two chunks."""

        for content in ("流式", "回复"):
            yield ChatChunk(content=content, model="fake")


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
