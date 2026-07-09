"""Tests for prompt rendering and model orchestration."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

from pydantic import BaseModel

from enterprise_ai_assistant.models import (
    ChatChunk,
    ChatMessage,
    ChatResponse,
    ResponseFormat,
)
from enterprise_ai_assistant.prompt import (
    PromptService,
    PromptTemplate,
    StructuredOutputParser,
)


class Classification(BaseModel):
    """Structured response used by the service test."""

    category: str


class RecordingChatModel:
    """Chat model that records the provider-independent request."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.messages: Sequence[ChatMessage] = ()
        self.options: dict[str, Any] = {}

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: ResponseFormat = "text",
    ) -> ChatResponse:
        """Record the call and return configured content."""

        self.messages = messages
        self.options = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
        }
        return ChatResponse(content=self.content, model="fake")

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: ResponseFormat = "text",
    ) -> AsyncIterator[ChatChunk]:
        """Satisfy the ChatModel protocol for this focused test fake."""

        if False:
            yield ChatChunk(content="", model="fake")


def test_run_structured_enables_json_mode_and_validates_result() -> None:
    """Structured execution should combine schema instructions and JSON Mode."""

    model = RecordingChatModel('{"category":"售后"}')
    service = PromptService(model)
    template = PromptTemplate(
        name="classify",
        version=1,
        system_template="你是分类助手。",
        user_template="{question}",
    )

    result = asyncio.run(
        service.run_structured(
            template,
            {"question": "退款进度"},
            StructuredOutputParser(Classification),
        )
    )

    assert result.value.category == "售后"
    assert result.response.model == "fake"
    assert model.options == {
        "temperature": 0,
        "max_tokens": None,
        "response_format": "json_object",
    }
    assert model.messages[0].role == "system"
    assert "JSON Schema" in model.messages[0].content


def test_run_supports_free_text_prompt() -> None:
    """Free-text execution should preserve normal chat response behavior."""

    model = RecordingChatModel("普通回答")
    service = PromptService(model)
    template = PromptTemplate(
        name="answer",
        version=1,
        user_template="{question}",
    )

    response = asyncio.run(
        service.run(
            template,
            {"question": "什么是 RAG?"},
            temperature=0.3,
            max_tokens=200,
        )
    )

    assert response.content == "普通回答"
    assert model.options["response_format"] == "text"
