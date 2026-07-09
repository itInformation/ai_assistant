"""Provider-independent language-model interfaces."""

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from enterprise_ai_assistant.models import ChatChunk, ChatMessage, ChatResponse


@runtime_checkable
class ChatModel(Protocol):
    """Port implemented by any conversational language-model provider."""

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Generate one complete response for a conversation."""

        ...

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatChunk]:
        """Generate incremental response chunks for a conversation."""

        ...
