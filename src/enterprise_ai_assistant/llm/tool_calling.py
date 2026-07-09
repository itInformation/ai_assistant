"""Language-model port for native function calling."""

from collections.abc import Sequence
from typing import Protocol

from enterprise_ai_assistant.models import (
    AgentMessage,
    AgentModelResponse,
    ToolSpec,
)


class ToolCallingModel(Protocol):
    """Select tools or produce a final answer from an Agent conversation."""

    async def chat_with_tools(
        self,
        messages: Sequence[AgentMessage],
        tools: Sequence[ToolSpec],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AgentModelResponse:
        """Return either tool calls or a natural-language final answer."""
