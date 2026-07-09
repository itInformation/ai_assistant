"""Bounded in-process conversation memory."""

import asyncio
from collections import deque

from enterprise_ai_assistant.models import AgentMessage


class ConversationMemory:
    """Keep recent completed turns without persisting tool observations."""

    def __init__(
        self,
        *,
        max_turns: int = 6,
        max_chars: int = 12_000,
    ) -> None:
        if max_turns <= 0 or max_chars <= 0:
            raise ValueError("memory limits must be positive")
        self._max_turns = max_turns
        self._max_chars = max_chars
        self._turns: deque[tuple[AgentMessage, AgentMessage]] = deque()
        self._lock = asyncio.Lock()

    async def snapshot(self) -> tuple[AgentMessage, ...]:
        """Return an immutable, ordered copy of current conversation history."""

        async with self._lock:
            return tuple(message for turn in self._turns for message in turn)

    async def remember(self, user: str, assistant: str) -> None:
        """Append one completed turn and evict oldest context when bounded."""

        turn = (
            AgentMessage(role="user", content=user),
            AgentMessage(role="assistant", content=assistant),
        )
        async with self._lock:
            self._turns.append(turn)
            while len(self._turns) > self._max_turns or self._char_count() > (
                self._max_chars
            ):
                self._turns.popleft()

    async def clear(self) -> None:
        """Forget all conversation state."""

        async with self._lock:
            self._turns.clear()

    def _char_count(self) -> int:
        return sum(len(message.content) for turn in self._turns for message in turn)
