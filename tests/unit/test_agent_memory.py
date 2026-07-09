"""Tests for bounded conversation memory."""

import asyncio

from enterprise_ai_assistant.agent import ConversationMemory


def test_memory_evicts_oldest_completed_turns() -> None:
    """Turn limits should preserve only the most recent complete dialogue."""

    memory = ConversationMemory(max_turns=2, max_chars=1_000)
    asyncio.run(memory.remember("问题1", "回答1"))
    asyncio.run(memory.remember("问题2", "回答2"))
    asyncio.run(memory.remember("问题3", "回答3"))

    messages = asyncio.run(memory.snapshot())

    assert [message.content for message in messages] == [
        "问题2",
        "回答2",
        "问题3",
        "回答3",
    ]


def test_memory_can_be_cleared() -> None:
    """Explicit reset should remove all session state."""

    memory = ConversationMemory()
    asyncio.run(memory.remember("问题", "回答"))

    asyncio.run(memory.clear())

    assert asyncio.run(memory.snapshot()) == ()
