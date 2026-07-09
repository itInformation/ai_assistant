"""Tests for the explicit tool allow-list."""

import asyncio
from collections.abc import Mapping

import pytest

from enterprise_ai_assistant.models import JSONValue, ToolResult, ToolSpec
from enterprise_ai_assistant.tools import ToolNotFoundError, ToolRegistry


class FakeTool:
    """Minimal tool used to verify registry behavior."""

    spec = ToolSpec(
        name="fake",
        description="A fake tool.",
        parameters={"type": "object", "properties": {}},
    )

    def __init__(self) -> None:
        self.closed = False

    async def invoke(
        self,
        arguments: Mapping[str, JSONValue],
    ) -> ToolResult:
        return ToolResult("fake", "done", dict(arguments))

    async def close(self) -> None:
        self.closed = True


def test_registry_dispatches_and_lists_tools() -> None:
    """Registered tools should be discoverable and callable by stable name."""

    tool = FakeTool()
    registry = ToolRegistry([tool])

    result = asyncio.run(registry.invoke("fake", {"value": 1}))
    asyncio.run(registry.close())

    assert registry.specs() == (tool.spec,)
    assert result.data == {"value": 1}
    assert tool.closed is True


def test_registry_rejects_duplicate_and_unknown_tools() -> None:
    """The allow-list must not silently replace or invent tools."""

    registry = ToolRegistry([FakeTool()])

    with pytest.raises(ValueError, match="already registered"):
        registry.register(FakeTool())
    with pytest.raises(ToolNotFoundError, match="unknown"):
        registry.get("missing")
