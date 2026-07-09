"""Ports implemented by all application tools."""

from collections.abc import Mapping
from typing import Protocol

from enterprise_ai_assistant.models import JSONValue, ToolResult, ToolSpec


class Tool(Protocol):
    """Execute one bounded capability through a common contract."""

    @property
    def spec(self) -> ToolSpec:
        """Return the immutable name, description, and input schema."""

    async def invoke(self, arguments: Mapping[str, JSONValue]) -> ToolResult:
        """Validate arguments and execute the tool."""

    async def close(self) -> None:
        """Release resources owned by the tool."""
