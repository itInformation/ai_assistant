"""Explicit allow-list registry for application tools."""

import asyncio
from collections.abc import Iterable, Mapping

from enterprise_ai_assistant.models import JSONValue, ToolResult, ToolSpec
from enterprise_ai_assistant.tools.exceptions import ToolNotFoundError
from enterprise_ai_assistant.tools.protocols import Tool


class ToolRegistry:
    """Register tools by stable name and dispatch validated invocations."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register one tool without silently replacing an existing tool."""

        name = tool.spec.name
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        """Resolve an allowed tool or raise a stable lookup error."""

        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"unknown tool: {name}") from exc

    def specs(self) -> tuple[ToolSpec, ...]:
        """Return deterministic tool metadata for CLI or future LLM binding."""

        return tuple(self._tools[name].spec for name in sorted(self._tools))

    async def invoke(
        self,
        name: str,
        arguments: Mapping[str, JSONValue],
    ) -> ToolResult:
        """Dispatch an invocation only to a registered tool."""

        return await self.get(name).invoke(arguments)

    async def close(self) -> None:
        """Close every registered tool, even if one close operation fails."""

        results = await asyncio.gather(
            *(tool.close() for tool in self._tools.values()),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
