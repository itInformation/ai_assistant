"""Tool ports, registry, and built-in adapters."""

from pathlib import Path

from enterprise_ai_assistant.config import Settings
from enterprise_ai_assistant.tools.database import SQLiteDatabaseTool
from enterprise_ai_assistant.tools.exceptions import (
    ToolConfigurationError,
    ToolError,
    ToolInputError,
    ToolNotFoundError,
    ToolProviderError,
)
from enterprise_ai_assistant.tools.protocols import Tool
from enterprise_ai_assistant.tools.registry import ToolRegistry
from enterprise_ai_assistant.tools.search import TavilySearchTool
from enterprise_ai_assistant.tools.weather import OpenMeteoWeatherTool

__all__ = [
    "OpenMeteoWeatherTool",
    "SQLiteDatabaseTool",
    "TavilySearchTool",
    "Tool",
    "ToolConfigurationError",
    "ToolError",
    "ToolInputError",
    "ToolNotFoundError",
    "ToolProviderError",
    "ToolRegistry",
    "create_tool_registry",
]


def create_tool_registry(settings: Settings) -> ToolRegistry:
    """Build the Phase 7 allow-list from application settings."""

    return ToolRegistry(
        [
            OpenMeteoWeatherTool(
                timeout_seconds=settings.tool_http_timeout_seconds,
                max_retries=settings.tool_max_retries,
            ),
            TavilySearchTool(
                api_key=settings.tavily_api_key or "",
                timeout_seconds=settings.tool_http_timeout_seconds,
                max_retries=settings.tool_max_retries,
                max_content_chars=settings.search_max_content_chars,
            ),
            SQLiteDatabaseTool(
                database_path=Path(settings.database_path),
                max_rows=settings.database_max_rows,
                timeout_seconds=settings.database_timeout_seconds,
            ),
        ]
    )
