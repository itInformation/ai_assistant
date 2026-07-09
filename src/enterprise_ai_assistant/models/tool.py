"""Provider-independent models used by callable tools."""

from dataclasses import dataclass, field
from typing import Any

from enterprise_ai_assistant.models.vectorstore import JSONValue


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Describe a tool in a format that can later be exposed to an LLM."""

    name: str
    description: str
    parameters: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("tool name must be a non-empty identifier")
        if not self.description.strip():
            raise ValueError("tool description must not be empty")
        if self.parameters.get("type") != "object":
            raise ValueError("tool parameters must be an object JSON Schema")


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Return structured, bounded data from a tool invocation."""

    tool_name: str
    content: str
    data: JSONValue
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tool_name.strip():
            raise ValueError("tool_name must not be empty")
        if not self.content.strip():
            raise ValueError("tool content must not be empty")
