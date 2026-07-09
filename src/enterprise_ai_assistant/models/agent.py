"""Provider-independent models for tool-calling agents."""

from dataclasses import dataclass
from typing import Literal

from enterprise_ai_assistant.models.tool import ToolResult
from enterprise_ai_assistant.models.vectorstore import JSONValue

AgentRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class AgentToolCall:
    """One validated tool request selected by the model."""

    id: str
    name: str
    arguments: dict[str, JSONValue]

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.name.strip():
            raise ValueError("tool call id and name must not be empty")


@dataclass(frozen=True, slots=True)
class AgentMessage:
    """A message that can carry assistant tool calls or tool observations."""

    role: AgentRole
    content: str = ""
    tool_calls: tuple[AgentToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if self.role in {"system", "user"} and not self.content.strip():
            raise ValueError("system and user agent messages must not be empty")
        if self.role == "assistant" and not (self.content.strip() or self.tool_calls):
            raise ValueError("assistant message requires content or tool calls")
        if self.role == "tool" and (
            not self.content.strip() or not (self.tool_call_id or "").strip()
        ):
            raise ValueError("tool message requires content and tool_call_id")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("only assistant messages can contain tool calls")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("only tool messages can contain tool_call_id")


@dataclass(frozen=True, slots=True)
class AgentModelResponse:
    """One complete tool-capable model response."""

    message: AgentMessage
    model: str
    finish_reason: str | None = None
    request_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class AgentTrace:
    """A safe, user-visible ReAct step without private chain-of-thought."""

    thought: str
    action: AgentToolCall
    observation: ToolResult | str


@dataclass(frozen=True, slots=True)
class AgentAnswer:
    """Final agent answer and its auditable execution trace."""

    answer: str
    traces: tuple[AgentTrace, ...]
    model: str
    total_prompt_tokens: int
    total_completion_tokens: int
