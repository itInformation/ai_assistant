"""Provider-independent models used by chat language models."""

from dataclasses import dataclass
from typing import Literal

MessageRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A single text message in a model conversation."""

    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        """Reject empty messages before they reach an external provider."""

        if not self.content.strip():
            raise ValueError("message content must not be empty")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts reported by a model provider."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Complete provider-independent chat response."""

    content: str
    model: str
    finish_reason: str | None = None
    request_id: str | None = None
    usage: TokenUsage | None = None


@dataclass(frozen=True, slots=True)
class ChatChunk:
    """Incremental provider-independent streaming response."""

    content: str
    model: str
    finish_reason: str | None = None
    request_id: str | None = None
    usage: TokenUsage | None = None
