"""Structured observability events with bounded prompt and response previews."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

from enterprise_ai_assistant.core import get_logger
from enterprise_ai_assistant.models import JSONValue, TokenUsage


@dataclass(frozen=True, slots=True)
class ObservabilityEvent:
    """One sanitized event emitted for an API operation."""

    request_id: str
    operation: str
    latency_ms: float
    prompt_preview: str | None = None
    response_preview: str | None = None
    token_usage: TokenUsage | None = None
    tool_trace: tuple[dict[str, JSONValue], ...] = ()
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)


class ObservabilityRecorder(Protocol):
    """Port for writing trace events to logs or an external backend."""

    def record(self, event: ObservabilityEvent) -> None:
        """Persist or emit one sanitized observability event."""


class InMemoryObservabilityRecorder:
    """Record events in memory and emit matching structlog records."""

    def __init__(self) -> None:
        self._events: list[ObservabilityEvent] = []
        self._logger = get_logger(component="observability")

    @property
    def events(self) -> tuple[ObservabilityEvent, ...]:
        """Return events captured in this process for tests or local debugging."""

        return tuple(self._events)

    def record(self, event: ObservabilityEvent) -> None:
        """Store one event and log it with a stable event name."""

        self._events.append(event)
        payload = asdict(event)
        if event.error_type:
            self._logger.error("api_operation_failed", **payload)
            return
        self._logger.info("api_operation_completed", **payload)


def preview_text(value: str, *, max_chars: int) -> str:
    """Return a bounded single-field preview suitable for logs."""

    text = value.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"
