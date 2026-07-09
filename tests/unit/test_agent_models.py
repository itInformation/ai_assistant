"""Tests for Agent messages and tool calls."""

import pytest

from enterprise_ai_assistant.models import AgentMessage, AgentToolCall


def test_agent_messages_support_function_call_round_trip() -> None:
    """Assistant calls and tool observations should preserve call IDs."""

    call = AgentToolCall(
        id="call-1",
        name="weather",
        arguments={"location": "北京"},
    )
    assistant = AgentMessage(role="assistant", tool_calls=(call,))
    observation = AgentMessage(
        role="tool",
        content='{"temperature":30}',
        tool_call_id="call-1",
    )

    assert assistant.tool_calls[0] == call
    assert observation.tool_call_id == call.id


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""},
        {"role": "tool", "content": "result"},
    ],
)
def test_agent_messages_reject_incomplete_protocol(
    message: dict[str, str],
) -> None:
    """Malformed histories should fail before reaching DashScope."""

    with pytest.raises(ValueError):
        AgentMessage(**message)  # type: ignore[arg-type]
