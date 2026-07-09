"""Tests for provider-independent language-model models."""

import pytest

from enterprise_ai_assistant.models import ChatMessage


def test_chat_message_rejects_blank_content() -> None:
    """Blank messages should fail before an external API call."""

    with pytest.raises(ValueError, match="must not be empty"):
        ChatMessage(role="user", content="  ")
