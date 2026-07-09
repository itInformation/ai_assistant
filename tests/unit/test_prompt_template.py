"""Tests for prompt templates and debug information."""

import pytest

from enterprise_ai_assistant.prompt import (
    FewShotExample,
    PromptRenderError,
    PromptTemplate,
)


def build_template() -> PromptTemplate:
    """Build a representative prompt used by template tests."""

    return PromptTemplate(
        name="answer",
        version=1,
        system_template="你是{department}助手。",
        user_template="问题: {question}",
        few_shot_examples=(FewShotExample(user="示例问题", assistant="示例回答"),),
    )


def test_render_orders_system_few_shot_and_user_messages() -> None:
    """Rendered chat messages should preserve intentional prompt ordering."""

    messages = build_template().render(department="技术", question="如何部署?")

    assert [message.role for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[0].content == "你是技术助手。"
    assert messages[-1].content == "问题: 如何部署?"


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        ({"department": "技术"}, "missing"),
        (
            {"department": "技术", "question": "问题", "extra": "值"},
            "unexpected",
        ),
    ],
)
def test_render_rejects_inexact_variables(
    variables: dict[str, str],
    expected: str,
) -> None:
    """Missing and unexpected variables should fail before model invocation."""

    with pytest.raises(PromptRenderError, match=expected):
        build_template().render(**variables)


def test_template_rejects_attribute_access_placeholders() -> None:
    """Templates should not traverse attributes or mapping indexes."""

    with pytest.raises(ValueError, match="unsupported prompt placeholder"):
        PromptTemplate(
            name="unsafe",
            version=1,
            user_template="{user.password}",
        )


def test_debug_returns_rendered_metadata() -> None:
    """Debug information should make prompt composition inspectable."""

    debug = build_template().debug(department="技术", question="如何部署?")

    assert debug.required_variables == ("department", "question")
    assert debug.message_count == 4
    assert debug.character_count == sum(
        len(message.content) for message in debug.messages
    )


def test_few_shot_example_rejects_blank_content() -> None:
    """Incomplete examples should not be accepted as model guidance."""

    with pytest.raises(ValueError, match="must not be empty"):
        FewShotExample(user="", assistant="回答")
