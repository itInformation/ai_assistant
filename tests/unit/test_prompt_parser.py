"""Tests for structured model-output parsing."""

import pytest
from pydantic import BaseModel, Field

from enterprise_ai_assistant.prompt import OutputParserError, StructuredOutputParser


class ProductAnswer(BaseModel):
    """Business schema used to exercise structured parsing."""

    product: str
    confidence: float = Field(ge=0, le=1)


def test_parser_generates_json_schema_instructions() -> None:
    """Format instructions should explicitly request JSON and include the schema."""

    instructions = StructuredOutputParser(ProductAnswer).format_instructions()

    assert "JSON" in instructions
    assert '"confidence"' in instructions
    assert '"maximum":1' in instructions


def test_parser_returns_typed_model() -> None:
    """Valid JSON should become a strongly typed business model."""

    result = StructuredOutputParser(ProductAnswer).parse(
        '{"product":"知识库","confidence":0.9}'
    )

    assert result.product == "知识库"
    assert result.confidence == 0.9


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"product":"知识库","confidence":2}',
        '```json\n{"product":"知识库","confidence":0.9}\n```',
    ],
)
def test_parser_rejects_invalid_or_wrapped_output(content: str) -> None:
    """Invalid JSON, schema failures, and Markdown wrappers should be rejected."""

    with pytest.raises(OutputParserError, match="ProductAnswer validation"):
        StructuredOutputParser(ProductAnswer).parse(content)
