"""Tests for provider-independent tool models."""

import pytest

from enterprise_ai_assistant.models import ToolResult, ToolSpec


def test_tool_models_validate_stable_contract() -> None:
    """Names and schemas must be safe to expose to an LLM."""

    spec = ToolSpec(
        name="company_search",
        description="Search company records.",
        parameters={"type": "object", "properties": {}},
    )
    result = ToolResult(
        tool_name=spec.name,
        content="one result",
        data={"count": 1},
    )

    assert result.tool_name == "company_search"


@pytest.mark.parametrize(
    ("name", "parameters"),
    [
        ("", {"type": "object"}),
        ("invalid-name", {"type": "object"}),
        ("valid", {"type": "array"}),
    ],
)
def test_tool_spec_rejects_invalid_llm_contract(
    name: str,
    parameters: dict[str, str],
) -> None:
    """Malformed metadata should fail before registration."""

    with pytest.raises(ValueError):
        ToolSpec(name=name, description="description", parameters=parameters)
