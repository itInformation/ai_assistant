"""Tests for immutable prompt version registration."""

import pytest

from enterprise_ai_assistant.prompt import (
    PromptRegistry,
    PromptTemplate,
    PromptVersionError,
)


def template(version: int) -> PromptTemplate:
    """Build one version of a stable prompt name."""

    return PromptTemplate(
        name="answer",
        version=version,
        user_template=f"v{version}: {{question}}",
    )


def test_registry_resolves_exact_and_latest_versions() -> None:
    """Callers should be able to pin or follow the latest prompt."""

    registry = PromptRegistry()
    registry.register(template(2))
    registry.register(template(1))

    assert registry.get("answer", 1).version == 1
    assert registry.get("answer").version == 2
    assert registry.versions("answer") == (1, 2)


def test_registry_rejects_duplicate_version() -> None:
    """Registered prompt versions should remain immutable."""

    registry = PromptRegistry()
    registry.register(template(1))

    with pytest.raises(PromptVersionError, match="already exists"):
        registry.register(template(1))


def test_registry_reports_missing_prompt() -> None:
    """Unknown prompt lookups should produce a stable domain exception."""

    registry = PromptRegistry()

    with pytest.raises(PromptVersionError, match="was not found"):
        registry.get("missing")
