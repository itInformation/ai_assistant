"""Stable exceptions raised by the prompt module."""


class PromptError(Exception):
    """Base exception for prompt engineering failures."""


class PromptRenderError(PromptError):
    """Raised when template variables cannot be rendered safely."""


class PromptVersionError(PromptError):
    """Raised when prompt version registration or lookup fails."""


class OutputParserError(PromptError):
    """Raised when model output does not match the required schema."""
