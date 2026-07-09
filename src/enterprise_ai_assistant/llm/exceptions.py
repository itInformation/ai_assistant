"""Stable application exceptions for language-model failures."""


class LLMError(Exception):
    """Base exception for language-model operations."""


class LLMConfigurationError(LLMError):
    """Raised when a language-model adapter is not configured safely."""


class LLMProviderError(LLMError):
    """Raised when an external model provider cannot complete a request."""
