"""Stable exceptions for embedding operations."""


class EmbeddingError(Exception):
    """Base exception for embedding operations."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when an embedding adapter is configured incorrectly."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when an external embedding provider fails."""
