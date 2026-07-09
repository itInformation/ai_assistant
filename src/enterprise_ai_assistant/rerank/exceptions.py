"""Stable exceptions for reranking operations."""


class RerankerError(Exception):
    """Base exception for reranking operations."""


class RerankerConfigurationError(RerankerError):
    """Raised when a reranker is configured incorrectly."""


class RerankerProviderError(RerankerError):
    """Raised when an external reranking provider fails."""
