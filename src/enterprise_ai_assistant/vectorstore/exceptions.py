"""Stable exceptions for vector-store operations."""


class VectorStoreError(Exception):
    """Base exception for vector-store operations."""


class VectorStoreConfigurationError(VectorStoreError):
    """Raised when a vector store is configured incorrectly."""


class VectorStoreProviderError(VectorStoreError):
    """Raised when a vector database operation fails."""
