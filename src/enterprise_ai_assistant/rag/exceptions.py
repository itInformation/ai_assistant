"""Stable exceptions for document ingestion and RAG operations."""


class RagError(Exception):
    """Base exception for RAG operations."""


class DocumentLoadError(RagError):
    """Raised when a source document cannot be parsed safely."""


class UnsupportedDocumentError(DocumentLoadError):
    """Raised when no loader supports the requested file type."""


class IngestionError(RagError):
    """Raised when parsed chunks cannot be indexed."""


class RetrievalError(RagError):
    """Raised when the retrieval pipeline cannot complete."""
