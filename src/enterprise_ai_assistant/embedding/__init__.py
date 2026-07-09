"""Embedding ports and provider adapters."""

from enterprise_ai_assistant.embedding.dashscope import (
    DashScopeEmbeddingModel,
    create_dashscope_embedding_model,
)
from enterprise_ai_assistant.embedding.exceptions import (
    EmbeddingConfigurationError,
    EmbeddingError,
    EmbeddingProviderError,
)
from enterprise_ai_assistant.embedding.protocols import EmbeddingModel

__all__ = [
    "DashScopeEmbeddingModel",
    "EmbeddingConfigurationError",
    "EmbeddingError",
    "EmbeddingModel",
    "EmbeddingProviderError",
    "create_dashscope_embedding_model",
]
