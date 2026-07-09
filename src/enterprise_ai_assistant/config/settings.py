"""Type-safe application settings loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the application and external adapters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "企业级 AI 知识助手"
    app_env: Literal["development", "testing", "staging", "production"] = "development"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    dashscope_api_key: str | None = Field(default=None, repr=False)
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_chat_model: str = Field(
        default="qwen-plus",
        validation_alias=AliasChoices(
            "DASHSCOPE_CHAT_MODEL",
            "DASHSCOPE_MODEL",
        ),
    )
    dashscope_embedding_model: str = "text-embedding-v3"
    dashscope_embedding_dimensions: int = Field(default=1024, gt=0)
    dashscope_rerank_url: str = (
        "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
    )
    dashscope_rerank_model: str = "qwen3-rerank"
    dashscope_timeout_seconds: float = Field(default=30.0, gt=0)
    dashscope_max_retries: int = Field(default=2, ge=0, le=10)

    milvus_uri: str = "milvus/knowledge.db"
    milvus_token: str | None = Field(default=None, repr=False)
    milvus_collection_name: str = "knowledge_chunks"
    milvus_timeout_seconds: float = Field(default=30.0, gt=0)

    rag_chunk_size: int = Field(default=800, ge=100, le=8_000)
    rag_chunk_overlap: int = Field(default=120, ge=0)
    rag_embedding_batch_size: int = Field(default=10, ge=1, le=10)
    rag_insert_batch_size: int = Field(default=100, ge=1, le=1_000)
    rag_initial_top_k: int = Field(default=20, ge=1, le=100)
    rag_final_top_k: int = Field(default=5, ge=1, le=20)
    rag_max_context_chars: int = Field(default=8_000, ge=500)
    rag_max_answer_tokens: int = Field(default=1_200, ge=100, le=8_000)
    rag_max_file_size_mb: int = Field(default=20, ge=1, le=100)

    tavily_api_key: str | None = Field(default=None, repr=False)
    tool_http_timeout_seconds: float = Field(default=10.0, gt=0)
    tool_max_retries: int = Field(default=2, ge=0, le=10)
    search_max_content_chars: int = Field(default=1_500, ge=100, le=10_000)
    database_path: str = "data/assistant.db"
    database_max_rows: int = Field(default=100, ge=1, le=1_000)
    database_timeout_seconds: float = Field(default=5.0, gt=0)

    agent_max_tool_calls: int = Field(default=5, ge=1, le=20)
    agent_memory_turns: int = Field(default=6, ge=1, le=50)
    agent_memory_max_chars: int = Field(default=12_000, ge=1_000)
    agent_max_observation_chars: int = Field(default=6_000, ge=500)
    agent_max_answer_tokens: int = Field(default=1_200, ge=100, le=8_000)

    @model_validator(mode="after")
    def validate_rag_settings(self) -> "Settings":
        """Validate relationships between RAG tuning parameters."""

        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("RAG chunk overlap must be smaller than chunk size")
        if self.rag_final_top_k > self.rag_initial_top_k:
            raise ValueError("RAG final TopK must not exceed initial TopK")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""

    return Settings()
