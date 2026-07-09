"""Type-safe application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
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
    dashscope_timeout_seconds: float = Field(default=30.0, gt=0)
    dashscope_max_retries: int = Field(default=2, ge=0, le=10)

    milvus_uri: Path = Path("milvus/knowledge.db")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""

    return Settings()
