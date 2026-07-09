"""Tests for environment-backed application settings."""

from enterprise_ai_assistant.config import Settings


def test_settings_have_safe_defaults(monkeypatch: object) -> None:
    """Defaults should allow local startup without secrets."""

    # PyMilvus may load a project .env during import, so explicitly isolate
    # this default-value test from process-level provider credentials.
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)  # type: ignore[attr-defined]
    settings = Settings(_env_file=None)

    assert settings.app_env == "development"
    assert settings.dashscope_api_key is None
    assert settings.dashscope_embedding_dimensions == 1024
    assert settings.dashscope_timeout_seconds == 30
    assert settings.dashscope_max_retries == 2
    assert settings.milvus_uri == "milvus/knowledge.db"
    assert settings.milvus_collection_name == "knowledge_chunks"


def test_settings_read_environment(monkeypatch: object) -> None:
    """Environment values should override defaults with type conversion."""

    monkeypatch.setenv("APP_ENV", "testing")  # type: ignore[attr-defined]
    monkeypatch.setenv("DEBUG", "true")  # type: ignore[attr-defined]

    settings = Settings(_env_file=None)

    assert settings.app_env == "testing"
    assert settings.debug is True


def test_settings_support_legacy_dashscope_model_name(
    monkeypatch: object,
) -> None:
    """Existing DASHSCOPE_MODEL files should configure the chat adapter."""

    monkeypatch.setenv("DASHSCOPE_MODEL", "qwen-turbo")  # type: ignore[attr-defined]

    settings = Settings(_env_file=None)

    assert settings.dashscope_chat_model == "qwen-turbo"
