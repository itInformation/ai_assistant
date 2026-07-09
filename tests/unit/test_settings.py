"""Tests for environment-backed application settings."""

from pathlib import Path

from enterprise_ai_assistant.config import Settings


def test_settings_have_safe_defaults() -> None:
    """Defaults should allow local startup without secrets."""

    settings = Settings(_env_file=None)

    assert settings.app_env == "development"
    assert settings.dashscope_api_key is None
    assert settings.dashscope_timeout_seconds == 30
    assert settings.dashscope_max_retries == 2
    assert settings.milvus_uri == Path("milvus/knowledge.db")


def test_settings_read_environment(monkeypatch: object) -> None:
    """Environment values should override defaults with type conversion."""

    monkeypatch.setenv("APP_ENV", "testing")  # type: ignore[attr-defined]
    monkeypatch.setenv("DEBUG", "true")  # type: ignore[attr-defined]

    settings = Settings(_env_file=None)

    assert settings.app_env == "testing"
    assert settings.debug is True
