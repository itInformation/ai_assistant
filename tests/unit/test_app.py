"""Tests for the Phase 1 application bootstrap."""

import json

from enterprise_ai_assistant.app import build_application_info, run_demo
from enterprise_ai_assistant.config import Settings
from enterprise_ai_assistant.core import configure_logging


def test_build_application_info() -> None:
    """Health information should reflect the active configuration."""

    settings = Settings(app_name="Knowledge Assistant", app_env="testing")

    result = build_application_info(settings)

    assert result.name == "Knowledge Assistant"
    assert result.environment == "testing"
    assert result.status == "ready"


def test_run_demo_returns_serializable_result() -> None:
    """The bootstrap demo should return stable structured data."""

    configure_logging()
    settings = Settings(app_env="testing")

    result = run_demo(settings)

    assert result["status"] == "ready"
    assert result["version"] == "0.1.0"


def test_run_demo_supports_json_logs(capsys: object) -> None:
    """Production-style logging should emit machine-readable events."""

    configure_logging(json_logs=True)

    run_demo(Settings(app_env="testing"))

    event = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert event["event"] == "application_ready"
    assert event["component"] == "bootstrap"
    assert event["status"] == "ready"
