"""Tests for the local command-line entry point."""

import json

from enterprise_ai_assistant import cli
from enterprise_ai_assistant.config import Settings


def test_main_prints_structured_application_info(
    monkeypatch: object,
    capsys: object,
) -> None:
    """CLI should initialize the app and finish with a JSON result."""

    settings = Settings(app_name="Knowledge Assistant", app_env="testing")
    monkeypatch.setattr(cli, "get_settings", lambda: settings)  # type: ignore[attr-defined]

    cli.main()

    output_lines = capsys.readouterr().out.strip().splitlines()  # type: ignore[attr-defined]
    result = json.loads(output_lines[-1])
    assert result == {
        "name": "Knowledge Assistant",
        "environment": "testing",
        "status": "ready",
        "version": "0.1.0",
    }
