"""Command-line entry point for local development."""

import json

from enterprise_ai_assistant.app import run_demo
from enterprise_ai_assistant.config import get_settings
from enterprise_ai_assistant.core import configure_logging


def main() -> None:
    """Load configuration, initialize logging, and run the smoke demo."""

    settings = get_settings()
    configure_logging(
        settings.log_level,
        json_logs=settings.app_env in {"staging", "production"},
    )
    result = run_demo(settings)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
