"""Application bootstrap and health information."""

from dataclasses import asdict, dataclass

from enterprise_ai_assistant import __version__
from enterprise_ai_assistant.config import Settings
from enterprise_ai_assistant.core import get_logger


@dataclass(frozen=True, slots=True)
class ApplicationInfo:
    """Serializable application health information."""

    name: str
    environment: str
    status: str
    version: str


def build_application_info(settings: Settings) -> ApplicationInfo:
    """Build health information without contacting external services."""

    return ApplicationInfo(
        name=settings.app_name,
        environment=settings.app_env,
        status="ready",
        version=__version__,
    )


def run_demo(settings: Settings) -> dict[str, str]:
    """Run the Phase 1 smoke demo and return its structured result."""

    info = build_application_info(settings)
    get_logger(component="bootstrap").info(
        "application_ready",
        **asdict(info),
    )
    return asdict(info)
