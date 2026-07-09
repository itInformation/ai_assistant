"""In-memory registry for immutable prompt template versions."""

from enterprise_ai_assistant.prompt.exceptions import PromptVersionError
from enterprise_ai_assistant.prompt.template import PromptTemplate


class PromptRegistry:
    """Register and resolve immutable prompt versions by name."""

    def __init__(self) -> None:
        """Create an empty prompt registry."""

        self._templates: dict[tuple[str, int], PromptTemplate] = {}

    def register(self, template: PromptTemplate) -> None:
        """Register one version and reject accidental replacement."""

        key = (template.name, template.version)
        if key in self._templates:
            raise PromptVersionError(
                f"prompt {template.name!r} version {template.version} already exists"
            )
        self._templates[key] = template

    def get(self, name: str, version: int | None = None) -> PromptTemplate:
        """Resolve an exact version or the latest registered version."""

        if version is not None:
            try:
                return self._templates[(name, version)]
            except KeyError as exc:
                raise PromptVersionError(
                    f"prompt {name!r} version {version} was not found"
                ) from exc

        candidates = [
            template
            for (template_name, _), template in self._templates.items()
            if template_name == name
        ]
        if not candidates:
            raise PromptVersionError(f"prompt {name!r} was not found")
        return max(candidates, key=lambda template: template.version)

    def versions(self, name: str) -> tuple[int, ...]:
        """List all registered versions for a prompt in ascending order."""

        return tuple(
            sorted(
                version
                for template_name, version in self._templates
                if template_name == name
            )
        )
