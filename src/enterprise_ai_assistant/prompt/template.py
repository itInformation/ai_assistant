"""Versioned prompt templates with system and few-shot message support."""

from dataclasses import dataclass
from string import Formatter
from typing import Any

from enterprise_ai_assistant.models import ChatMessage
from enterprise_ai_assistant.prompt.exceptions import PromptRenderError


@dataclass(frozen=True, slots=True)
class FewShotExample:
    """One static user/assistant example pair."""

    user: str
    assistant: str

    def __post_init__(self) -> None:
        """Reject examples that cannot teach the model a useful mapping."""

        if not self.user.strip() or not self.assistant.strip():
            raise ValueError("few-shot user and assistant content must not be empty")


@dataclass(frozen=True, slots=True)
class PromptDebugInfo:
    """Rendered prompt details safe to inspect before model invocation."""

    name: str
    version: int
    required_variables: tuple[str, ...]
    provided_variables: tuple[str, ...]
    message_count: int
    character_count: int
    messages: tuple[ChatMessage, ...]


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A versioned chat prompt with strict variable validation."""

    name: str
    version: int
    user_template: str
    system_template: str | None = None
    few_shot_examples: tuple[FewShotExample, ...] = ()

    def __post_init__(self) -> None:
        """Validate identity and placeholders when the template is defined."""

        if not self.name.strip():
            raise ValueError("prompt name must not be empty")
        if self.version < 1:
            raise ValueError("prompt version must be positive")
        if not self.user_template.strip():
            raise ValueError("user template must not be empty")
        self._extract_variables()

    @property
    def required_variables(self) -> tuple[str, ...]:
        """Return sorted variables required to render this prompt."""

        return tuple(sorted(self._extract_variables()))

    def render(self, **variables: Any) -> tuple[ChatMessage, ...]:
        """Render provider-independent messages with exact variable matching."""

        required = set(self.required_variables)
        provided = set(variables)
        missing = required - provided
        unexpected = provided - required
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={sorted(missing)}")
            if unexpected:
                details.append(f"unexpected={sorted(unexpected)}")
            raise PromptRenderError("invalid prompt variables: " + ", ".join(details))

        messages: list[ChatMessage] = []
        if self.system_template:
            messages.append(
                ChatMessage(
                    role="system",
                    content=self.system_template.format_map(variables),
                )
            )
        for example in self.few_shot_examples:
            messages.extend(
                (
                    ChatMessage(role="user", content=example.user),
                    ChatMessage(role="assistant", content=example.assistant),
                )
            )
        messages.append(
            ChatMessage(
                role="user",
                content=self.user_template.format_map(variables),
            )
        )
        return tuple(messages)

    def debug(self, **variables: Any) -> PromptDebugInfo:
        """Render a prompt and return deterministic inspection metadata."""

        messages = self.render(**variables)
        return PromptDebugInfo(
            name=self.name,
            version=self.version,
            required_variables=self.required_variables,
            provided_variables=tuple(sorted(variables)),
            message_count=len(messages),
            character_count=sum(len(message.content) for message in messages),
            messages=messages,
        )

    def _extract_variables(self) -> set[str]:
        templates = [self.user_template]
        if self.system_template:
            templates.append(self.system_template)

        variables: set[str] = set()
        formatter = Formatter()
        for template in templates:
            for _, field_name, _, _ in formatter.parse(template):
                if field_name is None:
                    continue
                # Restrict placeholders to plain identifiers. Attribute/index
                # access makes templates harder to audit and can expose data.
                if not field_name.isidentifier():
                    raise ValueError(f"unsupported prompt placeholder: {field_name!r}")
                variables.add(field_name)
        return variables
