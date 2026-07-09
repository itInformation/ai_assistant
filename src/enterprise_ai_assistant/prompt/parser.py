"""Strongly typed parsers for language-model output."""

import json
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from enterprise_ai_assistant.prompt.exceptions import OutputParserError

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class StructuredOutputParser(Generic[OutputModelT]):
    """Validate JSON model output against a Pydantic schema."""

    def __init__(self, model_type: type[OutputModelT]) -> None:
        """Create a parser for one explicit business output type."""

        self._model_type = model_type

    @property
    def model_type(self) -> type[OutputModelT]:
        """Return the Pydantic model validated by this parser."""

        return self._model_type

    def format_instructions(self) -> str:
        """Build deterministic JSON instructions from the business schema."""

        schema = json.dumps(
            self._model_type.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return (
            "请只输出合法 JSON, 不要包含 Markdown 代码块或额外解释。"
            f"输出必须严格符合以下 JSON Schema: {schema}"
        )

    def parse(self, content: str) -> OutputModelT:
        """Parse and validate provider output without leaking raw content."""

        try:
            return self._model_type.model_validate_json(content)
        except (ValidationError, ValueError) as exc:
            # Raw model output may contain business or personal information,
            # so the stable exception intentionally excludes response content.
            raise OutputParserError(
                f"model output failed {self._model_type.__name__} validation"
            ) from exc
