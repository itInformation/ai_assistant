"""Shared input validation for type-safe tools."""

from collections.abc import Mapping
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from enterprise_ai_assistant.models import JSONValue
from enterprise_ai_assistant.tools.exceptions import ToolInputError

_InputModel = TypeVar("_InputModel", bound=BaseModel)


def validate_arguments(
    model: type[_InputModel],
    arguments: Mapping[str, JSONValue],
) -> _InputModel:
    """Validate untrusted tool arguments without leaking raw input values."""

    try:
        return model.model_validate(dict(arguments))
    except ValidationError as exc:
        fields = sorted(
            {str(error["loc"][0]) for error in exc.errors() if error.get("loc")}
        )
        detail = ", ".join(fields) if fields else "arguments"
        raise ToolInputError(f"invalid tool input fields: {detail}") from exc
