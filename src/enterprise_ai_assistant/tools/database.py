"""Read-only, parameterized SQLite query tool."""

import asyncio
import re
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, field_validator

from enterprise_ai_assistant.models import JSONValue, ToolResult, ToolSpec
from enterprise_ai_assistant.tools._validation import validate_arguments
from enterprise_ai_assistant.tools.exceptions import (
    ToolConfigurationError,
    ToolInputError,
    ToolProviderError,
)

DatabaseScalar = str | int | float | bool | None
_READ_QUERY_PATTERN = re.compile(r"^\s*(?:SELECT|WITH)\b", re.IGNORECASE)
_FORBIDDEN_PATTERN = re.compile(
    r"\b(?:ATTACH|DETACH|PRAGMA|INSERT|UPDATE|DELETE|REPLACE|CREATE|ALTER|DROP|"
    r"VACUUM|REINDEX|ANALYZE|LOAD_EXTENSION)\b",
    re.IGNORECASE,
)


class DatabaseInput(BaseModel):
    """Validated arguments for a parameterized read-only query."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4_000)
    parameters: list[DatabaseScalar] | dict[str, DatabaseScalar] = Field(
        default_factory=list
    )

    @field_validator("parameters")
    @classmethod
    def validate_scalar_parameters(
        cls,
        value: list[DatabaseScalar] | dict[str, DatabaseScalar],
    ) -> list[DatabaseScalar] | dict[str, DatabaseScalar]:
        """SQLite bind parameters must be scalar, never nested structures."""

        values = value.values() if isinstance(value, dict) else value
        if any(isinstance(item, (list, dict)) for item in values):
            raise ValueError("database parameters must contain scalar values")
        return value


class SQLiteDatabaseTool:
    """Execute a single bounded query against a configured SQLite file."""

    spec = ToolSpec(
        name="database",
        description="对已配置 SQLite 数据库执行参数化只读 SELECT 查询。",
        parameters=DatabaseInput.model_json_schema(),
    )

    def __init__(
        self,
        *,
        database_path: Path,
        max_rows: int = 100,
        timeout_seconds: float = 5.0,
    ) -> None:
        if max_rows <= 0 or timeout_seconds <= 0:
            raise ValueError("row limit and timeout must be positive")
        self._path = database_path.expanduser().absolute()
        self._max_rows = max_rows
        self._timeout_seconds = timeout_seconds

    async def invoke(
        self,
        arguments: Mapping[str, JSONValue],
    ) -> ToolResult:
        """Validate and execute one read-only query in a worker thread."""

        request = validate_arguments(DatabaseInput, arguments)
        self._validate_query(request.query)
        if not self._path.exists():
            raise ToolConfigurationError("configured SQLite database does not exist")
        if self._path.is_symlink() or not self._path.is_file():
            raise ToolConfigurationError(
                "configured SQLite database must be a regular non-symlink file"
            )
        try:
            columns, rows, truncated = await asyncio.to_thread(
                self._execute,
                request.query,
                request.parameters,
            )
        except sqlite3.Error as exc:
            raise ToolProviderError(
                f"database query failed: {type(exc).__name__}"
            ) from exc
        suffix = "(结果已截断)" if truncated else ""
        return ToolResult(
            tool_name=self.spec.name,
            content=f"查询返回 {len(rows)} 行{suffix}",
            data={"columns": columns, "rows": rows, "truncated": truncated},
            metadata={"database": self._path.name, "max_rows": self._max_rows},
        )

    async def close(self) -> None:
        """No persistent database connection is retained."""

    @staticmethod
    def _validate_query(query: str) -> None:
        stripped = query.strip()
        if (
            not _READ_QUERY_PATTERN.match(stripped)
            or _FORBIDDEN_PATTERN.search(stripped)
            or SQLiteDatabaseTool._has_multiple_statements(stripped)
        ):
            raise ToolInputError(
                "database tool accepts one read-only SELECT or WITH query"
            )

    @staticmethod
    def _has_multiple_statements(query: str) -> bool:
        # sqlite3.complete_statement handles quoted semicolons correctly.
        if ";" not in query:
            return False
        head, separator, tail = query.partition(";")
        return bool(separator and tail.strip()) or not sqlite3.complete_statement(
            f"{head};"
        )

    def _execute(
        self,
        query: str,
        parameters: Sequence[DatabaseScalar] | Mapping[str, DatabaseScalar],
    ) -> tuple[list[JSONValue], list[JSONValue], bool]:
        uri = f"file:{quote(str(self._path), safe='/')}?mode=ro"
        with closing(
            sqlite3.connect(
                uri,
                uri=True,
                timeout=self._timeout_seconds,
            )
        ) as connection:
            # URI read-only mode is the primary boundary; query_only adds
            # defense in depth if the connection configuration changes later.
            connection.execute("PRAGMA query_only = ON")
            cursor = connection.execute(query, parameters)
            raw_rows = cursor.fetchmany(self._max_rows + 1)
            columns = [description[0] for description in cursor.description or ()]
        truncated = len(raw_rows) > self._max_rows
        rows = [
            [self._json_value(value) for value in row]
            for row in raw_rows[: self._max_rows]
        ]
        return columns, rows, truncated

    @staticmethod
    def _json_value(value: Any) -> JSONValue:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, bytes):
            return f"<BLOB {len(value)} bytes>"
        return str(value)
