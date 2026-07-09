"""Tests for parameterized read-only SQLite access."""

import asyncio
import sqlite3
from pathlib import Path

import pytest

from enterprise_ai_assistant.tools import (
    SQLiteDatabaseTool,
    ToolConfigurationError,
    ToolInputError,
)


def create_database(path: Path) -> None:
    """Create a small database fixture before opening it read-only."""

    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE employees (id INTEGER, name TEXT)")
        connection.executemany(
            "INSERT INTO employees VALUES (?, ?)",
            [(1, "张三"), (2, "李四"), (3, "王五")],
        )
        connection.commit()
    finally:
        # sqlite3's context manager commits or rolls back but does not close.
        connection.close()


def test_database_executes_parameterized_query_with_row_limit(
    tmp_path: Path,
) -> None:
    """Rows should be bounded and returned with stable column metadata."""

    path = tmp_path / "company.db"
    create_database(path)
    tool = SQLiteDatabaseTool(database_path=path, max_rows=1)

    result = asyncio.run(
        tool.invoke(
            {
                "query": "SELECT id, name FROM employees WHERE id >= ? ORDER BY id",
                "parameters": [2],
            }
        )
    )

    assert result.data["columns"] == ["id", "name"]
    assert result.data["rows"] == [[2, "李四"]]
    assert result.data["truncated"] is True


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM employees",
        "SELECT * FROM employees; DROP TABLE employees",
        "PRAGMA table_info(employees)",
        "WITH changed AS (DELETE FROM employees RETURNING *) SELECT * FROM changed",
    ],
)
def test_database_rejects_non_read_only_sql(tmp_path: Path, query: str) -> None:
    """Mutations, multiple statements, and PRAGMA access are denied."""

    path = tmp_path / "company.db"
    create_database(path)
    tool = SQLiteDatabaseTool(database_path=path)

    with pytest.raises(ToolInputError, match="read-only"):
        asyncio.run(tool.invoke({"query": query}))


def test_database_requires_regular_existing_file(tmp_path: Path) -> None:
    """The tool must not create a database from a missing path."""

    tool = SQLiteDatabaseTool(database_path=tmp_path / "missing.db")

    with pytest.raises(ToolConfigurationError, match="does not exist"):
        asyncio.run(tool.invoke({"query": "SELECT 1"}))
