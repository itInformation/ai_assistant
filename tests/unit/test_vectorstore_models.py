"""Tests for provider-independent vector-store models."""

import pytest

from enterprise_ai_assistant.models import VectorRecord, VectorSearchFilter


def record(**changes: object) -> VectorRecord:
    """Build a valid record with optional field overrides."""

    values = {
        "id": "doc-1-0",
        "document_id": "doc-1",
        "content": "企业知识库内容",
        "embedding": (0.1, 0.2),
        "source": "manual.md",
        "chunk_index": 0,
        "metadata": {"department": "研发"},
    }
    values.update(changes)
    return VectorRecord(**values)  # type: ignore[arg-type]


def test_vector_record_accepts_json_metadata() -> None:
    """Valid chunk metadata should remain available for later filtering."""

    result = record(metadata={"tags": ["ai", "rag"], "published": True})

    assert result.metadata["tags"] == ["ai", "rag"]


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"id": ""}, "must not be empty"),
        ({"embedding": ()}, "embedding must not be empty"),
        ({"chunk_index": -1}, "must not be negative"),
        ({"metadata": {"invalid": object()}}, "JSON serializable"),
        ({"metadata": {"invalid.key": "value"}}, "metadata keys"),
    ],
)
def test_vector_record_rejects_invalid_data(
    changes: dict[str, object],
    expected: str,
) -> None:
    """Invalid records should fail before reaching Milvus."""

    with pytest.raises(ValueError, match=expected):
        record(**changes)


def test_vector_search_filter_rejects_blank_values() -> None:
    """Blank structured filters should never become Milvus expressions."""

    with pytest.raises(ValueError, match="must not contain empty"):
        VectorSearchFilter(document_ids=("",))
