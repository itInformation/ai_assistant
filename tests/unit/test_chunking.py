"""Tests for deterministic overlapping document chunking."""

import pytest

from enterprise_ai_assistant.models import DocumentSection, LoadedDocument
from enterprise_ai_assistant.rag import TextChunker


def document(content: str) -> LoadedDocument:
    """Build one representative loaded document."""

    return LoadedDocument(
        id="doc-1",
        source="manual.md",
        sections=(
            DocumentSection(
                content=content,
                metadata={"page_number": 2},
            ),
        ),
        metadata={"department": "研发"},
    )


def test_chunker_splits_near_boundaries_with_overlap() -> None:
    """Long sections should become bounded, ordered, overlapping chunks."""

    content = ("第一段知识。" * 20) + "\n\n" + ("第二段流程。" * 20)
    chunks = TextChunker(chunk_size=100, overlap=20).split(document(content))

    assert len(chunks) >= 3
    assert all(len(chunk.content) <= 100 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].id == "doc-1-000000"
    assert chunks[0].metadata["page_number"] == 2
    assert chunks[1].metadata["char_start"] < chunks[0].metadata["char_end"]


def test_chunker_preserves_short_section_as_one_chunk() -> None:
    """A short section should not be split or padded."""

    chunks = TextChunker(chunk_size=100, overlap=20).split(document("简短知识。"))

    assert len(chunks) == 1
    assert chunks[0].content == "简短知识。"


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (99, 10),
        (100, -1),
        (100, 100),
    ],
)
def test_chunker_rejects_invalid_configuration(
    chunk_size: int,
    overlap: int,
) -> None:
    """Invalid chunk windows should fail at application startup."""

    with pytest.raises(ValueError):
        TextChunker(chunk_size=chunk_size, overlap=overlap)
