"""Deterministic paragraph-aware overlapping text chunking."""

import re

from enterprise_ai_assistant.models import LoadedDocument, TextChunk

_BOUNDARIES = (
    "\n\n",
    "\n",
    "。",
    "！",  # noqa: RUF001 - Chinese punctuation is a real split boundary.
    "？",  # noqa: RUF001 - Chinese punctuation is a real split boundary.
    "!",
    "?",
    "；",  # noqa: RUF001 - Chinese punctuation is a real split boundary.
    ";",
)


class TextChunker:
    """Split document sections near semantic boundaries with overlap."""

    def __init__(self, *, chunk_size: int = 800, overlap: int = 120) -> None:
        """Create a character-based chunker."""

        if chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 characters")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be between zero and chunk_size")
        self._chunk_size = chunk_size
        self._overlap = overlap

    def split(self, document: LoadedDocument) -> tuple[TextChunk, ...]:
        """Split all sections while preserving document and page metadata."""

        chunks: list[TextChunk] = []
        for section_index, section in enumerate(document.sections):
            text = self._normalize(section.content)
            for content, start, end in self._split_section(text):
                chunk_index = len(chunks)
                metadata = {
                    **document.metadata,
                    **section.metadata,
                    "section_index": section_index,
                    "char_start": start,
                    "char_end": end,
                }
                chunks.append(
                    TextChunk(
                        id=f"{document.id}-{chunk_index:06d}",
                        document_id=document.id,
                        content=content,
                        source=document.source,
                        chunk_index=chunk_index,
                        metadata=metadata,
                    )
                )
        if not chunks:
            raise ValueError("document produced no non-empty chunks")
        return tuple(chunks)

    def _split_section(
        self,
        text: str,
    ) -> tuple[tuple[str, int, int], ...]:
        pieces: list[tuple[str, int, int]] = []
        start = 0
        while start < len(text):
            hard_end = min(start + self._chunk_size, len(text))
            end = self._preferred_end(text, start, hard_end)
            left = start
            while left < end and text[left].isspace():
                left += 1
            right = end
            while right > left and text[right - 1].isspace():
                right -= 1
            if left < right:
                pieces.append((text[left:right], left, right))
            if end >= len(text):
                break
            # Character overlap preserves terms and sentences that straddle a
            # boundary; the next preferred split can still end on punctuation.
            start = max(end - self._overlap, start + 1)
        return tuple(pieces)

    def _preferred_end(self, text: str, start: int, hard_end: int) -> int:
        if hard_end >= len(text):
            return hard_end
        lower_bound = max(start + 1, hard_end - self._chunk_size // 3)
        candidates = [
            position + len(boundary)
            for boundary in _BOUNDARIES
            if (position := text.rfind(boundary, lower_bound, hard_end)) >= 0
        ]
        return max(candidates, default=hard_end)

    @staticmethod
    def _normalize(content: str) -> str:
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        content = re.sub(r"[ \t]+\n", "\n", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content.strip()
