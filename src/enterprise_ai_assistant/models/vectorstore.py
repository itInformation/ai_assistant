"""Provider-independent models for vector persistence and search."""

import json
import re
from dataclasses import dataclass, field
from typing import TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
_METADATA_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """One document chunk and its dense embedding."""

    id: str
    document_id: str
    content: str
    embedding: tuple[float, ...]
    source: str
    chunk_index: int
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate data before it crosses the vector-store boundary."""

        if not self.id.strip() or not self.document_id.strip():
            raise ValueError("record id and document_id must not be empty")
        if not self.content.strip() or not self.source.strip():
            raise ValueError("record content and source must not be empty")
        if len(self.id) > 128 or len(self.document_id) > 128:
            raise ValueError("record id and document_id must not exceed 128 characters")
        if len(self.content) > 65_535 or len(self.source) > 1_024:
            raise ValueError("record content or source exceeds the Milvus field limit")
        if not self.embedding:
            raise ValueError("record embedding must not be empty")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must not be negative")
        try:
            encoded_metadata = json.dumps(self.metadata, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON serializable") from exc
        if len(encoded_metadata.encode("utf-8")) > 65_536:
            raise ValueError("metadata exceeds the Milvus JSON field limit")
        if any(_METADATA_KEY_PATTERN.fullmatch(key) is None for key in self.metadata):
            raise ValueError("metadata keys must use letters, numbers, and underscores")


@dataclass(frozen=True, slots=True)
class VectorSearchFilter:
    """Safe structured filters supported by the vector-store port."""

    document_ids: tuple[str, ...] = ()
    source: str | None = None

    def __post_init__(self) -> None:
        """Reject empty values that would create ambiguous filters."""

        if any(not document_id.strip() for document_id in self.document_ids):
            raise ValueError("filter document_ids must not contain empty values")
        if self.source is not None and not self.source.strip():
            raise ValueError("filter source must not be empty")


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """One vector-search hit with its chunk metadata."""

    id: str
    score: float
    document_id: str
    content: str
    source: str
    chunk_index: int
    metadata: dict[str, JSONValue]


@dataclass(frozen=True, slots=True)
class VectorInsertResult:
    """Summary of a vector insert operation."""

    inserted_count: int
    primary_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VectorDeleteResult:
    """Summary of a vector delete operation."""

    deleted_count: int
