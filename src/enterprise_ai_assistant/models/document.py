"""Provider-independent document and chunk models."""

from dataclasses import dataclass, field

from enterprise_ai_assistant.models.vectorstore import JSONValue


@dataclass(frozen=True, slots=True)
class DocumentSection:
    """A logical source section such as a PDF page or Word table."""

    content: str
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject empty sections before chunking."""

        if not self.content.strip():
            raise ValueError("document section content must not be empty")


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """A parsed source document with one or more logical sections."""

    id: str
    source: str
    sections: tuple[DocumentSection, ...]
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate stable identity and parsed content."""

        if not self.id.strip() or not self.source.strip():
            raise ValueError("document id and source must not be empty")
        if not self.sections:
            raise ValueError("loaded document must contain at least one section")


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A deterministic overlapping text chunk ready for embedding."""

    id: str
    document_id: str
    content: str
    source: str
    chunk_index: int
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate chunk identity and content."""

        if not self.id.strip() or not self.document_id.strip():
            raise ValueError("chunk id and document_id must not be empty")
        if not self.content.strip() or not self.source.strip():
            raise ValueError("chunk content and source must not be empty")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must not be negative")
