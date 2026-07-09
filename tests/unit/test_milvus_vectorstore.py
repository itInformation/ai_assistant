"""Unit tests for the Milvus vector-store adapter."""

import asyncio
from typing import Any, cast

import pytest
from pymilvus import MilvusClient

from enterprise_ai_assistant.models import VectorRecord, VectorSearchFilter
from enterprise_ai_assistant.vectorstore import (
    MilvusVectorStore,
    VectorStoreConfigurationError,
)


class FakeMilvusClient:
    """Small synchronous fake matching methods used by the adapter."""

    def __init__(
        self,
        *,
        exists: bool = False,
        dimension: int = 2,
        lite_delete_result: bool = False,
    ) -> None:
        self.exists = exists
        self.dimension = dimension
        self.lite_delete_result = lite_delete_result
        self.created: dict[str, Any] | None = None
        self.inserted: list[dict[str, Any]] = []
        self.deleted_ids: list[str] = []
        self.search_request: dict[str, Any] | None = None
        self.closed = False

    def has_collection(self, **kwargs: Any) -> bool:
        """Return configured collection existence."""

        return self.exists

    def describe_collection(self, **kwargs: Any) -> dict[str, Any]:
        """Return an explicit compatible collection description."""

        names = [
            "id",
            "document_id",
            "content",
            "source",
            "chunk_index",
            "metadata",
        ]
        fields = [{"name": name} for name in names]
        fields.append({"name": "embedding", "params": {"dim": self.dimension}})
        return {"fields": fields}

    def create_collection(self, **kwargs: Any) -> None:
        """Record collection creation arguments."""

        self.created = kwargs
        self.exists = True

    def insert(self, **kwargs: Any) -> dict[str, Any]:
        """Record entities and return their primary keys."""

        self.inserted = kwargs["data"]
        return {
            "insert_count": len(self.inserted),
            "ids": [entity["id"] for entity in self.inserted],
        }

    def delete(self, **kwargs: Any) -> dict[str, int] | list[str]:
        """Record exact primary-key deletion."""

        self.deleted_ids = kwargs["ids"]
        if self.lite_delete_result:
            return self.deleted_ids
        return {"delete_count": len(self.deleted_ids)}

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        """Record search arguments and return one realistic hit."""

        self.search_request = kwargs
        return [
            [
                {
                    "id": "doc-1-0",
                    "distance": 0.05,
                    "entity": {
                        "document_id": "doc-1",
                        "content": "企业知识库",
                        "source": "manual.md",
                        "chunk_index": 0,
                        "metadata": {"department": "研发"},
                    },
                }
            ]
        ]

    def close(self) -> None:
        """Record client closure."""

        self.closed = True


def record(
    *,
    record_id: str = "doc-1-0",
    embedding: tuple[float, ...] = (1.0, 0.0),
) -> VectorRecord:
    """Build one valid vector record."""

    return VectorRecord(
        id=record_id,
        document_id="doc-1",
        content="企业知识库",
        embedding=embedding,
        source="manual.md",
        chunk_index=0,
        metadata={"department": "研发"},
    )


def build_store(
    client: FakeMilvusClient,
    *,
    dimension: int = 2,
) -> MilvusVectorStore:
    """Build an adapter around the synchronous fake client."""

    return MilvusVectorStore(
        uri="unused.db",
        collection_name="knowledge_chunks",
        dimension=dimension,
        client=cast(MilvusClient, client),
    )


def test_ensure_collection_creates_explicit_schema_and_index() -> None:
    """Missing collections should be created with an explicit schema."""

    client = FakeMilvusClient()
    store = build_store(client)

    asyncio.run(store.ensure_collection())

    assert client.created is not None
    assert client.created["collection_name"] == "knowledge_chunks"
    assert client.created["schema"].enable_dynamic_field is False
    index = client.created["index_params"][0].to_dict()
    assert index["index_type"] == "FLAT"
    assert index["metric_type"] == "COSINE"


def test_existing_collection_dimension_is_validated() -> None:
    """A collection created for another model dimension must fail fast."""

    store = build_store(
        FakeMilvusClient(exists=True, dimension=3),
        dimension=2,
    )

    with pytest.raises(VectorStoreConfigurationError, match="dimension"):
        asyncio.run(store.ensure_collection())


def test_insert_maps_domain_records() -> None:
    """Insert should preserve scalar fields, JSON metadata, and vectors."""

    client = FakeMilvusClient(exists=True)
    store = build_store(client)

    result = asyncio.run(store.insert([record()]))

    assert result.inserted_count == 1
    assert result.primary_keys == ("doc-1-0",)
    assert client.inserted[0]["metadata"] == {"department": "研发"}
    assert client.inserted[0]["embedding"] == [1.0, 0.0]


def test_insert_rejects_duplicate_ids_and_wrong_dimensions() -> None:
    """Batch corruption should be rejected before a provider call."""

    store = build_store(FakeMilvusClient(exists=True))

    with pytest.raises(ValueError, match="unique"):
        asyncio.run(store.insert([record(), record()]))
    with pytest.raises(ValueError, match="dimension"):
        asyncio.run(store.insert([record(embedding=(1.0,))]))


def test_search_maps_hits_and_builds_safe_filter() -> None:
    """Search should map metadata and build expressions from typed filters."""

    client = FakeMilvusClient(exists=True)
    store = build_store(client)

    results = asyncio.run(
        store.search(
            [1.0, 0.0],
            top_k=3,
            search_filter=VectorSearchFilter(
                document_ids=("doc-1", "doc-2"),
                source="manual.md",
            ),
        )
    )

    assert results[0].id == "doc-1-0"
    assert results[0].score == 0.95
    assert results[0].metadata == {"department": "研发"}
    assert client.search_request is not None
    assert client.search_request["filter"] == (
        'document_id in ["doc-1", "doc-2"] and source == "manual.md"'
    )
    assert client.search_request["search_params"]["metric_type"] == "COSINE"


def test_delete_uses_exact_primary_keys() -> None:
    """Delete should avoid exposing raw provider expressions."""

    client = FakeMilvusClient(exists=True)
    store = build_store(client)

    result = asyncio.run(store.delete(["doc-1-0"]))

    assert result.deleted_count == 1
    assert client.deleted_ids == ["doc-1-0"]


def test_delete_normalizes_milvus_lite_primary_key_result() -> None:
    """Lite primary-key lists should map to the shared delete result."""

    client = FakeMilvusClient(exists=True, lite_delete_result=True)
    store = build_store(client)

    result = asyncio.run(store.delete(["doc-1-0", "doc-2-0"]))

    assert result.deleted_count == 2


def test_search_validates_query_and_top_k() -> None:
    """Invalid search requests should fail before reaching Milvus."""

    store = build_store(FakeMilvusClient(exists=True))

    with pytest.raises(ValueError, match="dimension"):
        asyncio.run(store.search([1.0]))
    with pytest.raises(ValueError, match="top_k"):
        asyncio.run(store.search([1.0, 0.0], top_k=0))


def test_configuration_rejects_invalid_collection_name() -> None:
    """Milvus resource names should be checked during construction."""

    with pytest.raises(VectorStoreConfigurationError, match="identifier"):
        MilvusVectorStore(
            uri="unused.db",
            collection_name="invalid-name",
            dimension=2,
            client=cast(MilvusClient, FakeMilvusClient()),
        )


def test_close_releases_client() -> None:
    """Adapter closure should release PyMilvus resources."""

    client = FakeMilvusClient()
    store = build_store(client)

    asyncio.run(store.close())

    assert client.closed is True
