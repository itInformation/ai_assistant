"""Milvus vector-store adapter supporting Lite and remote deployments."""

import asyncio
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import MilvusException

from enterprise_ai_assistant.config import Settings
from enterprise_ai_assistant.core import get_logger
from enterprise_ai_assistant.models import (
    JSONValue,
    VectorDeleteResult,
    VectorInsertResult,
    VectorRecord,
    VectorSearchFilter,
    VectorSearchResult,
)
from enterprise_ai_assistant.vectorstore.exceptions import (
    VectorStoreConfigurationError,
    VectorStoreProviderError,
)

_OUTPUT_FIELDS = [
    "document_id",
    "content",
    "source",
    "chunk_index",
    "metadata",
]
_COLLECTION_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class MilvusVectorStore:
    """Persist and search document chunks in Milvus."""

    def __init__(
        self,
        *,
        uri: str,
        collection_name: str,
        dimension: int,
        token: str | None = None,
        timeout_seconds: float = 30.0,
        client: MilvusClient | None = None,
    ) -> None:
        """Create a Milvus adapter for a local file or remote endpoint."""

        if not uri.strip():
            raise VectorStoreConfigurationError("Milvus URI must not be empty")
        if _COLLECTION_NAME_PATTERN.fullmatch(collection_name) is None:
            raise VectorStoreConfigurationError(
                "collection_name must be a valid identifier"
            )
        if dimension <= 0:
            raise VectorStoreConfigurationError("dimension must be positive")
        if timeout_seconds <= 0:
            raise VectorStoreConfigurationError(
                "timeout_seconds must be greater than zero"
            )

        self._uri = uri
        self._is_lite = "://" not in uri
        self._collection_name = collection_name
        self._dimension = dimension
        self._timeout_seconds = timeout_seconds
        self._collection_ready = False
        self._collection_lock = asyncio.Lock()
        if client is None:
            self._prepare_local_parent(uri)
            self._client = MilvusClient(
                uri=uri,
                token=token or "",
                timeout=timeout_seconds,
            )
        else:
            self._client = client
        self._logger = get_logger(
            provider="milvus",
            collection=collection_name,
        )

    async def ensure_collection(self) -> None:
        """Create an explicit collection schema or validate the existing one."""

        if self._collection_ready:
            return
        async with self._collection_lock:
            if self._collection_ready:
                return
            try:
                exists = await asyncio.to_thread(
                    self._client.has_collection,
                    collection_name=self._collection_name,
                    timeout=self._timeout_seconds,
                )
                if exists:
                    description = await asyncio.to_thread(
                        self._client.describe_collection,
                        collection_name=self._collection_name,
                        timeout=self._timeout_seconds,
                    )
                    self._validate_collection(description)
                else:
                    await asyncio.to_thread(self._create_collection)
                # Persistent collections can be released when the previous
                # client exits. Load explicitly so a fresh process can search.
                await asyncio.to_thread(
                    self._client.load_collection,
                    collection_name=self._collection_name,
                    timeout=self._timeout_seconds,
                )
            except MilvusException as exc:
                raise self._provider_error("ensure_collection", exc) from exc
            self._collection_ready = True

    async def insert(
        self,
        records: Sequence[VectorRecord],
    ) -> VectorInsertResult:
        """Insert validated document chunks and their vectors."""

        self._validate_records(records)
        await self.ensure_collection()
        entities = [
            {
                "id": record.id,
                "document_id": record.document_id,
                "content": record.content,
                "source": record.source,
                "chunk_index": record.chunk_index,
                "metadata": dict(record.metadata),
                "embedding": list(record.embedding),
            }
            for record in records
        ]
        try:
            result = await asyncio.to_thread(
                self._client.insert,
                collection_name=self._collection_name,
                data=entities,
                timeout=self._timeout_seconds,
            )
        except MilvusException as exc:
            raise self._provider_error("insert", exc) from exc
        return VectorInsertResult(
            inserted_count=int(result["insert_count"]),
            primary_keys=tuple(str(value) for value in result["ids"]),
        )

    async def delete(self, ids: Sequence[str]) -> VectorDeleteResult:
        """Delete records using exact primary keys instead of raw expressions."""

        if not ids or any(not value.strip() for value in ids):
            raise ValueError("delete ids must contain non-empty values")
        try:
            await self.ensure_collection()
            result = await asyncio.to_thread(
                self._client.delete,
                collection_name=self._collection_name,
                ids=list(ids),
                timeout=self._timeout_seconds,
            )
        except MilvusException as exc:
            raise self._provider_error("delete", exc) from exc
        return VectorDeleteResult(deleted_count=self._deleted_count(result))

    async def delete_by_document_id(
        self,
        document_id: str,
    ) -> VectorDeleteResult:
        """Delete all existing chunks before a document is re-indexed."""

        if not document_id.strip():
            raise ValueError("document_id must not be empty")
        await self.ensure_collection()
        expression = f"document_id == {json.dumps(document_id, ensure_ascii=False)}"
        try:
            result = await asyncio.to_thread(
                self._client.delete,
                collection_name=self._collection_name,
                filter=expression,
                timeout=self._timeout_seconds,
            )
        except MilvusException as exc:
            raise self._provider_error("delete_by_document_id", exc) from exc
        return VectorDeleteResult(deleted_count=self._deleted_count(result))

    @staticmethod
    def _deleted_count(result: Any) -> int:
        # Remote Milvus returns {"delete_count": n}; Milvus Lite 3.0 returns
        # the deleted primary-key list. Normalize both behind the port.
        return int(result["delete_count"]) if isinstance(result, dict) else len(result)

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        search_filter: VectorSearchFilter | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Search with COSINE similarity and optional safe scalar filters."""

        if len(query_vector) != self._dimension:
            raise ValueError(f"query vector dimension must be {self._dimension}")
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        await self.ensure_collection()
        try:
            result = await asyncio.to_thread(
                self._client.search,
                collection_name=self._collection_name,
                data=[list(query_vector)],
                anns_field="embedding",
                filter=self._build_filter(search_filter),
                limit=top_k,
                output_fields=_OUTPUT_FIELDS,
                search_params={"metric_type": "COSINE", "params": {}},
                timeout=self._timeout_seconds,
            )
        except MilvusException as exc:
            raise self._provider_error("search", exc) from exc
        if not result:
            return ()
        return tuple(self._map_search_hit(hit) for hit in result[0])

    async def close(self) -> None:
        """Close the underlying PyMilvus client."""

        await asyncio.to_thread(self._client.close)

    def _create_collection(self) -> None:
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=128,
        )
        schema.add_field(
            field_name="document_id",
            datatype=DataType.VARCHAR,
            max_length=128,
        )
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65_535,
        )
        schema.add_field(
            field_name="source",
            datatype=DataType.VARCHAR,
            max_length=1_024,
        )
        schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=self._dimension,
        )

        index_params = MilvusClient.prepare_index_params()
        # Milvus Lite currently supports FLAT only. The same schema remains
        # portable; production can replace this index without domain changes.
        index_params.add_index(
            field_name="embedding",
            index_name="embedding_cosine_idx",
            index_type="FLAT",
            metric_type="COSINE",
        )
        self._client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=index_params,
            timeout=self._timeout_seconds,
        )

    def _validate_collection(self, description: dict[str, Any]) -> None:
        fields = {field["name"]: field for field in description.get("fields", [])}
        required_fields = {
            "id",
            "document_id",
            "content",
            "source",
            "chunk_index",
            "metadata",
            "embedding",
        }
        if not required_fields.issubset(fields):
            raise VectorStoreConfigurationError(
                "existing Milvus collection has an incompatible schema"
            )
        vector_field = fields["embedding"]
        params = vector_field.get("params", {})
        actual_dimension = int(params.get("dim", vector_field.get("dim", 0)))
        if actual_dimension != self._dimension:
            raise VectorStoreConfigurationError(
                "existing Milvus collection dimension does not match settings"
            )

    def _validate_records(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            raise ValueError("at least one vector record is required")
        ids = [record.id for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("vector record ids must be unique within a batch")
        if any(len(record.embedding) != self._dimension for record in records):
            raise ValueError(f"record embedding dimension must be {self._dimension}")

    @staticmethod
    def _build_filter(search_filter: VectorSearchFilter | None) -> str:
        if search_filter is None:
            return ""
        expressions: list[str] = []
        if search_filter.document_ids:
            values = ", ".join(
                json.dumps(value, ensure_ascii=False)
                for value in search_filter.document_ids
            )
            expressions.append(f"document_id in [{values}]")
        if search_filter.source is not None:
            source = json.dumps(search_filter.source, ensure_ascii=False)
            expressions.append(f"source == {source}")
        return " and ".join(expressions)

    def _map_search_hit(self, hit: dict[str, Any]) -> VectorSearchResult:
        entity = hit["entity"]
        metadata: dict[str, JSONValue] = dict(entity.get("metadata") or {})
        raw_distance = float(hit["distance"])
        # Milvus Lite 3.0 currently exposes COSINE as distance (0 is exact),
        # while remote Milvus exposes similarity (1 is exact). Normalize the
        # domain score so callers always treat larger values as more relevant.
        score = 1.0 - raw_distance if self._is_lite else raw_distance
        return VectorSearchResult(
            id=str(hit["id"]),
            score=score,
            document_id=str(entity["document_id"]),
            content=str(entity["content"]),
            source=str(entity["source"]),
            chunk_index=int(entity["chunk_index"]),
            metadata=metadata,
        )

    def _provider_error(
        self,
        operation: str,
        error: Exception,
    ) -> VectorStoreProviderError:
        self._logger.error(
            "vector_store_operation_failed",
            operation=operation,
            error_type=type(error).__name__,
        )
        return VectorStoreProviderError(
            f"Milvus {operation} failed: {type(error).__name__}"
        )

    @staticmethod
    def _prepare_local_parent(uri: str) -> None:
        if "://" in uri:
            return
        path = Path(uri).expanduser().absolute()
        # Never follow a database-file symlink into an unapproved location.
        if path.is_symlink():
            raise VectorStoreConfigurationError(
                "Milvus local database URI must not be a symbolic link"
            )
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


def create_milvus_vector_store(settings: Settings) -> MilvusVectorStore:
    """Build the configured local or remote Milvus adapter."""

    return MilvusVectorStore(
        uri=settings.milvus_uri,
        collection_name=settings.milvus_collection_name,
        dimension=settings.dashscope_embedding_dimensions,
        token=settings.milvus_token,
        timeout_seconds=settings.milvus_timeout_seconds,
    )
