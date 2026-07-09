"""Local integration test for Milvus Lite persistence and vector search."""

import asyncio
from pathlib import Path

from enterprise_ai_assistant.models import VectorRecord, VectorSearchFilter
from enterprise_ai_assistant.vectorstore import MilvusVectorStore


def test_milvus_lite_create_insert_search_delete(tmp_path: Path) -> None:
    """The real Lite engine should execute the complete Phase 5 lifecycle."""

    async def exercise() -> None:
        store = MilvusVectorStore(
            uri=str(tmp_path / "knowledge.db"),
            collection_name="knowledge_chunks",
            dimension=2,
        )
        records = [
            VectorRecord(
                id="doc-1-0",
                document_id="doc-1",
                content="企业知识库",
                embedding=(1.0, 0.0),
                source="manual.md",
                chunk_index=0,
                metadata={"department": "研发"},
            ),
            VectorRecord(
                id="doc-2-0",
                document_id="doc-2",
                content="摄影器材",
                embedding=(0.0, 1.0),
                source="camera.md",
                chunk_index=0,
                metadata={"department": "市场"},
            ),
        ]
        try:
            await store.ensure_collection()
            inserted = await store.insert(records)
            assert inserted.inserted_count == 2

            results = await store.search(
                (1.0, 0.0),
                top_k=2,
                search_filter=VectorSearchFilter(source="manual.md"),
            )
            assert len(results) == 1
            assert results[0].id == "doc-1-0"
            assert results[0].score == 1.0

            deleted = await store.delete(["doc-1-0", "doc-2-0"])
            assert deleted.deleted_count == 2
        finally:
            await store.close()

    asyncio.run(exercise())
