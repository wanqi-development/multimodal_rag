"""双版本 Milvus 路由：新批次先隐藏，活动指针切换后才进入检索。"""

from __future__ import annotations

import asyncio

from rag.config.schema import VectorIndexConfig
from rag.contracts import Chunk, Embeddings, ScoredChunk
from rag.indexing.milvus.versioned import DualVersionMilvusIndex


def _hit(chunk_id: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=chunk_id, doc_id=chunk_id.split(":")[0], text=chunk_id),
        score=1.0,
        channel="hybrid",
        rank=0,
    )


class _Registry:
    def __init__(self, active: dict[str, str]) -> None:
        self.active = active

    async def active_batches(self) -> dict[str, str]:
        return dict(self.active)


class _VersionIndex:
    def __init__(self) -> None:
        self.filters: list[str | None] = []
        self.upserts: list[tuple[str, list[str]]] = []
        self.doc_chunks: list[Chunk] = []

    async def search_hybrid(self, vector, query, **kwargs) -> list[ScoredChunk]:
        self.filters.append(kwargs.get("filter_expr"))
        filt = kwargs.get("filter_expr") or ""
        if "old" in filt:
            return [_hit("d1:0001")]
        if "new" in filt:
            return [_hit("d1:0002")]
        return []

    async def search_dense(self, vector, top_k, filter_expr=None) -> list[ScoredChunk]:
        return []

    async def search_sparse(self, query, top_k, filter_expr=None) -> list[ScoredChunk]:
        return []

    async def get_chunks(self, chunk_ids, **kwargs) -> list[Chunk]:
        return []

    async def get_doc_chunks(self, doc_id: str, *, extra_filter: str | None = None) -> list[Chunk]:
        self.filters.append(extra_filter)
        return list(self.doc_chunks)

    async def upsert(self, chunks, embeddings, *, batch_id: str) -> int:
        self.upserts.append((batch_id, [chunk.chunk_id for chunk in chunks]))
        return len(chunks)

    async def delete_batch(self, batch_id: str) -> None:
        return None

    async def delete_docs(self, doc_ids: list[str]) -> None:
        return None

    async def count(self) -> int:
        return len(self.doc_chunks)


class _LegacyIndex:
    def __init__(self) -> None:
        self.filters: list[str | None] = []

    async def search_hybrid(self, vector, query, **kwargs) -> list[ScoredChunk]:
        self.filters.append(kwargs.get("filter_expr"))
        return [_hit("d2:0001")]

    async def search_dense(self, vector, top_k, filter_expr=None) -> list[ScoredChunk]:
        return []

    async def search_sparse(self, query, top_k, filter_expr=None) -> list[ScoredChunk]:
        return []

    async def get_chunks(self, chunk_ids, **kwargs) -> list[Chunk]:
        return []

    async def get_doc_chunks(self, doc_id: str, *, extra_filter: str | None = None) -> list[Chunk]:
        return []

    async def delete_docs(self, doc_ids: list[str]) -> None:
        return None

    async def count(self) -> int:
        return 1

    async def has_collection(self) -> bool:
        return True

    async def ping(self, timeout_s: float = 5.0) -> None:
        return None

    async def ensure_ready(self) -> None:
        return None


def _dual(active: dict[str, str]) -> tuple[DualVersionMilvusIndex, _VersionIndex, _LegacyIndex]:
    index = DualVersionMilvusIndex(VectorIndexConfig(collection="base"), 2, _Registry(active))
    versions, legacy = _VersionIndex(), _LegacyIndex()
    index.versions = versions  # type: ignore[assignment]
    index.legacy = legacy  # type: ignore[assignment]
    index._legacy_exists = True
    return index, versions, legacy


def test_only_the_active_batch_is_searchable_after_switching() -> None:
    active = {"d1": "old"}
    index, versions, legacy = _dual(active)

    before = asyncio.run(index.search_hybrid([0.1, 0.2], "q", top_k=5))
    active["d1"] = "new"
    after = asyncio.run(index.search_hybrid([0.1, 0.2], "q", top_k=5))

    assert [hit.chunk.chunk_id for hit in before if hit.chunk.doc_id == "d1"] == ["d1:0001"]
    assert [hit.chunk.chunk_id for hit in after if hit.chunk.doc_id == "d1"] == ["d1:0002"]
    assert "old" in (versions.filters[0] or "")
    assert "new" in (versions.filters[1] or "")
    assert "d1" in (legacy.filters[0] or ""), "legacy 要排除已有活动版本的文档"


def test_upsert_batch_delegates_to_the_versioned_collection() -> None:
    index, versions, _ = _dual({"d1": "old"})
    chunk = Chunk(chunk_id="d1:0001", doc_id="d1", text="正文")
    vectors = Embeddings(dense=[[0.1, 0.2]])

    written = asyncio.run(index.upsert_batch([chunk], vectors, batch_id="new"))

    assert written == 1
    assert versions.upserts == [("new", ["d1:0001"])]


def test_get_doc_chunks_reads_the_active_batch_filter() -> None:
    index, versions, _ = _dual({"d1": "batch-1"})
    versions.doc_chunks = [Chunk(chunk_id="d1:0001", doc_id="d1", text="正文")]

    chunks = asyncio.run(index.get_doc_chunks("d1"))

    assert [chunk.chunk_id for chunk in chunks] == ["d1:0001"]
    assert "batch-1" in (versions.filters[0] or "")
