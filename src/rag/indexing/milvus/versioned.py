"""双版本 Milvus：legacy 只读，新批次写版本化 collection。"""

from __future__ import annotations

import json
from typing import Any

from rag.config.schema import VectorIndexConfig
from rag.contracts import Chunk, Embeddings, ScoredChunk
from rag.indexing.milvus.client import MilvusIndex


def _and(*parts: str | None) -> str | None:
    kept = [f"({part})" for part in parts if part]
    return " and ".join(kept) or None


def _quoted(values: set[str] | list[str]) -> str:
    return ", ".join(json.dumps(value, ensure_ascii=False) for value in values)


class DualVersionMilvusIndex:
    """保持旧 collection 可读，同时从版本化 collection 切活动批次。

    - 新写入只进 `<collection>_versions`，不删旧批次；
    - 只有 registry 中 `index_status=active` 的 batch 参与版本化检索；
    - legacy collection 仍在读，但排除已经有活动版本的 doc_id。
    """

    def __init__(
        self,
        cfg: VectorIndexConfig,
        dim: int,
        registry: Any = None,
        settings: Any = None,
    ) -> None:
        versions = cfg.versions_collection or f"{cfg.collection}_versions"
        version_cfg = cfg.model_copy(update={"collection": versions})
        self._cfg = cfg
        self._registry_instance = registry
        self._settings = settings
        self.legacy = MilvusIndex(cfg, dim, include_batch_id=False)
        self.versions = MilvusIndex(version_cfg, dim, include_batch_id=True)
        self.name = f"milvus-dual:{cfg.collection}+{versions}"
        self._legacy_exists: bool | None = None

    @property
    def collection(self) -> str:
        return self.versions.collection

    async def ensure_ready(self) -> None:
        await self.versions.ensure_ready()
        self._legacy_exists = await self.legacy.has_collection()

    async def ping(self, timeout_s: float = 5.0) -> None:
        await self.versions.ping(timeout_s)

    async def upsert_batch(
        self,
        chunks: list[Chunk],
        embeddings: Embeddings,
        *,
        batch_id: str,
    ) -> int:
        return await self.versions.upsert(chunks, embeddings, batch_id=batch_id)

    async def delete_batch(self, batch_id: str) -> None:
        await self.versions.delete_batch(batch_id)

    async def delete_docs(self, doc_ids: list[str]) -> None:
        await self.versions.delete_docs(doc_ids)
        if await self._legacy_available():
            await self.legacy.delete_docs(doc_ids)

    async def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        active = await self._active_batches()
        version_filter = _batch_filter(set(active.values()))
        versioned = await self.versions.get_chunks(
            chunk_ids,
            logical=True,
            filter_expr=version_filter,
        )
        legacy: list[Chunk] = []
        if await self._legacy_available():
            legacy = await self.legacy.get_chunks(
                chunk_ids,
                filter_expr=_legacy_filter(set(active)),
            )
        return _unique_chunks([*versioned, *legacy])

    async def get_doc_chunks(self, doc_id: str) -> list[Chunk]:
        active = await self._active_batches()
        batch_id = active.get(doc_id, "")
        if batch_id:
            return await self.versions.get_doc_chunks(
                doc_id,
                extra_filter=_batch_filter({batch_id}),
            )
        if await self._legacy_available():
            return await self.legacy.get_doc_chunks(doc_id)
        return []

    async def count(self) -> int:
        counts = [await self.versions.count()]
        if await self._legacy_available():
            counts.append(await self.legacy.count())
        return sum(counts)

    async def search_dense(
        self, vector: list[float], top_k: int, filter_expr: str | None = None
    ) -> list[ScoredChunk]:
        active = await self._active_batches()
        boards: list[list[ScoredChunk]] = []
        if active:
            boards.append(
                await self.versions.search_dense(
                    vector,
                    top_k,
                    filter_expr=_and(filter_expr, _batch_filter(set(active.values()))),
                )
            )
        if await self._legacy_available():
            boards.append(
                await self.legacy.search_dense(
                    vector,
                    top_k,
                    filter_expr=_and(filter_expr, _legacy_filter(set(active))),
                )
            )
        return _merge(boards, top_k)

    async def search_sparse(
        self, query: str, top_k: int, filter_expr: str | None = None
    ) -> list[ScoredChunk]:
        active = await self._active_batches()
        boards: list[list[ScoredChunk]] = []
        if active:
            boards.append(
                await self.versions.search_sparse(
                    query,
                    top_k,
                    filter_expr=_and(filter_expr, _batch_filter(set(active.values()))),
                )
            )
        if await self._legacy_available():
            boards.append(
                await self.legacy.search_sparse(
                    query,
                    top_k,
                    filter_expr=_and(filter_expr, _legacy_filter(set(active))),
                )
            )
        return _merge(boards, top_k)

    async def search_hybrid(
        self,
        vector: list[float],
        query: str,
        *,
        top_k: int,
        dense_limit: int = 20,
        sparse_limit: int = 20,
        ranker: Any = None,
        filter_expr: str | None = None,
    ) -> list[ScoredChunk]:
        active = await self._active_batches()
        boards: list[list[ScoredChunk]] = []
        if active:
            boards.append(
                await self.versions.search_hybrid(
                    vector,
                    query,
                    top_k=top_k,
                    dense_limit=dense_limit,
                    sparse_limit=sparse_limit,
                    ranker=ranker,
                    filter_expr=_and(filter_expr, _batch_filter(set(active.values()))),
                )
            )
        if await self._legacy_available():
            boards.append(
                await self.legacy.search_hybrid(
                    vector,
                    query,
                    top_k=top_k,
                    dense_limit=dense_limit,
                    sparse_limit=sparse_limit,
                    ranker=ranker,
                    filter_expr=_and(filter_expr, _legacy_filter(set(active))),
                )
            )
        return _merge(boards, top_k)

    async def _legacy_available(self) -> bool:
        if self._legacy_exists is None:
            self._legacy_exists = await self.legacy.has_collection()
        return self._legacy_exists

    async def _active_batches(self) -> dict[str, str]:
        if self._registry_instance is None:
            if self._settings is None:
                raise RuntimeError("双版本索引缺少 registry / settings")
            from rag.indexing import build_registry

            self._registry_instance = build_registry(self._settings)
        return await self._registry_instance.active_batches()


def _batch_filter(batch_ids: set[str]) -> str | None:
    if not batch_ids:
        return 'index_batch_id == "__none__"'
    return f"index_batch_id in [{_quoted(batch_ids)}]"


def _legacy_filter(active_docs: set[str]) -> str | None:
    if not active_docs:
        return None
    return f"doc_id not in [{_quoted(active_docs)}]"


def _merge(boards: list[list[ScoredChunk]], top_k: int) -> list[ScoredChunk]:
    if not boards:
        return []
    if len(boards) == 1:
        return boards[0][:top_k]
    # 两路都是已融合榜，只按名次再做一次 RRF；不让某一路的分数尺度假装可比。
    from rag.retrieval.fusion import rrf_fuse

    return rrf_fuse(boards, k=60)[:top_k]


def _unique_chunks(chunks: list[Chunk]) -> list[Chunk]:
    found: dict[str, Chunk] = {}
    for chunk in chunks:
        found.setdefault(chunk.chunk_id, chunk)
    return list(found.values())
