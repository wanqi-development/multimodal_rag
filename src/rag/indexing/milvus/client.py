"""Milvus 索引的读写。三条实测得来的硬约束都体现在这里：

1. **写后必须 flush**，且查询要带 ``consistency_level="Strong"``，否则刚写的
   数据查不到 —— 表现是静默返回 0 条，看起来像「检索没召回」；
2. **命中的主键按 schema 字段名取**（``hit["chunk_id"]``），不是 ``hit["id"]``；
3. **sparse 字段不能写也不能读**。写入会抛 DataNotMatchException
   （unexpected function output field），读取会抛 MilvusException
   （not allowed to retrieve raw data）—— 它由服务端 BM25 function 从 text 生成，
   想验证它只能靠检索命中。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import Any

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from rag.config.schema import VectorIndexConfig
from rag.contracts import Chunk, Embeddings, ScoredChunk
from rag.indexing.milvus.schema import (
    MAX_ASSETS,
    MAX_TEXT_BYTES,
    build_index_params,
    build_schema,
)

# 一次 upsert 的行数。攒太多会撞 gRPC 消息上限，太小则往返多；
# 按 chunk 正文 ~1KB 估，256 行约 300KB，安全区间内。
UPSERT_BATCH = 256

# 刚写完就读的场景（ingest 后立刻验证）必须强一致，否则可能读到旧快照
STRONG: dict[str, Any] = {"consistency_level": "Strong"}

_SYNC_TIMEOUT_S = 30


class MilvusIndex:
    """Milvus 的 Index 实现。

    pymilvus 是同步客户端，所有调用都扔进线程池，别堵事件循环。
    """

    def __init__(
        self,
        cfg: VectorIndexConfig,
        dim: int,
        *,
        include_batch_id: bool = False,
    ) -> None:
        self._cfg = cfg
        self._dim = dim
        self._include_batch_id = include_batch_id
        self.name = f"milvus:{cfg.collection}"

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def collection(self) -> str:
        return self._cfg.collection

    @cached_property
    def _client(self) -> MilvusClient:
        return MilvusClient(uri=self._cfg.uri, token=self._cfg.token, timeout=_SYNC_TIMEOUT_S)

    @property
    def _output_fields(self) -> list[str]:
        fields = list(_OUTPUT_FIELDS)
        if self._include_batch_id:
            fields.append("index_batch_id")
            fields.append("logical_chunk_id")
        return fields

    # ---- 生命周期 ----

    async def ensure_ready(self) -> None:
        """建表建索引并 load，幂等。"""
        await asyncio.to_thread(self._ensure_ready_sync)

    async def ping(self, timeout_s: float = 5.0) -> None:
        """探活：连不上就**在这个超时内**抛出来，别让它慢慢重连。

        为什么单独有一个探活：Milvus 停掉时 pymilvus 的重连会空转很久 —— 实测 `rag ingest`
        在写库那一步烧了三十几分钟 CPU、数据一行没进（§16），而且 OCR 那几十分钟的活白干。
        开工前先拿最便宜的一次 RPC 试一下，接不住就当场退出。

        注意 `asyncio.wait_for` 截断的是**我们这一侧的等待**：超时后那个线程还在后台跑
        （`to_thread` 取消不了线程）。CLI 会立刻退出所以无所谓，**但别在长驻服务里靠它回收**。
        """
        await asyncio.wait_for(
            asyncio.to_thread(self._client.has_collection, self._cfg.collection), timeout_s
        )

    def _ensure_ready_sync(self) -> None:
        if self._cfg.backend == "milvus-lite":
            Path(self._cfg.uri).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        if not self._client.has_collection(self._cfg.collection):
            self._client.create_collection(
                self._cfg.collection,
                schema=build_schema(
                    self._cfg,
                    self._dim,
                    include_batch_id=self._include_batch_id,
                ),
                index_params=build_index_params(self._client, self._cfg),
            )
        # 带 index_params 建表时 Milvus 会自动 load；对已存在的 collection 则要显式 load。
        # 实测重复 load 不报错，所以无条件调用，省掉一次状态判断。
        self._client.load_collection(self._cfg.collection)

    async def get_chunks(
        self,
        chunk_ids: list[str],
        *,
        logical: bool = False,
        filter_expr: str | None = None,
    ) -> list[Chunk]:
        """按主键批量取块（**图通道**用它把「关系命中的 chunk_id」换回正文）。

        只读一次 query、不 embed：图谱那边只给得出 chunk_id，正文还在 Milvus 里。
        """
        ids = [cid for cid in dict.fromkeys(chunk_ids) if cid and '"' not in cid]
        if not ids:
            return []
        rows: list[dict[str, Any]] = []
        for batch in _batched(ids, UPSERT_BATCH):
            rows.extend(
                await asyncio.to_thread(
                    self._query_chunks,
                    batch,
                    "logical_chunk_id" if logical else "chunk_id",
                    filter_expr,
                )
            )
        return [_to_chunk(row) for row in rows]

    def _query_chunks(
        self,
        chunk_ids: list[str],
        field: str = "chunk_id",
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        expr = f"{field} in [" + ", ".join(f'"{cid}"' for cid in chunk_ids) + "]"
        if filter_expr:
            expr = f"({expr}) and ({filter_expr})"
        return self._client.query(
            self._cfg.collection, filter=expr, output_fields=self._output_fields, **STRONG
        )

    async def drop(self) -> None:
        """删表。只给测试与重建用 —— 生产环境没有「顺手删库」这种操作。"""
        await asyncio.to_thread(self._drop_sync)

    def _drop_sync(self) -> None:
        if self._client.has_collection(self._cfg.collection):
            self._client.release_collection(self._cfg.collection)
            self._client.drop_collection(self._cfg.collection)

    async def has_collection(self) -> bool:
        return await asyncio.to_thread(self._client.has_collection, self._cfg.collection)

    async def count(self) -> int:
        rows = await asyncio.to_thread(
            self._client.query,
            self._cfg.collection,
            filter="chunk_id != ''",
            output_fields=["count(*)"],
            **STRONG,
        )
        return int(rows[0]["count(*)"])

    async def get_doc_chunks(
        self,
        doc_id: str,
        *,
        extra_filter: str | None = None,
    ) -> list[Chunk]:
        """按 doc_id 取全部块：对账要比块数、chunk_id 与内容指纹。"""
        rows = await asyncio.to_thread(self._query_doc_chunks, doc_id, extra_filter)
        return [_to_chunk(row) for row in rows]

    def _query_doc_chunks(
        self,
        doc_id: str,
        extra_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        expr = f"doc_id == {json.dumps(doc_id, ensure_ascii=False)}"
        if extra_filter:
            expr = f"({expr}) and ({extra_filter})"
        return self._client.query(
            self._cfg.collection,
            filter=expr,
            output_fields=self._output_fields,
            **STRONG,
        )

    # ---- 写 ----

    async def upsert(
        self,
        chunks: list[Chunk],
        embeddings: Embeddings,
        *,
        batch_id: str = "",
    ) -> int:
        return await asyncio.to_thread(self._upsert_sync, chunks, embeddings, batch_id)

    def _upsert_sync(
        self,
        chunks: list[Chunk],
        embeddings: Embeddings,
        batch_id: str,
    ) -> int:
        if len(chunks) != len(embeddings.dense):
            raise ValueError(
                f"chunk 数与向量数不一致：{len(chunks)} 个 chunk vs "
                f"{len(embeddings.dense)} 条 dense 向量"
            )
        if embeddings.dense and len(embeddings.dense[0]) != self._dim:
            raise ValueError(
                f"向量维度 {len(embeddings.dense[0])} 与 collection 的 {self._dim} 不符，"
                f"检查 providers.embedding.dim"
            )

        if self._include_batch_id and not batch_id:
            raise ValueError("版本化 collection 写入必须提供 batch_id")
        rows = [
            self._row(c, vec, batch_id) for c, vec in zip(chunks, embeddings.dense, strict=True)
        ]
        written = 0
        for batch in _batched(rows, UPSERT_BATCH):
            self._client.upsert(self._cfg.collection, data=batch)
            written += len(batch)
        # 不 flush 的话，紧接着的检索会看不到这批数据（见模块开头的第 1 条）
        self._client.flush(self._cfg.collection)
        return written

    def _row(self, chunk: Chunk, dense: list[float], batch_id: str = "") -> dict[str, Any]:
        # 这里绝不能出现 sparse：它是 BM25 function 的输出字段，
        # 手动带上会被服务端直接拒绝
        size = len(chunk.text.encode("utf-8"))
        if size > MAX_TEXT_BYTES:
            raise ValueError(
                f"chunk {chunk.chunk_id} 正文 {size} 字节，超过 text 字段上限 {MAX_TEXT_BYTES} 字节"
                f"（max_length 按 UTF-8 字节算，一个汉字 3 字节）。调小 indexing.chunk.size。"
            )
        if len(chunk.assets) > MAX_ASSETS:
            # 与 text 超长同一个立场：当场报错，不静默截断 —— 悄悄少几张图是查不出来的
            raise ValueError(
                f"chunk {chunk.chunk_id} 引用了 {len(chunk.assets)} 个资产，"
                f"超过 assets 字段容量 {MAX_ASSETS}。调小 indexing.chunk.size。"
            )
        row = {
            "chunk_id": (
                f"{batch_id}:{chunk.chunk_id}" if self._include_batch_id else chunk.chunk_id
            ),
            "doc_id": chunk.doc_id,
            "chunk_index": chunk.index,
            "source_uri": str(chunk.metadata.get("source_uri", "")),
            "doc_type": chunk.doc_type,
            "modality": chunk.modality,
            "heading_path": chunk.heading_path,
            "heading_numbers": chunk.heading_numbers,
            "page": chunk.page,
            "line": chunk.line,
            "assets": list(chunk.assets),
            "content_hash": chunk.content_hash,
            "metadata": dict(chunk.metadata),
            "text": chunk.text,
            "dense": dense,
        }
        if self._include_batch_id:
            row["index_batch_id"] = batch_id
            row["logical_chunk_id"] = chunk.chunk_id
        return row

    async def delete(self, chunk_ids: list[str]) -> int:
        return await asyncio.to_thread(self._delete_sync, chunk_ids)

    def _delete_sync(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        quoted = ", ".join(f'"{cid}"' for cid in chunk_ids)
        self._client.delete(self._cfg.collection, filter=f"chunk_id in [{quoted}]")
        self._client.flush(self._cfg.collection)
        return len(chunk_ids)

    async def delete_doc(self, doc_id: str) -> None:
        """按 doc_id 清掉一份文档的全部块。删文档时用，避免留下孤儿块。"""
        await self.delete_docs([doc_id])

    async def delete_docs(self, doc_ids: list[str]) -> None:
        """批量按 doc_id 清理。

        重新 ingest 一份「变短了」的文档时必须先清旧块：chunk_id 是 doc_id + 序号，
        upsert 只覆盖同序号的块，多出来的旧序号会变成永远检索得到、却已不存在的正文。
        """
        if not doc_ids:
            return
        await asyncio.to_thread(self._delete_docs_sync, doc_ids)

    def _delete_docs_sync(self, doc_ids: list[str]) -> None:
        quoted = ", ".join(f'"{doc_id}"' for doc_id in doc_ids)
        self._client.delete(self._cfg.collection, filter=f"doc_id in [{quoted}]")
        self._client.flush(self._cfg.collection)

    async def delete_batch(self, batch_id: str) -> None:
        """只删一个候选/旧批次；活动版本不受影响。"""
        if not self._include_batch_id:
            raise ValueError("legacy collection 没有 index_batch_id，不能按批次清理")
        await asyncio.to_thread(self._delete_batch_sync, batch_id)

    def _delete_batch_sync(self, batch_id: str) -> None:
        self._client.delete(
            self._cfg.collection,
            filter=f"index_batch_id == {json.dumps(batch_id, ensure_ascii=False)}",
        )
        self._client.flush(self._cfg.collection)

    # ---- 读 ----

    async def search_dense(
        self, vector: list[float], top_k: int, filter_expr: str | None = None
    ) -> list[ScoredChunk]:
        hits = await asyncio.to_thread(
            self._search_sync,
            vector,
            "dense",
            self._cfg.indexes["dense"].metric,
            top_k,
            filter_expr,
        )
        return _to_scored(hits, "dense")

    async def search_sparse(
        self, query: str, top_k: int, filter_expr: str | None = None
    ) -> list[ScoredChunk]:
        """sparse 路直接传查询原文：分词和 term 权重由服务端 BM25 算。"""
        hits = await asyncio.to_thread(
            self._search_sync,
            query,
            "sparse",
            self._cfg.indexes["sparse"].metric,
            top_k,
            filter_expr,
        )
        return _to_scored(hits, "sparse")

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
        """dense 与 sparse 各召回一批，由 Milvus 服务端融合。

        ranker 由调用方给（RRFRanker / WeightedRanker）：融合策略属于检索层的配置，
        索引层不该替它决定。分数是融合分，与余弦相似度不同量纲，不能跨通道直接比。
        """
        return await asyncio.to_thread(
            self._hybrid_sync,
            vector,
            query,
            top_k,
            dense_limit,
            sparse_limit,
            ranker if ranker is not None else RRFRanker(60),
            filter_expr,
        )

    def _hybrid_sync(
        self,
        vector: list[float],
        query: str,
        top_k: int,
        dense_limit: int,
        sparse_limit: int,
        ranker: Any,
        filter_expr: str | None,
    ) -> list[ScoredChunk]:
        extra = {"filter": filter_expr} if filter_expr else {}
        reqs = [
            AnnSearchRequest(
                data=[vector],
                anns_field="dense",
                param={"metric_type": self._cfg.indexes["dense"].metric},
                limit=dense_limit,
                **extra,
            ),
            AnnSearchRequest(
                data=[query],
                anns_field="sparse",
                param={"metric_type": self._cfg.indexes["sparse"].metric},
                limit=sparse_limit,
                **extra,
            ),
        ]
        hits = self._client.hybrid_search(
            self._cfg.collection,
            reqs,
            ranker=ranker,
            limit=top_k,
            output_fields=self._output_fields,
            **STRONG,
        )[0]
        return _to_scored(hits, "hybrid")

    def _search_sync(
        self,
        data: Any,
        field: str,
        metric: str,
        top_k: int,
        filter_expr: str | None,
    ) -> list[dict[str, Any]]:
        extra = {"filter": filter_expr} if filter_expr else {}
        return self._client.search(
            self._cfg.collection,
            data=[data],
            anns_field=field,
            search_params={"metric_type": metric},
            limit=top_k,
            output_fields=self._output_fields,
            **STRONG,
            **extra,
        )[0]


_OUTPUT_FIELDS = [
    "doc_id",
    "chunk_index",
    "source_uri",
    "doc_type",
    "modality",
    "heading_path",
    "heading_numbers",
    "page",
    "line",
    "assets",
    "content_hash",
    "metadata",
    "text",
]


def _batched(rows: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _to_chunk(row: dict[str, Any]) -> Chunk:
    """一行 Milvus 记录 → Chunk。

    主键从 ``row["chunk_id"]`` 取（字段名），不是 ``row["id"]`` —— 实测 pymilvus 3.x
    把主键暴露在 schema 的字段名上。其余字段在 ``row["entity"]`` 里，Hit 做了转发，
    直接 ``row["text"]`` 也能取到。
    """
    return Chunk(
        chunk_id=row.get("logical_chunk_id") or row["chunk_id"],
        doc_id=row.get("doc_id", ""),
        text=row.get("text", ""),
        index=int(row.get("chunk_index", 0)),
        doc_type=row.get("doc_type") or "",
        modality=row.get("modality") or "text",
        heading_path=row.get("heading_path") or "",
        heading_numbers=row.get("heading_numbers") or "",
        page=int(row.get("page") or 0),
        line=int(row.get("line") or 0),
        assets=list(row.get("assets") or []),
        content_hash=row.get("content_hash") or "",
        metadata=dict(row.get("metadata") or {}),
    )


def _to_scored(hits: list[dict[str, Any]], channel: str) -> list[ScoredChunk]:
    """Hit → ScoredChunk（正文映射见 `_to_chunk`）。"""
    return [
        ScoredChunk(
            chunk=_to_chunk(hit),
            score=float(hit.get("distance", 0.0)),
            channel=channel,
            rank=rank,
        )
        for rank, hit in enumerate(hits)
    ]
