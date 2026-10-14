"""文档注册表：每份文档一行，增量更新的依据。

**为什么单开一张表**：块表是 O(块数) 的，做 diff 时去扫它既慢又没必要；注册表是
O(文档数) 的，读一遍就能算出「新增 / 变更 / 跳过 / 删除」。它也是「这份文件上次灌的
是什么、什么时候灌的」的答案 —— 没有它就只能每次全量重灌。

设计见 docs/milvus-schema-design.md §2.2 / §4。
"""

from __future__ import annotations

import asyncio
from functools import cached_property
from typing import Any

from pymilvus import MilvusClient

from rag.config.schema import VectorIndexConfig
from rag.contracts import DocRecord
from rag.indexing.milvus.schema import (
    build_registry_index_params,
    build_registry_schema,
    registry_collection,
)

_SYNC_TIMEOUT_S = 30
STRONG: dict[str, Any] = {"consistency_level": "Strong"}

# 一次能列出来的文档数上限。Milvus query 默认封顶 16384；
# 语料规模超过这个数时要分页 —— 先记着，不假装它不存在。
LIST_LIMIT = 16384

_FIELDS = [
    "doc_id",
    "source_uri",
    "content_hash",
    "chunk_count",
    "ingested_at",
    "pipeline_version",
]


class DocRegistry:
    """注册表的读写。与 MilvusIndex 同一套约定：同步客户端扔线程池。"""

    def __init__(self, cfg: VectorIndexConfig) -> None:
        self._cfg = cfg
        self.name = f"milvus:{self.collection}"

    @property
    def collection(self) -> str:
        return registry_collection(self._cfg)

    @cached_property
    def _client(self) -> MilvusClient:
        return MilvusClient(uri=self._cfg.uri, token=self._cfg.token, timeout=_SYNC_TIMEOUT_S)

    async def ensure_ready(self) -> None:
        """建表并 load，幂等。只有 DDL，不写任何数据。"""
        await asyncio.to_thread(self._ensure_ready_sync)

    def _ensure_ready_sync(self) -> None:
        if not self._client.has_collection(self.collection):
            self._client.create_collection(
                self.collection,
                schema=build_registry_schema(),
                index_params=build_registry_index_params(self._client),
            )
        # 与块表同一套：重复 load 不报错，所以无条件调用
        self._client.load_collection(self.collection)

    async def drop(self) -> None:
        """删表。只给测试与重建用 —— 生产环境没有「顺手删库」这种操作。"""
        await asyncio.to_thread(self._drop_sync)

    def _drop_sync(self) -> None:
        if self._client.has_collection(self.collection):
            self._client.release_collection(self.collection)
            self._client.drop_collection(self.collection)

    async def all(self) -> dict[str, DocRecord]:
        """全部记录，键是 doc_id。做 diff 时读一遍就够。"""
        rows = await asyncio.to_thread(
            self._client.query,
            self.collection,
            filter="doc_id != ''",
            output_fields=_FIELDS,
            limit=LIST_LIMIT,
            **STRONG,
        )
        return {row["doc_id"]: _to_record(row) for row in rows}

    async def upsert(self, records: list[DocRecord]) -> int:
        if not records:
            return 0
        return await asyncio.to_thread(self._upsert_sync, records)

    async def get(self, doc_id: str) -> DocRecord | None:
        """取**这一份**的注册记录（详情页要答「这次灌了多少块、什么时候灌的」）。

        单查一行，不是把整张表读回来：`all()` 是给增量 diff 用的（那个必须全读），
        而详情页一次只看一份。doc_id 是我们自己算的 sha1，带引号直接排除掉。
        """
        if not doc_id or '"' in doc_id:
            return None
        rows = await asyncio.to_thread(
            self._client.query,
            self.collection,
            filter=f'doc_id == "{doc_id}"',
            output_fields=_FIELDS,
            limit=1,
            **STRONG,
        )
        return _to_record(rows[0]) if rows else None

    def _upsert_sync(self, records: list[DocRecord]) -> int:
        # placeholder 是占位向量（见 schema.build_registry_schema），固定 [0.0, 0.0]
        rows = [{**r.model_dump(), "placeholder": [0.0, 0.0]} for r in records]
        self._client.upsert(self.collection, data=rows)
        # 与块表同一条硬约束：不 flush 的话紧接着的查询看不到这批数据
        self._client.flush(self.collection)
        return len(records)

    async def delete(self, doc_ids: list[str]) -> None:
        """删注册行。

        **它不会删块** —— 那一步在块表上，两者不是原子的（见设计稿 §6）。
        调用方要先删块再删注册行：顺序反了、中途失败会留下「有块没注册」的孤儿，
        而反过来只会留下一份「注册在、块没了」的记录，下次跑会重做，是安全的方向。
        """
        if not doc_ids:
            return
        await asyncio.to_thread(self._delete_sync, doc_ids)

    def _delete_sync(self, doc_ids: list[str]) -> None:
        quoted = ", ".join(f'"{doc_id}"' for doc_id in doc_ids)
        self._client.delete(self.collection, filter=f"doc_id in [{quoted}]")
        self._client.flush(self.collection)


def _to_record(row: dict[str, Any]) -> DocRecord:
    return DocRecord(
        doc_id=row["doc_id"],
        source_uri=row.get("source_uri") or "",
        content_hash=row.get("content_hash") or "",
        chunk_count=int(row.get("chunk_count") or 0),
        ingested_at=int(row.get("ingested_at") or 0),
        pipeline_version=row.get("pipeline_version") or "",
    )
