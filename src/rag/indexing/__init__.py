"""索引层：Chunk + 向量 → 可检索的存储。

向量库维度写死在表结构里，而维度只有一个来源（providers.embedding.dim），
所以装配时要把两者一起交给实现 —— 这是本层唯一需要跨段配置的地方。
"""

from __future__ import annotations

from typing import Any

from rag.config.schema import Settings
from rag.contracts import Index
from rag.indexing.milvus import MilvusIndex
from rag.indexing.milvus.versioned import DualVersionMilvusIndex

__all__ = [
    "MilvusIndex",
    "DualVersionMilvusIndex",
    "build_index",
    "build_registry",
]


def build_index(settings: Settings) -> Index:
    """按 indexing.vector.backend 选实现。

    这里没做成注册表：目前向量库只有一个后端（父块存储、图存储是 architecture
    里并列的独立模块，不是它的替代实现）。等真出现第二个后端再抽注册表。
    """
    cfg = settings.indexing.vector
    if cfg.backend != "milvus":
        raise KeyError(f"未实现的向量库后端: {cfg.backend!r}；目前只有 'milvus'")
    if cfg.versions_enabled:
        return DualVersionMilvusIndex(
            cfg,
            settings.providers.embedding.dim,
            settings=settings,
        )
    return MilvusIndex(cfg, dim=settings.providers.embedding.dim)


def build_registry(settings: Settings) -> Any:
    """文档注册表（每篇一行：走到哪一步 + 入库信息）。

    **2026-10-09 起落数据库**（PG / SQLite，见 `persistence/documents.py`），不再进 Milvus：
    文档列表/详情要的只是"这份到哪一步了"，为它去问向量库不值 —— Milvus 一抖，
    页面就白等它那 9 秒的超时。块与向量仍然在 Milvus，注册表只记账。

    名字保留 `build_registry`：`commit_chunks`、CLI、agent 工具、文档接口四处调用点
    都按这个名字取的，换的只是它背后落在哪。
    """
    from rag.persistence.documents import build_document_store

    return build_document_store(settings)
