"""Milvus 索引实现：schema.py 管表结构，client.py 管块读写，registry.py 管文档注册表。"""

from __future__ import annotations

from rag.indexing.milvus.client import MilvusIndex
from rag.indexing.milvus.registry import DocRegistry
from rag.indexing.milvus.schema import (
    build_index_params,
    build_registry_index_params,
    build_registry_schema,
    build_schema,
    registry_collection,
)

__all__ = [
    "DocRegistry",
    "MilvusIndex",
    "build_index_params",
    "build_registry_index_params",
    "build_registry_schema",
    "build_schema",
    "registry_collection",
]
