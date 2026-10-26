"""**内置**召回通道：只有 `milvus` 这一路。

**通道 = 数据源**（2026-10-08 定）：语义、关键词、图谱各自是**不同数据源**的结果，
不是"同一个库换几种搜法"。所以：

* `milvus` —— 多模态融合向量 + 内建 BM25，两腿由 Milvus 服务端融合（L1）。
  它就是**语义 + 关键词**那一路：原来的 `milvus_bm25` 独立通道已经删掉
  （实测关掉它 recall@8 一样、MRR 反而略好，见 dev.yaml 的注释）。
* **图谱（Neo4j）不在这里** —— 它被搬成了插件包 `plugins/rag-channel-neo4j`
  （`type: neo4j`，入口点组名 `rag.channels`）。核心只认识"按 type 找工厂"这套机制，
  不认识 Neo4j 本身；装了那个包配置里才能写 `type: neo4j`（见 `retrieval/plugins.py`）。

在核心里加一路（不推荐，通常该做成插件）= 写一个满足 Retriever 的类 + 在这里 import
一行触发注册。
"""

from __future__ import annotations

from rag.retrieval.channels.milvus_hybrid import (
    MilvusHybridRetriever,
    build_ranker,
    filter_expr,
)

__all__ = ["MilvusHybridRetriever", "build_ranker", "filter_expr"]
