"""图谱通道（Neo4j）：问题 → 实体 → 邻居关系 → **关系挂着的原始块**。

这一路的关键是**映射回块**：图里存的是实体与关系，但融合层只认 `ScoredChunk`（块级），
所以每条关系的 `chunk_id` 要换回正文 —— 正文在 Milvus 里（`index.get_chunks`），
图里只留 chunk_id 和一段证据。这样图通道召回的东西**能和 Milvus 那一路的候选并排比较**，
而不是两个不可比的排行榜。

打分两条规矩：

1. **跳数越近越值钱**（第 1 跳 1.0、第 2 跳 0.5、第 n 跳 1/n）；
2. **一条边的两端都被问句认出来，分值翻倍** —— 问句里同时出现两个实体时，
   「这两个实体之间的关系」才是有信息量的那条，而只连着一个实体的边只是它的邻居。

同一块被多条关系命中就取最大值（不是求和）—— 不然一个热门实体的十条边会把同一块
堆成一个虚高的分数，融合时压过所有别的通道。

**失败就抛**：Neo4j 连不上、查不动，直接让异常冒到 `MultiChannelRetriever` —— 它会把
这一路记进 `stats.error` 并当作没召回，其余通道照常（这是既定的降级路径，别在这里
自己吞掉错误：那样 `--stats` 里看不出图通道到底有没有工作）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.contracts import Retriever, ScoredChunk
from rag.indexing.milvus import MilvusIndex
from rag_channel_neo4j.store import GraphStore


class GraphRetriever:
    """一路召回：实体链接（纯子串匹配，零模型调用）+ 1..hops 邻居 + 映射回块。"""

    # **只补位**（2026-10-05 实测定的）：这一路带回来的块与 Milvus 那一路高度重叠，
    # 重复投票改不了名次（27 题名次变化 0 条），所以融合层会把与别路重复的候选丢掉，
    # 它只贡献**别人没找到的块**。见 `MultiChannelRetriever._complement_only`。
    complement_only = True

    def __init__(self, index: MilvusIndex, store: GraphStore, cfg: Any) -> None:
        self._index = index
        self._store = store
        self._cfg = cfg
        self.name = f"graph:{store.name}"

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        image: Path | None = None,
    ) -> list[ScoredChunk]:
        # 没有查询词就没有实体可认：这一路返回空是**正常**（融合层当作没召回）
        if not query.strip():
            return []
        keys = await self._store.find_entities(query, limit=self._cfg.max_entities)
        if not keys:
            return []
        edges = await self._store.neighbors(
            keys, hops=self._cfg.hops, limit=self._cfg.edges_per_entity
        )
        matched = set(keys)
        scores: dict[str, float] = {}
        for edge in edges:
            chunk_id = str(edge.get("chunk_id") or "")
            if not chunk_id:
                continue
            hops_away = 1.0 / max(1, int(edge.get("hop") or 1))
            endpoints = {str(edge.get("source") or ""), str(edge.get("neighbor") or "")}
            score = hops_away * (2.0 if len(endpoints & matched) >= 2 else 1.0)
            scores[chunk_id] = max(scores.get(chunk_id, 0.0), score)
        if not scores:
            return []

        chunks = await self._index.get_chunks(list(scores))
        hits = [
            ScoredChunk(chunk=chunk, score=scores[chunk.chunk_id], channel=self.name, rank=0)
            for chunk in chunks
        ]
        hits = [hit for hit in hits if _passes(hit, filters)]
        hits.sort(key=lambda hit: -hit.score)
        return hits[:top_k]


def _passes(hit: ScoredChunk, filters: dict[str, Any] | None) -> bool:
    """本通道的过滤在**取回块之后**做。

    别的路把过滤下推给 Milvus 的表达式，这条路的「检索」发生在 Neo4j 里，
    那边没有块的元数据。所以这里对取回来的块做同一套判断 —— 常见字段覆盖到，
    认不出的字段**不拦**（宁可多给，也别悄悄少给）。
    """
    if not filters:
        return True
    chunk = hit.chunk
    values: dict[str, Any] = {
        "doc_id": chunk.doc_id,
        "doc_type": chunk.doc_type,
        "modality": chunk.modality,
        "page": chunk.page,
        "source_uri": str(chunk.metadata.get("source_uri", "")),
    }
    for key, expected in filters.items():
        if key not in values:
            continue
        wanted = expected if isinstance(expected, list) else [expected]
        if values[key] not in wanted:
            return False
    return True


def build(*, index: MilvusIndex, settings: Any, cfg: Any, embedder: Any = None) -> Retriever:
    """**入口点指的那个工厂**（接口约定见核心的 `retrieval/plugins.py`）。

    不吃 embedder（图谱走实体链接，不做向量检索）；参数取核心的 `retrieval.graph`
    （uri / user / hops / 超时）。要开**第二路**同类源，可以把参数写在这一项的
    `options` 里（`uri` / `hops` / `max_entities`），下面按需覆盖。
    """
    from rag_channel_neo4j.store import build_graph_store

    graph_cfg = settings.retrieval.graph
    overrides = dict(cfg.options or {})
    if overrides:
        graph_cfg = graph_cfg.model_copy(deep=True)
        for key in ("uri", "user", "hops", "max_entities", "edges_per_entity", "timeout_s"):
            if key in overrides:
                setattr(graph_cfg, key, overrides[key])
    return GraphRetriever(index, build_graph_store(settings, cfg=graph_cfg), graph_cfg)
