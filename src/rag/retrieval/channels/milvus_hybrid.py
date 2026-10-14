"""Milvus 通道：dense（多模态融合向量）+ 内建 BM25，由 Milvus 服务端融合（L1）。

**语义与关键词都在这一路里**（2026-10-08 定）：关键词不再是一个独立通道。
跨通道融合（L2，把这一路与图谱/ES 等**别的数据源**的结果再合一次）不在这里，
那属于 retrieval/fusion。本类只负责一路召回。

作为**通道类型 `milvus`** 注册（见文件末尾的 `build`），装配由配置里的
`retrieval.channels` 那张表驱动。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pymilvus import RRFRanker, WeightedRanker

from rag.config.schema import ChannelConfig, HybridChannelConfig, Settings
from rag.contracts import CHANNEL_TYPES, Embedder, ScoredChunk
from rag.indexing.milvus import MilvusIndex


def build_ranker(cfg: HybridChannelConfig) -> Any:
    """按配置选融合策略。

    RRFRanker 只有 k、不支持按腿加权，所以 strategy=rrf 时权重必须都是 1.0；
    想让权重生效就换 weighted（WeightedRanker）。不静默忽略配置里的权重。
    """
    strategy = cfg.fusion.strategy
    weights = (cfg.dense.weight, cfg.sparse.weight)
    if strategy == "rrf":
        if weights != (1.0, 1.0):
            raise ValueError(
                f"fusion.strategy=rrf 不支持按腿加权（Milvus 的 RRFRanker 只能设 k）。"
                f"当前权重 {weights}：改用 weighted，或把权重都设回 1.0"
            )
        return RRFRanker(cfg.fusion.k)
    if strategy == "weighted":
        return WeightedRanker(*weights)
    raise KeyError(f"未知的融合策略: {strategy!r}；已知: rrf / weighted")


def filter_expr(filters: dict[str, Any] | None) -> str | None:
    """把 {字段: 值} 翻成 Milvus 过滤表达式。

    限制在等值 / in / like 是有意的：过滤值来自调用方，直接拼表达式等于把注入面交给上层。
    这里只接受能安全转义的类型，其余一律报错而不是放过。

    ``like`` 是给**层级检索**用的（用户问「第 2.1 节讲了什么」）：`heading_path` 是一整条
    路径字符串，等值匹配要求一字不差，而问法千变万化。写法是给字段一个单键字典：

        {"heading_path": {"like": "%2.1%"}}   →  heading_path like "%2.1%"

    通配符由调用方给（`%` 自己写），我们只负责转义 —— 不替它猜「该不该加 %」。
    """
    if not filters:
        return None
    parts: list[str] = []
    for field, value in filters.items():
        if isinstance(value, bool):  # 必须先判 bool：它是 int 的子类
            parts.append(f"{field} == {str(value).lower()}")
        elif isinstance(value, str):
            parts.append(f'{field} == "{_escape(value)}"')
        elif isinstance(value, int | float):
            parts.append(f"{field} == {value}")
        elif isinstance(value, list | tuple) and all(isinstance(v, str) for v in value):
            items = ", ".join(f'"{_escape(v)}"' for v in value)
            parts.append(f"{field} in [{items}]")
        elif isinstance(value, dict) and set(value) == {"like"}:
            # 只认 `{"like": "…"}` 这一种形态：多键字典说明调用方在表达更复杂的东西，
            # 猜它的意思不如报错 —— 过滤条件猜错会静默少召回。
            parts.append(f'{field} like "{_escape(str(value["like"]))}"')
        else:
            raise TypeError(
                f"不支持的过滤值：{field}={value!r}"
                f'（支持 str / 数字 / 字符串列表 / {{"like": "…"}}）'
            )
    return " and ".join(parts)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


class MilvusHybridRetriever:
    """一路召回通道：查询向量化后交给 Milvus 做 dense + sparse 两路召回与服务端融合。"""

    def __init__(
        self,
        index: MilvusIndex,
        embedder: Embedder,
        cfg: HybridChannelConfig,
    ) -> None:
        self._index = index
        self._embedder = embedder
        self._cfg = cfg
        self._ranker = build_ranker(cfg)
        self.name = f"hybrid:{index.name}"

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        image: Path | None = None,
    ) -> list[ScoredChunk]:
        vector = (await self._embedder.embed_query(query, image)).dense[0]
        if image is not None and not query.strip():
            # **纯图查询**：没有查询词，稀疏腿无事可做。硬走 hybrid 等于拿空串去请求
            # BM25 —— 那一路的结果要么空、要么没有意义，不如直接只走 dense。
            return await self._index.search_dense(vector, top_k, filter_expr=filter_expr(filters))
        return await self._index.search_hybrid(
            vector,
            query,
            top_k=top_k,
            dense_limit=self._cfg.dense.limit,
            sparse_limit=self._cfg.sparse.limit,
            ranker=self._ranker,
            filter_expr=filter_expr(filters),
        )


def build(
    *, index: MilvusIndex, embedder: Embedder, settings: Settings, cfg: ChannelConfig
) -> MilvusHybridRetriever:
    """**通道类型 `milvus` 的工厂**（接口约定见 `retrieval/plugins.py`）。

    参数默认取 `retrieval.hybrid`（有类型校验）；这一项的 `options` 可以覆盖两条腿的
    上限 —— 那是"同一个库再开一路"的场景（比如另建一个只走 dense 的 collection）。
    """
    hybrid = settings.retrieval.hybrid
    overrides = dict(cfg.options or {})
    if overrides:
        hybrid = hybrid.model_copy(deep=True)
        for key, leg in (("dense_limit", hybrid.dense), ("sparse_limit", hybrid.sparse)):
            if key in overrides:
                leg.limit = int(overrides[key])
    return MilvusHybridRetriever(index, embedder, hybrid)


CHANNEL_TYPES.register("milvus")(build)
