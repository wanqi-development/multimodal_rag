"""多路召回 + 客户端融合：`hybrid` 与可选通道（bm25 / 以后的 graph、web）并成一张榜。

三件事在这里发生：

1. **并发跑各通道**（``asyncio.gather``）。串行跑没有理由 —— 它们之间没有依赖，
   总耗时按最慢那一路算而不是求和（真机上 hybrid 那一路本身就含一次线上 embedding）。
2. **单路失败不拖垮整个查询**。一路挂了（Milvus 抖动、以后的 web 超时）就把它记进 stats
   并当作没召回，其余照常出结果 —— 与 ``ingest/images.py`` 里「一张图失败不牵连整份文档」
   同一个立场。但**全挂就是真失败**：那种情况返回空列表会让下游拿不到上下文还照常生成，
   反而更难查，所以原样把第一个异常抛出去。
3. **融合 + 重排 + 记账**。融合用 RRF（``retrieval/fusion.py``），重排用 ``Reranker``
   （融合**之后**才轮到它 —— 它要看到所有通道的候选）；stats 用早就定义好、一直没用过的
   ``ChannelStats``：每路召回多少条、进了最终榜几条、耗时多少、有没有报错。

这一层实现的是同一个 ``Retriever`` 协议，所以 CLI 与以后的编排层都不用改。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.config.schema import FusionConfig
from rag.contracts import ChannelStats, Retriever, ScoredChunk
from rag.contracts.protocols import Reranker
from rag.obs import record_channels
from rag.retrieval.fusion import rrf_fuse


class RetrievalError(RuntimeError):
    """所有通道都没能返回结果 —— 这时候没有「部分可用」可言。"""


@dataclass
class _Outcome:
    """一路通道这次的结果（`dropped` 只有「只补位」的通道会非零）。"""

    channel: Retriever
    hits: list[ScoredChunk]
    error: Exception | None = None
    latency_ms: float = 0.0
    dropped: int = 0


class MultiChannelRetriever:
    """把若干路通道并成一路。``last_stats`` 是上一次检索的账本（``--stats`` 打印它）。"""

    def __init__(
        self,
        channels: list[Retriever],
        fusion: FusionConfig,
        reranker: Reranker | None = None,
        *,
        limits: dict[str, int] | None = None,
    ) -> None:
        if not channels:
            raise ValueError("MultiChannelRetriever 至少要有一路通道")
        self._channels = channels
        self._fusion = fusion
        # 每一路自己的候选数（配置里 `channels[].top_k`）；没给就全通道一样多。
        # **默认必须一样多**：给得多那一侧在融合里票更多（实测 hybrid 5 条 vs bm25 20 条）。
        self._limits = dict(limits or {})
        # None = 不重排。调用方（build_retriever）已经按 retrieval.rerank.enabled 判断过了
        self._reranker = reranker
        self.last_stats: list[ChannelStats] = []
        parts = [channel.name for channel in channels]
        if reranker is not None:
            parts.append(reranker.name)
        self.name = "multi[" + ",".join(parts) + "]"

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        image: Path | None = None,
    ) -> list[ScoredChunk]:
        # **每路要一样多**：候选数不对等时，给得多那一侧在融合里票更多（实测：hybrid 5 条
        # vs bm25 20 条）。池子同时留给下一层的 rerank —— 精排得有得挑才有意义。
        pool = max(top_k, self._fusion.candidates_per_channel)
        outcomes = await asyncio.gather(
            *(
                self._one(channel, query, self._pool_for(channel, pool), filters, image)
                for channel in self._channels
            )
        )
        failed = [item.error for item in outcomes if item.error is not None]
        if len(failed) == len(outcomes):
            # 全挂了：把第一个异常原样抛出去。返回空列表会让上层以为「就是没命中」
            raise RetrievalError(f"{len(outcomes)} 路通道全部失败：{failed[0]}") from failed[0]

        outcomes = self._complement_only(list(outcomes))
        fused = rrf_fuse(
            [item.hits for item in outcomes],
            k=self._fusion.k,
            sources=[channel.name for channel in self._channels],
        )
        reranked, rerank_stat = await self._rerank(query, fused, top_k)
        self.last_stats = self._stats(outcomes, reranked[:top_k], rerank_stat)
        # **通道日志**（三类日志里的第三类）在这里落：这是唯一同时握着「每一路账本」
        # 和「最终结果」的地方，放到调用点去记一定会漏（调用点有五个）。
        # 落到 channel.log（挂在自己的 logger 上，见 obs/setup.py）。
        record_channels(
            query=query,
            channels=[channel.name for channel in self._channels],
            stats=self.last_stats,
            rerank=self._reranker is not None,
            hits=reranked[:top_k],
            filters=filters,
        )
        return reranked[:top_k]

    def _pool_for(self, channel: Retriever, default: int) -> int:
        """这一路要多少候选。配置里给了 `channels[].top_k` 就用它，否则用统一值。

        为什么要能按路给：某一路是"只补位"的重活（图谱每次多花 300-400ms），给它少一点
        候选是划算的；但**默认必须一样多** —— 给得多那一侧票更多（见 `_limits` 的注释）。
        """
        return max(1, int(self._limits.get(channel.name) or default))

    @staticmethod
    def _complement_only(outcomes: list[_Outcome]) -> list[_Outcome]:
        """「只补位」的通道：把它与别路**重复**的候选丢掉，只贡献新块。

        为什么要有这条：实测（2026-10-05）graph 通道上线后，27 条黄金题的名次变化 **0 条** ——
        它带回来的块和 hybrid 命中的是同一批，只是再投一票；RRF 的票差不足以改变顺序，
        而每次查询还要多花 ~400ms。同一批块重复投票没有信息量，**只有新块才有**。

        谁算「只补位」由通道自己声明（`complement_only = True`）：这是它的固有性质，
        不是调用方的策略。补位通道之间也互相去重（先到的先占），顺序与 `self._channels` 一致。
        """
        taken: set[str] = set()
        for item in outcomes:
            if item.error is None and not getattr(item.channel, "complement_only", False):
                taken.update(hit.chunk.chunk_id for hit in item.hits)
        result: list[_Outcome] = []
        for item in outcomes:
            if item.error is not None or not getattr(item.channel, "complement_only", False):
                result.append(item)
                continue
            kept = [hit for hit in item.hits if hit.chunk.chunk_id not in taken]
            taken.update(hit.chunk.chunk_id for hit in kept)
            result.append(
                _Outcome(
                    channel=item.channel,
                    hits=kept,
                    error=item.error,
                    latency_ms=item.latency_ms,
                    dropped=len(item.hits) - len(kept),
                )
            )
        return result

    async def _rerank(
        self, query: str, fused: list[ScoredChunk], top_n: int
    ) -> tuple[list[ScoredChunk], ChannelStats | None]:
        """重排；**失败就退回融合顺序**，并把这笔账记下来。

        为什么不让它抛：重排只是精排，融合结果本身是可用的。模型下不下来、ONNX 加载失败
        这类问题不该让整个查询失败 —— 但也不能装作没事，所以记进 stats（``--stats`` 会打出来）。
        """
        if self._reranker is None or not fused:
            return fused, None
        if not query.strip():
            # **纯图查询**：交叉编码器靠「查询词 ↔ 正文」算相关性，没有词它只是在白花
            # 几秒（还可能把融合顺序打乱）。这时候 dense 的名次就是最好的答案，跳过。
            return fused, ChannelStats(
                channel=self._reranker.name,
                candidates=len(fused),
                kept=len(fused),
                latency_ms=0.0,
                error="纯图查询：重排需要查询词，已跳过",
            )
        started = time.perf_counter()
        try:
            reranked = await self._reranker.rerank(query, fused, top_n)
        except Exception as exc:  # noqa: BLE001  —— 见上面的理由
            return fused, ChannelStats(
                channel=self._reranker.name,
                candidates=len(fused),
                kept=0,
                latency_ms=_ms(started),
                error=f"{type(exc).__name__}: {exc}（已退回融合顺序）",
            )
        if not reranked:  # 空结果也算失败：别把好好的融合结果换成一片空白
            return fused, ChannelStats(
                channel=self._reranker.name,
                candidates=len(fused),
                kept=0,
                latency_ms=_ms(started),
                error="重排返回空结果（已退回融合顺序）",
            )
        return reranked, ChannelStats(
            channel=self._reranker.name,
            candidates=len(fused),
            kept=len(reranked),
            latency_ms=_ms(started),
        )

    async def _one(
        self,
        channel: Retriever,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
        image: Path | None,
    ) -> _Outcome:
        started = time.perf_counter()
        try:
            hits = await channel.retrieve(query, top_k, filters=filters, image=image)
        except Exception as exc:  # noqa: BLE001  —— 单路的问题（含实现自己的意外）都不该让整次查询失败
            return _Outcome(channel=channel, hits=[], error=exc, latency_ms=_ms(started))
        return _Outcome(channel=channel, hits=hits, latency_ms=_ms(started))

    def _stats(
        self,
        outcomes: list[_Outcome],
        fused: list[ScoredChunk],
        rerank_stat: ChannelStats | None = None,
    ) -> list[ChannelStats]:
        """每路一行 + 融合自己一行（+ 重排一行）。``kept`` 是「最终结果里有几条来自这一路」。"""
        stats = [
            ChannelStats(
                channel=item.channel.name,
                # 召回数按**原始**算（补位丢掉的也算它召回过），补位丢弃单记一笔
                candidates=len(item.hits) + item.dropped,
                kept=sum(1 for hit in fused if item.channel.name in hit.sources),
                latency_ms=item.latency_ms,
                error=None if item.error is None else f"{type(item.error).__name__}: {item.error}",
                complement_dropped=item.dropped,
            )
            for item in outcomes
        ]
        stats.append(
            ChannelStats(
                channel="rrf",
                candidates=sum(len(item.hits) + item.dropped for item in outcomes),
                kept=len(fused),
                latency_ms=None,
            )
        )
        if rerank_stat is not None:
            stats.append(rerank_stat)
        return stats


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
