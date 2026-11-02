"""HyDE 查询变换与双路检索。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from rag.config.schema import FusionConfig
from rag.contracts import ChannelStats, Reranker, Retriever, ScoredChunk
from rag.retrieval.fusion import rrf_fuse
from rag.retrieval.multichannel import rerank_hits

HydeFn = Callable[[str], Awaitable[str]]

HYDE_PROMPT = """请根据用户问题写一段“可能出现在相关技术文档中的资料段落”，供向量检索使用。

要求：
1. 只写资料正文，300～600 字；不要回答用户，不要称呼用户，不要加解释。
2. 保留问题里的专有名词、编号、配置名和代码符号；不确定的事实不要编造具体数字。
3. 不要输出引用编号、Markdown 围栏或“根据资料”等聊天口吻。

用户问题：{question}"""


def build_hyde(chat: Any, *, max_chars: int = 600) -> HydeFn:
    """用当前 chat 模型生成 HyDE 假设资料。"""
    if chat is None:
        raise ValueError("RAG_HYDE 已启用，但 chat provider 没有建起来")
    from langchain_core.messages import HumanMessage, SystemMessage

    system = SystemMessage("你是技术检索查询生成器，只输出用于检索的资料正文。")

    async def generate(question: str) -> str:
        response = await chat.ainvoke(
            [system, HumanMessage(HYDE_PROMPT.format(question=question.strip()))]
        )
        content = response.content
        text = content if isinstance(content, str) else str(content)
        return " ".join(text.split())[:max_chars]

    return generate


def configured_hyde(settings: Any, chat: Any) -> HydeFn | None:
    cfg = settings.retrieval.transform
    if not cfg.hyde:
        return None
    return build_hyde(chat, max_chars=cfg.hyde_max_chars)


class HyDERetriever:
    """原问题与 HyDE 假设资料双路召回，RRF 融合后用原问题重排一次。"""

    def __init__(
        self,
        original: Retriever,
        hypothetical: Retriever,
        reranker: Reranker | None,
        fusion: FusionConfig,
        *,
        generate: HydeFn,
    ) -> None:
        self._original = original
        self._hypothetical = hypothetical
        self._reranker = reranker
        self._fusion = fusion
        self._generate = generate
        self.last_stats: list[ChannelStats] = []
        self.name = "hyde"

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        image: Path | None = None,
    ) -> list[ScoredChunk]:
        if image is not None or not query.strip():
            return await self._plain(query, top_k, filters=filters, image=image)

        pool = max(top_k, self._fusion.candidates_per_channel)
        started = time.perf_counter()
        original_task = asyncio.create_task(self._original.retrieve(query, pool, filters=filters))
        try:
            hypothetical_query = await self._generate(query)
        except Exception as exc:  # noqa: BLE001 —— HyDE 只是增强，失败退回原查询
            hits = await original_task
            reranked, rerank_stat = await rerank_hits(query, hits, self._reranker, top_k)
            self.last_stats = [
                *_prefixed(self._original.last_stats, "original"),
                ChannelStats(
                    channel="hyde",
                    candidates=0,
                    kept=0,
                    latency_ms=_ms(started),
                    error=f"{type(exc).__name__}: {exc}（已退回原问题检索）",
                ),
                *([rerank_stat] if rerank_stat else []),
            ]
            return reranked[:top_k]

        if not hypothetical_query.strip():
            hits = await original_task
            reranked, rerank_stat = await rerank_hits(query, hits, self._reranker, top_k)
            self.last_stats = [
                *_prefixed(self._original.last_stats, "original"),
                ChannelStats(
                    channel="hyde",
                    candidates=0,
                    kept=0,
                    latency_ms=_ms(started),
                    error="HyDE 返回空文本（已退回原问题检索）",
                ),
                *([rerank_stat] if rerank_stat else []),
            ]
            return reranked[:top_k]

        hypothetical_task = asyncio.create_task(
            self._hypothetical.retrieve(hypothetical_query, pool, filters=filters)
        )
        try:
            original_hits, hypothetical_hits = await asyncio.gather(
                original_task, hypothetical_task
            )
        except Exception as exc:  # noqa: BLE001 —— 假设腿失败时保留原查询结果
            if original_task.done() and not original_task.cancelled():
                original_hits = original_task.result()
            else:
                original_hits = await self._original.retrieve(query, pool, filters=filters)
            reranked, rerank_stat = await rerank_hits(query, original_hits, self._reranker, top_k)
            self.last_stats = [
                *_prefixed(self._original.last_stats, "original"),
                ChannelStats(
                    channel="hyde",
                    candidates=0,
                    kept=0,
                    latency_ms=_ms(started),
                    error=f"{type(exc).__name__}: {exc}（已退回原问题检索）",
                ),
                *([rerank_stat] if rerank_stat else []),
            ]
            return reranked[:top_k]

        fused = rrf_fuse(
            [original_hits, hypothetical_hits],
            k=self._fusion.k,
            sources=["original", "hyde-query"],
        )
        reranked, rerank_stat = await rerank_hits(query, fused, self._reranker, top_k)
        self.last_stats = [
            *_prefixed(self._original.last_stats, "original"),
            *_prefixed(self._hypothetical.last_stats, "hyde"),
            ChannelStats(
                channel="hyde-fusion",
                candidates=len(original_hits) + len(hypothetical_hits),
                kept=len(fused),
                latency_ms=_ms(started),
            ),
            *([rerank_stat] if rerank_stat else []),
        ]
        return reranked[:top_k]

    async def _plain(
        self,
        query: str,
        top_k: int,
        *,
        filters: dict[str, Any] | None,
        image: Path | None,
    ) -> list[ScoredChunk]:
        pool = max(top_k, self._fusion.candidates_per_channel)
        hits = await self._original.retrieve(query, pool, filters=filters, image=image)
        reranked, rerank_stat = await rerank_hits(query, hits, self._reranker, top_k)
        self.last_stats = [
            *_prefixed(self._original.last_stats, "original"),
            *([rerank_stat] if rerank_stat else []),
        ]
        return reranked[:top_k]


def _prefixed(stats: list[ChannelStats], prefix: str) -> list[ChannelStats]:
    return [
        item.model_copy(update={"channel": f"{prefix}:{item.channel}"})
        for item in stats
        if item.channel != "rrf"
    ]


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
