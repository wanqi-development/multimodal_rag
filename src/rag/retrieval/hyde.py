"""HyDE 查询变换与双路检索。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from rag.config.schema import FusionConfig
from rag.contracts import (
    ChannelStats,
    Reranker,
    RetrievalDegradation,
    RetrievalResult,
    Retriever,
    ScoredChunk,
)
from rag.retrieval.fusion import rrf_fuse
from rag.retrieval.multichannel import rerank_hits, result_from_rerank, retrieve_channel_result

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
        return (await self.retrieve_result(query, top_k, filters=filters, image=image)).chunks

    async def retrieve_result(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        image: Path | None = None,
    ) -> RetrievalResult:
        pool = max(top_k, self._fusion.candidates_per_channel)
        if image is not None or not query.strip():
            original = await retrieve_channel_result(
                self._original,
                query,
                pool,
                filters=filters,
                image=image,
            )
            return await self._finish(
                query,
                top_k,
                original.chunks,
                _prefixed(original.stats, "original"),
                original.degradations,
            )

        started = time.perf_counter()
        original_task = asyncio.create_task(
            retrieve_channel_result(self._original, query, pool, filters=filters)
        )
        code = ""
        message = ""
        try:
            hypothetical_query = await self._generate(query)
        except Exception as exc:  # noqa: BLE001 —— HyDE 只是增强，失败退回原查询
            code = "hyde_failed"
            message = f"{type(exc).__name__}: {exc}（已退回原问题检索）"
        else:
            if not hypothetical_query.strip():
                code = "hyde_empty"
                message = "HyDE 返回空文本（已退回原问题检索）"
        if code:
            original = await original_task
            return await self._fallback(query, top_k, original, started, code, message)

        hypothetical_task = asyncio.create_task(
            retrieve_channel_result(self._hypothetical, hypothetical_query, pool, filters=filters)
        )
        # 等待两腿收束；原腿故障仍抛出，假设腿故障才回退，不重复召回。
        original, hypothetical = await asyncio.gather(
            original_task,
            hypothetical_task,
            return_exceptions=True,
        )
        if isinstance(original, BaseException):
            raise original
        if isinstance(hypothetical, BaseException):
            if not isinstance(hypothetical, Exception):
                raise hypothetical
            return await self._fallback(
                query,
                top_k,
                original,
                started,
                "hyde_failed",
                f"{type(hypothetical).__name__}: {hypothetical}（已退回原问题检索）",
            )
        fused = rrf_fuse(
            [original.chunks, hypothetical.chunks],
            k=self._fusion.k,
            sources=["original", "hyde-query"],
        )
        stats = [
            *_prefixed(original.stats, "original"),
            *_prefixed(hypothetical.stats, "hyde"),
            ChannelStats(
                channel="hyde-fusion",
                candidates=len(original.chunks) + len(hypothetical.chunks),
                kept=len(fused),
                latency_ms=_ms(started),
            ),
        ]
        return await self._finish(
            query, top_k, fused, stats, [*original.degradations, *hypothetical.degradations]
        )

    async def _fallback(
        self,
        query: str,
        top_k: int,
        original: RetrievalResult,
        started: float,
        code: str,
        message: str,
    ) -> RetrievalResult:
        return await self._finish(
            query,
            top_k,
            original.chunks,
            [
                *_prefixed(original.stats, "original"),
                ChannelStats(
                    channel="hyde", candidates=0, kept=0, latency_ms=_ms(started), error=message
                ),
            ],
            [
                *original.degradations,
                RetrievalDegradation(stage="hyde", code=code, message=message),
            ],
        )

    async def _finish(
        self,
        query: str,
        top_k: int,
        hits: list[ScoredChunk],
        stats: list[ChannelStats],
        degradations: list[RetrievalDegradation],
    ) -> RetrievalResult:
        outcome = await rerank_hits(query, hits, self._reranker, top_k)
        stats = [*stats, *([outcome.stat] if outcome.stat else [])]
        self.last_stats = stats
        return result_from_rerank(query, outcome, top_k, stats, degradations)


def _prefixed(stats: list[ChannelStats], prefix: str) -> list[ChannelStats]:
    return [
        item.model_copy(update={"channel": f"{prefix}:{item.channel}"})
        for item in stats
        if item.channel != "rrf"
    ]


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
