"""统一普通问答、QueryGraph 与流式问答的答案编排。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from rag.config.schema import Settings
from rag.contracts import AssembledContext, ChannelStats, Citation, ScoredChunk
from rag.graphs.query import build_query_preflight, run_query_preflight
from rag.graphs.wiring import build_json_caller


class PreparedAnswer(BaseModel):
    """生成前已经固定的检索上下文；普通与流式生成从这里分叉。"""

    mode: str
    hits: list[ScoredChunk]
    stats: list[ChannelStats]
    context: AssembledContext
    rounds: int = 1
    grade: dict[str, Any] = Field(default_factory=dict)


class AnswerRun(BaseModel):
    """一次非流式答案的完整结果。"""

    answer: str
    hits: list[ScoredChunk]
    stats: list[ChannelStats]
    citations: list[Citation]
    rounds: int = 1
    grade: dict[str, Any] = Field(default_factory=dict)
    mode: str
    usage: dict[str, Any] | None = None


class AnswerEngine:
    """检索预检与答案生成的唯一接线点。

    QueryGraph 只负责生成前的规划、检索、评分、改写和装配。非流式与流式生成
    都从同一个 `PreparedAnswer` 出发，避免 `stream=true` 绕过检索纠错。
    """

    def __init__(self, settings: Settings, index: Any, heavy: Any) -> None:
        self.settings = settings
        self.index = index
        self.heavy = heavy

    async def prepare(
        self,
        question: str,
        top_k: int,
        *,
        filters: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> PreparedAnswer:
        if not self.settings.graph.enabled:
            hits, stats = await self._retrieve(question, top_k, filters=filters)
            return PreparedAnswer(
                mode="plain",
                hits=hits,
                stats=stats,
                context=self._assemble(hits),
            )

        caller = build_json_caller(self.heavy, usage=usage)
        channels = [item.name for item in self.settings.retrieval.channels if item.enabled]
        graph = build_query_preflight(
            self.settings,
            retrieve=self._retrieve,
            plan_llm=caller,
            grade_llm=caller,
            channels=channels,
        )
        result = await run_query_preflight(graph, question, top_k=top_k)
        return PreparedAnswer(
            mode="graph",
            hits=result["hits"],
            stats=result["stats"],
            context=result["context"],
            rounds=result["rounds"],
            grade=result["grade"],
        )

    async def answer(
        self,
        question: str,
        top_k: int,
        *,
        filters: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> AnswerRun:
        usage = usage if usage is not None else {}
        prepared = await self.prepare(question, top_k, filters=filters, usage=usage)
        answer = await self.heavy.generator.generate(
            question,
            prepared.context,
            usage=usage,
        )
        return AnswerRun(
            answer=answer,
            hits=prepared.hits,
            stats=prepared.stats,
            citations=prepared.context.citations,
            rounds=prepared.rounds,
            grade=prepared.grade,
            mode=prepared.mode,
            usage=usage or None,
        )

    async def stream(
        self,
        question: str,
        top_k: int,
        *,
        filters: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        usage: dict[str, Any] = {}
        prepared = await self.prepare(question, top_k, filters=filters, usage=usage)
        yield {
            "type": "retrieval",
            "hits": prepared.hits,
            "stats": prepared.stats,
            "rounds": prepared.rounds,
            "grade": prepared.grade,
            "mode": prepared.mode,
        }

        pieces: list[str] = []
        async for piece in self.heavy.generator.astream(
            question,
            prepared.context,
            usage=usage,
        ):
            pieces.append(piece)
            yield {"type": "token", "text": piece}
        yield {
            "type": "done",
            "answer": "".join(pieces),
            "citations": prepared.context.citations,
            "rounds": prepared.rounds,
            "grade": prepared.grade,
            "mode": prepared.mode,
            "usage": usage or None,
        }

    async def _retrieve(
        self,
        query: str,
        top_k: int,
        *,
        filters: dict[str, Any] | None = None,
        channels: list[str] | None = None,
    ) -> tuple[list[ScoredChunk], list[ChannelStats]]:
        from rag.retrieval import build_retriever

        local = self.settings.model_copy(deep=True)
        if channels:
            wanted = set(channels)
            for item in local.retrieval.channels:
                item.enabled = item.name in wanted
        retriever = build_retriever(
            local,
            self.index,
            self.heavy.embedder,
            self.heavy.reranker,
        )
        hits = await retriever.retrieve(query, top_k, filters=filters)
        return hits, getattr(retriever, "last_stats", [])

    def _assemble(self, hits: list[ScoredChunk]) -> AssembledContext:
        from rag.retrieval import assemble

        return assemble(hits, self.settings.retrieval.assemble.max_tokens)


def build_answer_engine(settings: Settings, index: Any, heavy: Any) -> AnswerEngine:
    return AnswerEngine(settings, index, heavy)
