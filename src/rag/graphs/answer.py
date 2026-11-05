"""统一普通问答、QueryGraph 与流式问答的答案编排。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from rag.config.schema import Settings
from rag.contracts import (
    AssembledContext,
    ChannelStats,
    Citation,
    RerankStatus,
    RetrievalDegradation,
    RetrievalResult,
    ScoredChunk,
    ScoreKind,
)
from rag.graphs.query import build_query_preflight, run_query_preflight
from rag.graphs.verification import AnswerVerifier
from rag.graphs.wiring import build_json_caller


class PreparedAnswer(BaseModel):
    """生成前已经固定的检索上下文；普通与流式生成从这里分叉。"""

    mode: str
    hits: list[ScoredChunk]
    stats: list[ChannelStats]
    context: AssembledContext
    retrieval: RetrievalResult
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
    verification: dict[str, Any] | None = None
    score_kind: ScoreKind = "unknown"
    rerank_status: RerankStatus = "unknown"
    degradations: list[RetrievalDegradation] = Field(default_factory=list)


class AnswerEngine:
    """检索预检与答案生成的唯一接线点。

    QueryGraph 只负责生成前的规划、检索、评分、改写和装配。非流式与流式生成
    都从同一个 `PreparedAnswer` 出发，避免 `stream=true` 绕过检索纠错。
    """

    def __init__(
        self,
        settings: Settings,
        index: Any,
        heavy: Any,
        *,
        verifier: AnswerVerifier | None = None,
    ) -> None:
        self.settings = settings
        self.index = index
        self.heavy = heavy
        self.verifier = verifier

    async def prepare(
        self,
        question: str,
        top_k: int,
        *,
        filters: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> PreparedAnswer:
        if not self.settings.graph.enabled:
            retrieval = await self._retrieve(question, top_k, filters=filters)
            return PreparedAnswer(
                mode="plain",
                hits=retrieval.chunks,
                stats=retrieval.stats,
                context=self._assemble(retrieval),
                retrieval=retrieval,
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
            retrieval=result["retrieval"],
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
        verification = await self._verify(question, answer, prepared)
        return AnswerRun(
            answer=answer,
            hits=prepared.hits,
            stats=prepared.stats,
            citations=prepared.context.citations,
            rounds=prepared.rounds,
            grade=prepared.grade,
            mode=prepared.mode,
            usage=usage or None,
            verification=verification,
            score_kind=prepared.retrieval.score_kind,
            rerank_status=prepared.retrieval.rerank_status,
            degradations=prepared.retrieval.degradations,
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
        answer = "".join(pieces)
        verification = await self._verify(question, answer, prepared)
        if verification is not None:
            yield {"type": "verification", "verification": verification}
        yield {
            "type": "done",
            "answer": answer,
            "citations": prepared.context.citations,
            "rounds": prepared.rounds,
            "grade": prepared.grade,
            "mode": prepared.mode,
            "usage": usage or None,
            "verification": verification,
        }

    async def _verify(
        self, question: str, answer: str, prepared: PreparedAnswer
    ) -> dict[str, Any] | None:
        if self.verifier is None:
            return None
        by_id = {hit.chunk.chunk_id: hit.chunk.text for hit in prepared.hits}
        contexts = [
            by_id[item.chunk_id]
            for item in prepared.context.citations
            if item.source_type == "corpus" and item.chunk_id in by_id
        ]
        retrieval_incomplete = any(
            item.code == "channel_failed" for item in prepared.retrieval.degradations
        )
        return await self.verifier.verify(
            question, answer, contexts, retrieval_incomplete=retrieval_incomplete
        )

    async def _retrieve(
        self,
        query: str,
        top_k: int,
        *,
        filters: dict[str, Any] | None = None,
        channels: list[str] | None = None,
    ) -> RetrievalResult:
        from rag.service.retrieval import RetrievalService

        return await RetrievalService(
            self.settings,
            self.index,
            self.heavy.embedder,
            self.heavy.reranker,
            self.heavy.chat,
        ).retrieve(query, top_k, filters=filters, channels=channels)

    def _assemble(self, retrieval: RetrievalResult) -> AssembledContext:
        from rag.retrieval import assemble

        return assemble(retrieval, self.settings.retrieval.assemble.max_tokens)


def build_answer_engine(
    settings: Settings,
    index: Any,
    heavy: Any,
    *,
    verifier: AnswerVerifier | None = None,
) -> AnswerEngine:
    return AnswerEngine(settings, index, heavy, verifier=verifier)
