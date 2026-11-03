"""HyDE 双路检索与开关行为。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rag.config.schema import FusionConfig
from rag.contracts import ChannelStats, Chunk, ScoredChunk
from rag.retrieval.hyde import HyDERetriever, configured_hyde


def _hit(chunk_id: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=chunk_id, doc_id=chunk_id, text=f"{chunk_id} 正文"),
        score=score,
        channel="milvus",
        rank=0,
    )


class _Retriever:
    name = "raw"

    def __init__(self, hits: list[ScoredChunk]) -> None:
        self.hits = hits
        self.calls: list[dict[str, Any]] = []
        self.last_stats = [
            ChannelStats(channel="milvus", candidates=len(hits), kept=len(hits), latency_ms=1)
        ]

    async def retrieve(self, query: str, top_k: int, filters=None, image=None):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters, "image": image})
        return self.hits


class _Reranker:
    name = "test-reranker"

    def __init__(self) -> None:
        self.calls = 0

    async def rerank(self, query: str, hits: list[ScoredChunk], top_n: int):
        self.calls += 1
        return list(reversed(hits))[:top_n]


def test_hyde_uses_original_and_hypothetical_queries_and_reranks_once() -> None:
    original = _Retriever([_hit("a", 0.9), _hit("shared", 0.8)])
    hypothetical = _Retriever([_hit("shared", 0.9), _hit("b", 0.8)])
    reranker = _Reranker()
    calls: list[str] = []

    async def generate(question: str) -> str:
        calls.append(question)
        return "假设资料段落"

    retriever = HyDERetriever(
        original,
        hypothetical,
        reranker,
        FusionConfig(candidates_per_channel=10),
        generate=generate,
    )
    hits = asyncio.run(retriever.retrieve("用户问题", 5))

    assert calls == ["用户问题"]
    assert original.calls[0]["query"] == "用户问题"
    assert hypothetical.calls[0]["query"] == "假设资料段落"
    assert reranker.calls == 1
    assert {hit.chunk.chunk_id for hit in hits} == {"a", "b", "shared"}
    assert any(stat.channel == "hyde-fusion" for stat in retriever.last_stats)


def test_hyde_skips_generation_for_an_image_query() -> None:
    original = _Retriever([_hit("image", 0.9)])
    hypothetical = _Retriever([])
    called = False

    async def generate(question: str) -> str:
        nonlocal called
        called = True
        return "不应生成"

    retriever = HyDERetriever(
        original,
        hypothetical,
        None,
        FusionConfig(),
        generate=generate,
    )
    hits = asyncio.run(retriever.retrieve("", 5, image=Path("query.png")))

    assert not called
    assert [hit.chunk.chunk_id for hit in hits] == ["image"]
    assert hypothetical.calls == []


def test_hyde_generation_failure_falls_back_to_original_query() -> None:
    original = _Retriever([_hit("a", 0.9)])
    hypothetical = _Retriever([])

    async def generate(question: str) -> str:
        raise RuntimeError("LLM down")

    retriever = HyDERetriever(
        original,
        hypothetical,
        None,
        FusionConfig(),
        generate=generate,
    )
    hits = asyncio.run(retriever.retrieve("用户问题", 5))

    assert [hit.chunk.chunk_id for hit in hits] == ["a"]
    assert hypothetical.calls == []
    assert any(stat.channel == "hyde" and stat.error for stat in retriever.last_stats)


@pytest.mark.parametrize("mode", ["ok", "error", "empty", "leg_error", "image"])
@pytest.mark.parametrize("rerank_mode", ["ok", "error", "empty", "disabled"])
def test_hyde_atomic_status_on_every_branch(mode, rerank_mode) -> None:
    original = _Retriever([_hit("a", 0.03).model_copy(update={"score_kind": "rrf"})])
    hypothetical = _Retriever([_hit("b", 0.02)])

    async def broken(*args, **kwargs):
        raise RuntimeError("故障注入")

    if mode == "leg_error":
        hypothetical.retrieve = broken

    async def generate(question):
        if mode == "error":
            raise RuntimeError("生成失败")
        return "" if mode == "empty" else "假设"

    class Reranker(_Reranker):
        async def rerank(self, query, hits, top_n):
            self.calls += 1
            if rerank_mode == "error":
                raise RuntimeError("重排失败")
            return [] if rerank_mode == "empty" else hits[:top_n]

    reranker = Reranker()
    retriever = HyDERetriever(
        original,
        hypothetical,
        None if rerank_mode == "disabled" else reranker,
        FusionConfig(),
        generate=generate,
    )
    result = asyncio.run(
        retriever.retrieve_result(
            "问题",
            5,
            image=Path("q.png") if mode == "image" else None,
        )
    )
    assert len(original.calls) == 1
    assert reranker.calls == (0 if rerank_mode == "disabled" else 1)
    assert (
        result.rerank_status
        == {
            "disabled": "disabled",
            "ok": "succeeded",
            "error": "failed",
            "empty": "failed",
        }[rerank_mode]
    )
    assert result.score_kind == ("rerank" if rerank_mode == "ok" else "rrf")
    codes = [item.code for item in result.degradations]
    if mode in {"error", "leg_error"}:
        assert "hyde_failed" in codes
    if mode == "empty":
        assert "hyde_empty" in codes
    if rerank_mode in {"error", "empty"}:
        assert ("rerank_failed" if rerank_mode == "error" else "rerank_empty") in codes


@pytest.mark.parametrize("query", ["", "问题"])
def test_hyde_skipped_rerank_preserves_atomic_stats(query) -> None:
    from rag.contracts import RetrievalResult

    class Atomic(_Retriever):
        async def retrieve_result(self, query, top_k, **kwargs):
            await self.retrieve(query, top_k, **kwargs)
            return RetrievalResult(query=query, chunks=self.hits, stats=self.last_stats)

    hits = [] if query else [_hit("a", 0.03).model_copy(update={"score_kind": "rrf"})]
    original = Atomic(hits)
    hypothetical = Atomic([])
    reranker = _Reranker()
    generated = []

    async def generate(question):
        generated.append(question)
        return "假设"

    retriever = HyDERetriever(original, hypothetical, reranker, FusionConfig(), generate=generate)
    result = asyncio.run(retriever.retrieve_result(query, 5))
    assert result.rerank_status == "skipped"
    assert result.score_kind == ("unknown" if query else "rrf")
    assert reranker.calls == 0
    assert len(original.calls) == 1
    assert generated == ([query] if query else [])
    assert result.stats == retriever.last_stats
    assert result.stats[0].channel == "original:milvus"


def test_hyde_concurrent_queries_keep_degradations_and_stats_isolated() -> None:
    from rag.contracts import RetrievalDegradation, RetrievalResult

    class Atomic(_Retriever):
        @property
        def last_stats(self):
            raise AssertionError("新链路不读取共享状态")

        @last_stats.setter
        def last_stats(self, value):
            pass

        async def retrieve_result(self, query, top_k, **kwargs):
            await asyncio.sleep(0)
            return RetrievalResult(
                query=query,
                chunks=[_hit(query, 0.03).model_copy(update={"score_kind": "rrf"})],
                stats=[ChannelStats(channel=query, candidates=1, kept=1)],
                degradations=[
                    RetrievalDegradation(stage="source", code="channel_failed", message="注入")
                ]
                if query == "bad"
                else [],
            )

    async def generate(query):
        await asyncio.sleep(0)
        return "" if query == "bad" else "假设"

    retriever = HyDERetriever(Atomic([]), Atomic([]), None, FusionConfig(), generate=generate)

    async def run():
        return await asyncio.gather(
            retriever.retrieve_result("bad", 5), retriever.retrieve_result("ok", 5)
        )

    bad, ok = asyncio.run(run())
    assert [item.code for item in bad.degradations] == ["channel_failed", "hyde_empty"]
    assert not ok.degradations
    assert bad.stats[0].channel == "original:bad"
    assert ok.stats[0].channel == "original:ok"
    assert bad.rerank_status == ok.rerank_status == "disabled"


def test_configured_hyde_is_none_when_switched_off() -> None:
    settings = SimpleNamespace(
        retrieval=SimpleNamespace(transform=SimpleNamespace(hyde=False, hyde_max_chars=600))
    )

    assert configured_hyde(settings, object()) is None


def test_retriever_rejects_hyde_enabled_without_generator() -> None:
    from rag.config.schema import Settings
    from rag.indexing import build_index
    from rag.retrieval import build_retriever

    settings = Settings()
    settings.retrieval.transform.hyde = True
    settings.retrieval.rerank.enabled = False

    with pytest.raises(ValueError, match="HyDE"):
        build_retriever(settings, build_index(settings), object())
