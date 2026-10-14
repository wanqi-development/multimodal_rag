"""多路召回：并发、融合、记账，以及「单路挂了不拖垮整次查询」。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rag.config.schema import FusionConfig
from rag.contracts import Chunk, ScoredChunk
from rag.providers.rerank import RerankError
from rag.retrieval.multichannel import MultiChannelRetriever, RetrievalError


def _hit(chunk_id: str, *, channel: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=chunk_id, doc_id="d1", text=f"正文 {chunk_id}"),
        score=1.0,
        channel=channel,
        rank=0,
    )


class _Channel:
    """替身通道：要么吐固定结果，要么抛指定的异常。"""

    def __init__(self, name: str, ids: list[str], *, error: Exception | None = None) -> None:
        self.name = name
        self._ids = ids
        self._error = error
        self.calls: list[dict] = []

    async def retrieve(
        self, query: str, top_k: int, filters: dict | None = None, image: object = None
    ):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters, "image": image})
        if self._error is not None:
            raise self._error
        return [_hit(chunk_id, channel=self.name) for chunk_id in self._ids]


def _run(retriever: MultiChannelRetriever, query: str = "q", top_k: int = 10, **kw):
    return asyncio.run(retriever.retrieve(query, top_k, **kw))


def test_two_channels_are_fused_and_every_channel_is_accounted_for() -> None:
    hybrid = _Channel("hybrid:milvus:c", ["a", "b"])
    bm25 = _Channel("bm25:milvus:c", ["b"])
    retriever = MultiChannelRetriever([hybrid, bm25], FusionConfig())

    hits = _run(retriever)

    assert [hit.chunk.chunk_id for hit in hits] == ["b", "a"], "两路都命中的 b 排第一"
    assert [hit.sources for hit in hits][0] == ["hybrid:milvus:c", "bm25:milvus:c"]
    assert retriever.name == "multi[hybrid:milvus:c,bm25:milvus:c]"

    stats = {row.channel: row for row in retriever.last_stats}
    assert stats["hybrid:milvus:c"].candidates == 2
    assert stats["hybrid:milvus:c"].kept == 2, "a 和 b 都是它召回的"
    assert stats["bm25:milvus:c"].candidates == 1
    assert stats["bm25:milvus:c"].kept == 1, "只有 b 是 bm25 也召回到的"
    assert stats["rrf"].candidates == 3 and stats["rrf"].kept == 2, "融合去重掉了重复的 b"
    assert all(row.error is None for row in retriever.last_stats)
    assert stats["hybrid:milvus:c"].latency_ms is not None
    assert stats["rrf"].latency_ms is None, "融合本身不单独计时"


def test_one_broken_channel_does_not_take_the_query_down() -> None:
    """单路失败照常出结果，但要在 stats 里留下痕迹 —— 静默少一路是最难查的。"""
    hybrid = _Channel("hybrid:milvus:c", ["a"])
    bm25 = _Channel("bm25:milvus:c", [], error=TimeoutError("Milvus 抖动"))
    retriever = MultiChannelRetriever([hybrid, bm25], FusionConfig())

    hits = _run(retriever)

    assert [hit.chunk.chunk_id for hit in hits] == ["a"]
    broken = next(row for row in retriever.last_stats if row.channel.startswith("bm25"))
    assert broken.candidates == 0 and "TimeoutError" in (broken.error or "")


def test_all_channels_broken_is_a_real_failure() -> None:
    """全挂了要抛出去：返回空列表会让上层以为「就是没命中」，然后照常生成一个没依据的答案。"""
    retriever = MultiChannelRetriever(
        [
            _Channel("hybrid:x", [], error=RuntimeError("Milvus 连不上")),
            _Channel("bm25:x", [], error=TimeoutError("超时")),
        ],
        FusionConfig(),
    )

    with pytest.raises(RetrievalError, match="Milvus 连不上"):
        _run(retriever)


def test_filters_reach_every_channel_and_the_pool_is_the_same_for_all() -> None:
    """候选池必须**一路一样多**：给得多的那侧票更多（实测踩过 5 条 vs 20 条）。"""
    hybrid = _Channel("hybrid:x", ["a", "b"])
    bm25 = _Channel("bm25:x", ["a"])
    retriever = MultiChannelRetriever([hybrid, bm25], FusionConfig())

    _run(retriever, "怎么配", top_k=5, filters={"modality": "image"})

    for channel in (hybrid, bm25):
        assert channel.calls == [
            {"query": "怎么配", "top_k": 10, "filters": {"modality": "image"}, "image": None}
        ]


def test_the_query_image_reaches_every_channel() -> None:
    """以图搜图：图要送到每一路，由通道自己决定用不用（bm25 那路没词就会返回空）。"""
    hybrid = _Channel("hybrid:x", ["a"])
    bm25 = _Channel("bm25:x", [])
    retriever = MultiChannelRetriever([hybrid, bm25], FusionConfig())

    _run(retriever, "", image=Path("photo.png"))

    for channel in (hybrid, bm25):
        assert channel.calls[0]["image"] == Path("photo.png")


def test_a_bigger_top_k_wins_over_the_configured_pool() -> None:
    """调用方要得比配置多时听调用方的 —— 否则 `-k 50` 会被悄悄截成 20。"""
    channel = _Channel("hybrid:x", ["a"])

    _run(MultiChannelRetriever([channel], FusionConfig(candidates_per_channel=20)), top_k=50)

    assert channel.calls[0]["top_k"] == 50


def test_the_pool_size_can_be_configured() -> None:
    channel = _Channel("hybrid:x", ["a"])

    _run(MultiChannelRetriever([channel], FusionConfig(candidates_per_channel=3)), top_k=2)

    assert channel.calls[0]["top_k"] == 3


def test_the_fused_board_is_cut_to_top_k() -> None:
    retriever = MultiChannelRetriever(
        [_Channel("hybrid:x", ["a", "b", "c"]), _Channel("bm25:x", ["d", "e"])],
        FusionConfig(),
    )

    hits = _run(retriever, top_k=2)

    assert len(hits) == 2, "每路各给一份候选，最终只按 top_k 交出去"


def test_k_from_config_reaches_the_fusion() -> None:
    """k 变了融合分数就要变 —— 配了不生效比没有这个配置更坏。"""
    channels = [
        _Channel("hybrid:x", ["a", "b"]),
        _Channel("bm25:x", ["b", "a"]),
    ]
    low = _run(MultiChannelRetriever(channels, FusionConfig(k=1)))
    high = _run(MultiChannelRetriever(channels, FusionConfig(k=1000)))

    assert low[0].score > high[0].score


def test_no_channel_at_all_is_rejected() -> None:
    with pytest.raises(ValueError, match="至少要有一路通道"):
        MultiChannelRetriever([], FusionConfig())


# ---- 重排（融合之后、交货之前）----


class _Reranker:
    """替身重排器：默认把顺序倒过来；可改成失败或返回空。"""

    name = "rerank:stub"

    def __init__(self, *, error: Exception | None = None, empty: bool = False) -> None:
        self._error = error
        self._empty = empty
        self.calls: list[tuple[str, list[str], int]] = []

    async def rerank(self, query: str, chunks: list, top_n: int):
        self.calls.append((query, [hit.chunk.chunk_id for hit in chunks], top_n))
        if self._error is not None:
            raise self._error
        if self._empty:
            return []
        return list(reversed(chunks))[:top_n]


def _channels(*, top_k: int = 10):
    return MultiChannelRetriever(
        [_Channel("hybrid:x", ["a", "b", "c"]), _Channel("bm25:x", ["b"])], FusionConfig()
    )


def test_rerank_runs_on_the_fused_pool_and_is_accounted_for() -> None:
    """重排看的是**融合后整个池子**（不只是最终那几条），交货才按 top_k 切。"""
    reranker = _Reranker()
    retriever = MultiChannelRetriever(
        [_Channel("hybrid:x", ["a", "b", "c"]), _Channel("bm25:x", ["b"])],
        FusionConfig(),
        reranker,
    )

    hits = _run(retriever, top_k=2)

    assert reranker.calls == [("q", ["b", "a", "c"], 2)], "池子 3 条全看，只要 2 条"
    assert [hit.chunk.chunk_id for hit in hits] == ["c", "a"], "倒序后取前 2"
    row = next(item for item in retriever.last_stats if item.channel == "rerank:stub")
    assert (row.candidates, row.kept) == (3, 2), "账本记的是「看全池 3 条、留 2 条」"
    assert row.error is None
    assert retriever.name.endswith("rerank:stub]")


def test_a_broken_reranker_falls_back_to_the_fused_order() -> None:
    """重排只是精排：模型没下下来不该让整次查询失败 —— 但要记进 stats，不能装作没事。"""
    reranker = _Reranker(error=RerankError("模型没下下来"))
    retriever = MultiChannelRetriever(
        [_Channel("hybrid:x", ["a", "b"]), _Channel("bm25:x", ["b"])], FusionConfig(), reranker
    )

    hits = _run(retriever)

    assert [hit.chunk.chunk_id for hit in hits] == ["b", "a"], "退回融合顺序"
    row = next(item for item in retriever.last_stats if item.channel == "rerank:stub")
    assert row.kept == 0 and "已退回融合顺序" in (row.error or "")


def test_an_empty_rerank_result_also_falls_back() -> None:
    """空结果同样算失败：别把好好的融合结果换成一片空白。"""
    retriever = MultiChannelRetriever(
        [_Channel("hybrid:x", ["a"])], FusionConfig(), _Reranker(empty=True)
    )

    hits = _run(retriever)

    assert [hit.chunk.chunk_id for hit in hits] == ["a"]
    row = next(item for item in retriever.last_stats if item.channel == "rerank:stub")
    assert "空结果" in (row.error or "")


def test_without_a_reranker_there_is_no_rerank_row() -> None:
    retriever = _channels()

    _run(retriever)

    assert all(not item.channel.startswith("rerank") for item in retriever.last_stats)


def test_a_pure_image_query_skips_rerank() -> None:
    """没有查询词时交叉编码器无从下手 —— 跳过并记账，别白花几秒还打乱顺序。"""
    reranker = _Reranker()
    retriever = MultiChannelRetriever([_Channel("hybrid:x", ["a", "b"])], FusionConfig(), reranker)

    hits = _run(retriever, "", image=Path("photo.png"))

    assert reranker.calls == [], "没调用重排器"
    assert [hit.chunk.chunk_id for hit in hits] == ["a", "b"], "保持融合顺序"
    row = next(item for item in retriever.last_stats if item.channel == "rerank:stub")
    assert "纯图查询" in (row.error or "")


class _ComplementChannel(_Channel):
    """「只补位」的通道（如实测的 graph 通道）：重复的候选会被融合层丢掉。"""

    complement_only = True


def test_a_complement_channel_only_contributes_chunks_nobody_else_found() -> None:
    """实测（2026-10-05）：graph 通道带回来的块与 hybrid 高度重叠，重复投票改不了名次
    （27 题名次变化 0 条）。所以这类通道只贡献新块 —— 而账上要同时看得到
    「它召回了几条」与「补进来几条」，否则「召回 10 入选 0」看着像坏了。
    """
    hybrid = _Channel("hybrid:milvus:c", ["a", "b"])
    graph = _ComplementChannel("graph:neo4j:c", ["a", "b", "c"])
    retriever = MultiChannelRetriever([hybrid, graph], FusionConfig())

    hits = _run(retriever)

    ids = [hit.chunk.chunk_id for hit in hits]
    assert "c" in ids, "只有新块 c 是它补进来的"
    stats = {row.channel: row for row in retriever.last_stats}
    assert stats["graph:neo4j:c"].candidates == 3, "召回数按原始算（含被丢的）"
    assert stats["graph:neo4j:c"].complement_dropped == 2
    assert stats["graph:neo4j:c"].kept == 1


def test_a_complement_channel_that_brings_nothing_new_is_harmless() -> None:
    """它一条新块都没有时（覆盖不到这份语料），别的路的名次一点不该受影响。"""
    hybrid = _Channel("hybrid:milvus:c", ["a", "b"])
    graph = _ComplementChannel("graph:neo4j:c", ["a", "b"])

    with_graph = _run(MultiChannelRetriever([hybrid, graph], FusionConfig()))
    without = _run(MultiChannelRetriever([_Channel("hybrid:milvus:c", ["a", "b"])], FusionConfig()))

    assert [hit.chunk.chunk_id for hit in with_graph] == [hit.chunk.chunk_id for hit in without]
