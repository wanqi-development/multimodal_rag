"""跨通道融合（L2）：RRF 只看名次、同一路去重、顺序确定。"""

from __future__ import annotations

import pytest

from rag.contracts import Chunk, ScoredChunk
from rag.retrieval.fusion import rrf_fuse


def _hit(chunk_id: str, *, channel: str = "hybrid:milvus:c", score: float = 1.0) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=chunk_id, doc_id="d1", text=f"正文 {chunk_id}"),
        score=score,
        channel=channel,
        rank=0,
    )


def _ids(hits: list[ScoredChunk]) -> list[str]:
    return [hit.chunk.chunk_id for hit in hits]


def test_a_hit_found_by_both_channels_beats_one_found_by_a_single_channel() -> None:
    """RRF 的本意：两路都认可的东西排前面。"""
    fused = rrf_fuse(
        [[_hit("a"), _hit("b")], [_hit("b"), _hit("c")]], k=60, sources=["hybrid", "bm25"]
    )

    assert _ids(fused) == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_rank_is_what_counts_not_the_score() -> None:
    """分数不在一个量纲上（COSINE vs BM25 vs 以后 web 的某种分），所以只比名次：
    给一个 99 分的 sparse 也压不过两路都排第 1 的那条。"""
    dense = [_hit("a", score=0.9), _hit("b", score=0.8)]
    sparse = [_hit("a", channel="bm25:m", score=1.0), _hit("b", channel="bm25:m", score=99.0)]

    fused = rrf_fuse([dense, sparse], sources=["hybrid", "bm25"])

    assert _ids(fused) == ["a", "b"]
    assert fused[0].score == pytest.approx(2 / 61)
    assert fused[1].score == pytest.approx(2 / 62), "99 分那条也只按名次算"


def test_the_board_records_which_channels_found_each_hit() -> None:
    """融合之后 channel 变成 rrf，真实来源只能靠 sources 保留下来。"""
    fused = rrf_fuse([[_hit("a")], [_hit("a", channel="bm25:m")]], sources=["hybrid", "bm25"])

    assert fused[0].sources == ["hybrid", "bm25"]
    assert fused[0].channel == "rrf"
    assert fused[0].rank == 0, "与单路那条一致：rank 从 0 起"


def test_a_duplicate_row_inside_one_channel_is_counted_once() -> None:
    fused = rrf_fuse([[_hit("a"), _hit("a")]], sources=["hybrid"])

    assert len(fused) == 1
    assert fused[0].score == pytest.approx(1 / 61)


def test_ties_are_broken_by_chunk_id() -> None:
    """同分必须有个确定的顺序，否则同一份数据两次检索给出不同的榜。"""
    fused = rrf_fuse([[_hit("b")], [_hit("a")]], sources=["x", "y"])

    assert _ids(fused) == ["a", "b"]


def test_k_flattens_the_gap_between_ranks() -> None:
    """k 是 RRF 唯一要调的参数：越大名次带来的差距越小。"""
    tight = rrf_fuse([[_hit("a"), _hit("b")]], k=1)
    flat = rrf_fuse([[_hit("a"), _hit("b")]], k=1000)

    assert (tight[0].score - tight[1].score) > (flat[0].score - flat[1].score)


def test_no_input_gives_an_empty_board() -> None:
    assert rrf_fuse([[], []], sources=["a", "b"]) == []
    assert rrf_fuse([]) == []


def test_without_names_it_falls_back_to_the_channel_on_the_hit() -> None:
    fused = rrf_fuse([[_hit("a", channel="bm25:m")]])

    assert fused[0].sources == ["bm25:m"]
