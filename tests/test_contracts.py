"""contracts 层：数据模型与插件注册表。"""

from __future__ import annotations

import pytest

from rag.contracts import Chunk, Citation, Registry, RetrievalResult, ScoredChunk, format_sources
from rag.contracts.protocols import Retriever


def test_scored_chunk_carries_channel_and_rank() -> None:
    """融合层要靠 channel/rank 做 RRF，这两个字段不能丢。"""
    scored = ScoredChunk(
        chunk=Chunk(chunk_id="c1", doc_id="d1", text="hello"),
        score=0.5,
        channel="hybrid",
        rank=0,
    )
    assert scored.channel == "hybrid"
    assert scored.rank == 0


def test_retrieval_result_defaults_to_empty() -> None:
    result = RetrievalResult(query="q")
    assert result.chunks == []
    assert result.stats == []


def test_registry_rejects_duplicate_name() -> None:
    reg = Registry("demo")

    @reg.register("x")
    def first() -> None: ...

    with pytest.raises(ValueError, match="名称冲突"):
        reg.register("x")(lambda: None)


def test_registry_unknown_name_lists_registered() -> None:
    reg = Registry("demo")

    @reg.register("known")
    def impl() -> None: ...

    with pytest.raises(KeyError, match="未注册"):
        reg.get("known_typo")


def test_registry_contains_and_names() -> None:
    reg = Registry("demo")
    reg.register("b")(lambda: None)
    reg.register("a")(lambda: None)
    assert "a" in reg
    assert reg.names() == ["a", "b"]
    assert len(reg) == 2


def test_retriever_protocol_accepts_conforming_class() -> None:
    """协议是 runtime_checkable，装配时能直接做 isinstance 校验。"""

    class DummyRetriever:
        name = "dummy"

        async def retrieve(self, query: str, top_k: int, filters=None) -> list[ScoredChunk]:
            return []

    assert isinstance(DummyRetriever(), Retriever)


def test_retriever_protocol_rejects_missing_method() -> None:
    class NotARetriever:
        name = "nope"

    assert not isinstance(NotARetriever(), Retriever)


def _citation(index: int, *, uri: str = "a.pdf", heading: str = "第 1 章 -> 1.1") -> Citation:
    return Citation(index=index, chunk_id=f"c{index}", source_uri=uri, heading_path=heading)


def test_sources_list_one_line_per_section() -> None:
    """同一章节里命中三个块，清单里只该出现一次 —— 但**编号都要在**，
    否则正文里的 `[7]` 就成了悬空引用。"""
    text = format_sources([_citation(1), _citation(2), _citation(7)])

    assert text == "\n\n引用来源：\n[1][2][7] a.pdf · 第 1 章 -> 1.1"


def test_sources_are_sorted_by_number_and_keep_gaps() -> None:
    """编号沿用正文里的 `[n]`，重排就对不上了。"""
    text = format_sources([_citation(11, heading="第 2 章"), _citation(1)])

    assert text.strip().splitlines() == [
        "引用来源：",
        "[1] a.pdf · 第 1 章 -> 1.1",
        "[11] a.pdf · 第 2 章",
    ]


def test_a_citation_without_a_heading_shows_the_document_only() -> None:
    """图谱命中的行没有 heading_path，那就只写文档名。"""
    assert format_sources([_citation(3, heading="")]).endswith("[3] a.pdf")


def test_no_citations_means_no_tail() -> None:
    assert format_sources([]) == ""
