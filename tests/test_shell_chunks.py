"""空壳块过滤：剥掉标题路径与图片标记后没有任何可检索文字的块不入库。

证据来自那份 10MB 的 Word 文档（现已不在语料里）：173 个正文块里 21 个正文不足
50 字，最短的只剩一张表的 `| --- |` 分隔行，还有一个正文里只有一个图片标记。
这里把那些形状做成用例钉住。
"""

from __future__ import annotations

from rag.contracts import Chunk
from rag.ingest.pipeline import drop_shell_chunks, is_shell_chunk


def _chunk(text: str, *, heading: str = "", modality: str = "text") -> Chunk:
    body = f"{heading}\n\n{text}" if heading else text
    return Chunk(chunk_id="d:0000", doc_id="d", text=body, heading_path=heading, modality=modality)


# ---- 该丢的 ----


def test_only_an_image_marker_is_a_shell() -> None:
    """正文只有一个图片标记 —— 图在哪、什么标题下，图片描述块里已经有了。"""
    chunk = _chunk("![image](asset:doc1/img001.png)", heading="第一章 -> 第一节")

    assert is_shell_chunk(chunk) is True


def test_table_separator_only_is_a_shell() -> None:
    """只剩一张表的 `| --- |` 分隔行。"""
    assert is_shell_chunk(_chunk("| --- |", heading="第一章")) is True
    assert is_shell_chunk(_chunk("| --- | --- |")) is True
    assert is_shell_chunk(_chunk("---")) is True


def test_blank_body_is_a_shell() -> None:
    assert is_shell_chunk(_chunk("   \n\n  ")) is True


# ---- 该留的 ----


def test_short_but_real_text_is_kept() -> None:
    """短不等于空：只有几个字的正文照样可能是答案的唯一出处。"""
    assert is_shell_chunk(_chunk("见下。", heading="第一章")) is False
    assert is_shell_chunk(_chunk("OK")) is False


def test_a_table_row_with_content_is_kept() -> None:
    """一行表格是 `table-rows` 策略的最小单元，不能因为短就丢。"""
    assert is_shell_chunk(_chunk("| 客户 | 数量 |\n| 甲公司 | 512 |", modality="table")) is False


def test_an_image_with_some_text_around_it_is_kept() -> None:
    chunk = _chunk("上图是作业图。\n\n![image](asset:doc1/img001.png)", heading="第一章")

    assert is_shell_chunk(chunk) is False


def test_image_description_chunks_are_never_shells() -> None:
    """图片描述块是图片在库里的唯一入口，无论多短都留。"""
    assert is_shell_chunk(_chunk("![image](asset:doc1/img001.png)", modality="image")) is False


# ---- 过滤本身 ----


def test_drop_keeps_order_and_does_not_renumber() -> None:
    """序号不重排：chunk_id = doc_id:序号 要跨次灌库稳定。"""
    chunks = [
        Chunk(chunk_id="d:0000", doc_id="d", text="真内容。", index=0),
        Chunk(chunk_id="d:0001", doc_id="d", text="| --- |", index=1),
        Chunk(chunk_id="d:0002", doc_id="d", text="![image](asset:d/x.png)", index=2),
        Chunk(chunk_id="d:0003", doc_id="d", text="也是真内容。", index=3),
    ]

    kept, dropped = drop_shell_chunks(chunks)

    assert dropped == 2
    assert [c.chunk_id for c in kept] == ["d:0000", "d:0003"], "留下的块序号保持原样"


def test_heading_prefix_alone_is_not_content() -> None:
    """标题路径不是内容：一块只有标题（正文空）也算空壳。"""
    chunk = _chunk("", heading="第二章 -> 2.1 环境准备")

    assert is_shell_chunk(chunk) is True
