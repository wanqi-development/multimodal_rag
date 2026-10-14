"""图片资产：落盘、正文标记、以及「上边一段 / 下边一段」。

图片描述要带上下文 —— 这些用例钉住的就是上下文这件事能不能拿到。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config.schema import ChunkingConfig, IngestAssetsConfig, IngestConfig, IngestOcrConfig
from rag.contracts import Chunk, Document
from rag.ingest.assets import (
    AssetStore,
    context_for_marker,
    extract_images,
    image_paths_for,
    make_marker,
    markers_in,
    position_at,
)
from rag.ingest.chunkers import build_chunker
from rag.ingest.loaders.base import pages_to_text
from rag.ingest.ocr import OcrPage
from rag.storage import LocalBlobStore

_SRC = "imgs/img_in_image_box_69_91_1085_722.jpg"
# 产线真实输出的形态：一个居中的 div 包着 img，夹在前后段落之间
_PAGE = (
    "上边的段落讲的是 GraphRAG 的索引流程。\n\n"
    f'<div style="text-align: center;"><img src="{_SRC}" alt="Image" width="85%" /></div>\n\n'
    "下边的段落接着讲查询流程。"
)
_BYTES = b"\xff\xd8\xff\xe0jpeg-bytes"


def _extract(tmp_path: Path, *, markdown: str = _PAGE, images: dict[str, bytes] | None = None):
    return extract_images(
        markdown,
        {_SRC: _BYTES} if images is None else images,
        store=AssetStore(LocalBlobStore(tmp_path)),
        doc_id="doc1",
        page=3,
    )


# ---- 落盘 + 标记 ----


def test_image_is_written_to_disk_and_replaced_by_a_marker(tmp_path: Path) -> None:
    result = _extract(tmp_path)

    assert "<img" not in result.text and "base64" not in result.text
    assert len(result.images) == 1
    asset = result.images[0]
    assert (tmp_path / asset.asset_uri).read_bytes() == _BYTES, "图片本体没落盘"
    assert asset.marker in result.text
    assert asset.page == 3


def test_marker_keeps_the_image_in_place_between_the_paragraphs(tmp_path: Path) -> None:
    """位置就是上下文：标记必须仍夹在原来的前后段落之间。"""
    result = _extract(tmp_path)

    before, after = result.text.split(result.images[0].marker)
    assert "上边的段落" in before
    assert "下边的段落" in after


def test_marker_carries_the_asset_uri(tmp_path: Path) -> None:
    asset = _extract(tmp_path).images[0]

    # 文件名取「页 + 序号」，不带产线那串 bbox —— 标记要进正文吃 chunk 预算，
    # 原始名字留在资产记录里供回查。
    assert asset.asset_uri == "doc1/p0003_01.jpg"
    assert asset.source_name == _SRC
    assert asset.marker == make_marker(asset.asset_uri)
    assert markers_in(f"x {asset.marker} y") == [asset.marker]


def test_reingesting_the_same_doc_overwrites_in_place(tmp_path: Path) -> None:
    """按 doc_id 分目录、按页与序号命名 —— 同一份文档重灌不该越堆越多份图。"""
    first = _extract(tmp_path).images[0]
    second = _extract(tmp_path).images[0]

    assert first.asset_uri == second.asset_uri
    assert len(list(tmp_path.rglob("*.jpg"))) == 1


def test_missing_image_data_fails_loudly(tmp_path: Path) -> None:
    """不静默丢图：少一张图而文档照常进库，是最难发现的那种错。"""
    with pytest.raises(ValueError, match="return_images"):
        _extract(tmp_path, images={})


def test_pages_without_images_are_untouched(tmp_path: Path) -> None:
    text = "没有图的正文。"

    result = extract_images(
        text, {}, store=AssetStore(LocalBlobStore(tmp_path)), doc_id="doc1", page=1
    )

    assert result.text == text
    assert result.images == []


# ---- 上下文：上边一段 / 下边一段 ----


def _chunks(tmp_path: Path, *, size: int) -> tuple[list[Chunk], str]:
    pages = [OcrPage(page_index=2, markdown=_PAGE, images={_SRC: _BYTES})]
    extracted = pages_to_text(pages, store=AssetStore(LocalBlobStore(tmp_path)), doc_id="doc1")
    chunks = build_chunker(ChunkingConfig(strategy="recursive", size=size, overlap=0)).split(
        [Document(doc_id="doc1", source_uri="scan.pdf", text=extracted.text)]
    )
    return chunks, extracted.images[0].marker


def test_context_is_the_paragraph_above_and_below(tmp_path: Path) -> None:
    chunks, marker = _chunks(tmp_path, size=400)

    before, after = context_for_marker(chunks, marker)

    assert before == "上边的段落讲的是 GraphRAG 的索引流程。"
    assert after == "下边的段落接着讲查询流程。"


def test_context_falls_back_to_neighbouring_chunks_at_a_boundary(tmp_path: Path) -> None:
    """边界把上下文切断是常事。块首/块尾取空时要向相邻块要，否则描述会少一半信息。"""
    chunks, marker = _chunks(tmp_path, size=40)  # 切得很碎，标记必然落在某个块的边缘

    assert len(chunks) > 1, "没切碎，这条用例失去意义"
    before, after = context_for_marker(chunks, marker)

    assert "上边的段落" in before
    assert "下边的段落" in after


def test_consecutive_images_skip_over_each_other(tmp_path: Path) -> None:
    """连续整页图之间本来就没有文字 —— 那一侧不能把另一个图片标记当成上下文。

    实测撞到的：产线把连续三页的整图排在一起，取「上边一段」拿到的是
    `![image](asset:…)`。标记塞给模型等于没给。
    """
    text = (
        "上面这段是正文。\n\n"
        "![image](asset:doc1/p0001_01.jpg)\n\n"
        "![image](asset:doc1/p0002_01.jpg)\n\n"
        "下面这段是正文。"
    )
    chunks = build_chunker(ChunkingConfig(strategy="recursive", size=400, overlap=0)).split(
        [Document(doc_id="doc1", source_uri="scan.pdf", text=text)]
    )

    before, after = context_for_marker(chunks, "![image](asset:doc1/p0001_01.jpg)")

    assert before == "上面这段是正文。"
    assert after == "下面这段是正文。", "另一张图的标记不算上下文"


def test_unknown_marker_is_rejected(tmp_path: Path) -> None:
    chunks, _ = _chunks(tmp_path, size=400)

    with pytest.raises(KeyError):
        context_for_marker(chunks, "![image](asset:nope.jpg)")


# ---- 块 → 图片本体（多模态 embedding 那一步的输入） ----


def _chunk(*, index: int, text: str, modality: str = "text") -> Chunk:
    from rag.ingest.assets import asset_uris_in

    return Chunk(
        chunk_id=f"doc1:{index:04d}",
        doc_id="doc1",
        text=text,
        index=index,
        modality=modality,
        assets=asset_uris_in(text),
    )


def test_only_image_chunks_carry_the_image_body(tmp_path: Path) -> None:
    """正文块里的标记**不**把图再送一遍 —— 那张图有自己的图片块。"""
    marker = make_marker("doc1/p0001_01.jpg")
    (tmp_path / "doc1").mkdir()
    (tmp_path / "doc1" / "p0001_01.jpg").write_bytes(_BYTES)
    chunks = [
        _chunk(index=0, text=f"正文\n{marker}"),
        _chunk(index=1, text=f"架构图描述\n{marker}", modality="image"),
    ]

    paths, missing = image_paths_for(chunks, AssetStore(LocalBlobStore(tmp_path)))

    assert paths[0] == [], "正文块不该带图"
    assert paths[1] == [tmp_path / "doc1" / "p0001_01.jpg"]
    assert missing == []


def test_a_missing_image_body_is_reported_and_skipped(tmp_path: Path) -> None:
    """本体读不到时只跳过这一张 —— 但不能静默：调用方要把它打给用户看。"""
    chunks = [_chunk(index=0, text=f"描述\n{make_marker('doc1/gone.jpg')}", modality="image")]

    paths, missing = image_paths_for(chunks, AssetStore(LocalBlobStore(tmp_path)))

    assert paths == [[]]
    assert missing == ["doc1/gone.jpg"]


# ---- 配置默认值 ----


def test_images_are_requested_by_default() -> None:
    """要挂描述就必须拿得到图，所以默认开。"""
    assert IngestOcrConfig().return_images is True
    assert IngestConfig().assets.dir == "./data/assets"
    assert IngestAssetsConfig(dir=str(Path("x"))).dir == "x"


# ---- 编号兜底与逐行打点（2026-10-08）----


def test_a_promoted_heading_keeps_the_page_mapping(tmp_path: Path) -> None:
    """标题兜底只往行首插 `#`：**行数不变**，所以「第几页第几行」照旧对得上。

    形状取自 `第一章 Apache Flink 概述.pdf`：`1./2./3.` 在第 9 页、编号接得上的 `## 4.`
    在第 10 页 —— **跨页**，所以兜底必须在拼完整份之后再跑（逐页做就看不见那条判据）。
    """
    first = OcrPage(
        page_index=8,
        markdown="## 1.3 Flink核心特性\n\n1. 批流一体化\n\n正文。\n\n2. 同时支持高吞吐\n\n正文。",
        images={},
    )
    second = OcrPage(
        page_index=9,
        markdown=(
            "3. 支持事件时间\n\n正文。\n\n## 4. 支持有状态计算\n\n正文。\n\n"
            "## 5. 支持灵活的窗口操作\n\n正文。"
        ),
        images={},
    )

    extracted = pages_to_text(
        [first, second], store=AssetStore(LocalBlobStore(tmp_path)), doc_id="doc1"
    )

    assert "## 1. 批流一体化" in extracted.text, "跨页的那串编号行被补成标题"
    assert "## 3. 支持事件时间" in extracted.text

    offsets = [point.offset for point in extracted.positions]
    for line, page in (("## 1. 批流一体化", 9), ("3. 支持事件时间", 10)):
        at = extracted.text.index(line)
        assert position_at(extracted.positions, offsets, at).page == page, f"{line} 的页码错了"
