"""新字段的落地：schema 声明、写库映射、以及 make_chunk 自动算出来的那些。

全部本地 —— 不连 Milvus、不建表、不写数据。`build_schema` 只是构造 schema 对象，
`_row` 只是拼一个 dict，两者都不发请求。
"""

from __future__ import annotations

import pytest
from pymilvus import DataType

from rag.config.schema import VectorIndexConfig
from rag.contracts import Chunk, Document
from rag.indexing.milvus.client import MilvusIndex
from rag.indexing.milvus.registry import DocRegistry
from rag.indexing.milvus.schema import (
    MAX_ASSETS,
    build_registry_schema,
    build_schema,
    registry_collection,
)
from rag.ingest.chunkers.base import HeadingNumbering, make_chunk, with_heading_path

NEW_FIELDS = {
    "doc_type",
    "modality",
    "heading_path",
    "heading_numbers",
    "page",
    "line",
    "assets",
    "content_hash",
}


def test_schema_declares_the_new_fields() -> None:
    """这些是**显式列**，不是塞在 metadata JSON 里 —— 显式才能过滤。"""
    schema = build_schema(VectorIndexConfig(), dim=1024)
    names = {field.name for field in schema.fields}

    assert names >= NEW_FIELDS
    # 旧字段一个都不能少
    assert {"chunk_id", "doc_id", "chunk_index", "source_uri", "metadata", "text"} <= names


def test_assets_is_a_typed_array_not_json() -> None:
    """图片路径用 ARRAY<VARCHAR>：类型化、容量显式、能用 array_contains 过滤。"""
    schema = build_schema(VectorIndexConfig(), dim=1024)
    assets = next(field for field in schema.fields if field.name == "assets")

    assert assets.dtype == DataType.ARRAY
    assert assets.element_type == DataType.VARCHAR
    assert assets.max_capacity == MAX_ASSETS


# ---- 文档注册表（增量更新的依据）----


def test_registry_name_is_derived_from_the_block_collection() -> None:
    """两者必须成对：注册表描述的就是块表里的那些文档，派生就不会走散。"""
    cfg = VectorIndexConfig(collection="rag_dev_x")

    assert registry_collection(cfg) == "rag_dev_x_docs"
    assert DocRegistry(cfg).collection == "rag_dev_x_docs"


def test_registry_schema_has_the_bookkeeping_fields() -> None:
    names = {field.name for field in build_registry_schema().fields}

    assert {
        "doc_id",
        "source_uri",
        "content_hash",
        "chunk_count",
        "ingested_at",
        "pipeline_version",
    } <= names


def test_registry_needs_a_placeholder_vector_of_at_least_two_dims() -> None:
    """Milvus 的两条硬约束，都是实测撞出来的：

    ① 只有标量字段的 collection 建不了（`schema does not contain vector field`）；
    ② 维度不能小于 2（`invalid dimension: 1. should be in range 2 ~ 32768`）。

    所以注册表带一个 2 维占位向量 —— 不参与检索，纯粹为了能建表。
    """
    placeholder = next(f for f in build_registry_schema().fields if f.name == "placeholder")

    assert placeholder.dtype == DataType.FLOAT_VECTOR
    assert placeholder.params.get("dim") == 2


def _chunk(**over: object) -> Chunk:
    base: dict[str, object] = {
        "chunk_id": "d1:0000",
        "doc_id": "d1",
        "text": "正文\n\n第一章 -> 第一节",
        "index": 0,
        "doc_type": "docx",
        "modality": "text",
        "heading_path": "第一章 -> 第一节",
        "heading_numbers": "1-1",
        "page": 3,
        "line": 12,
        "assets": ["d1/img001.png"],
        "content_hash": "abc123",
    }
    return Chunk(**{**base, **over})  # type: ignore[arg-type]


def test_row_carries_every_new_field_to_milvus() -> None:
    index = MilvusIndex(VectorIndexConfig(), dim=4)

    row = index._row(_chunk(), [0.0, 0.0, 0.0, 0.0])

    names = NEW_FIELDS | {"chunk_id", "text", "dense", "metadata"}
    assert names <= set(row)
    assert row["assets"] == ["d1/img001.png"]
    assert row["page"] == 3 and row["line"] == 12
    # sparse 绝不能出现在写入行里：它是 BM25 function 的输出
    assert "sparse" not in row


def test_too_many_assets_fails_loudly_instead_of_truncating() -> None:
    """与 text 超长同一个立场：当场报错。悄悄少几张图是查不出来的。"""
    index = MilvusIndex(VectorIndexConfig(), dim=4)
    chunk = _chunk(assets=[f"d1/img{i:03d}.png" for i in range(MAX_ASSETS + 1)])

    with pytest.raises(ValueError, match="assets"):
        index._row(chunk, [0.0, 0.0, 0.0, 0.0])


# ---- make_chunk 自动算出来的东西 ----


def test_make_chunk_extracts_assets_and_content_hash() -> None:
    """图片路径与内容哈希能从正文直接看出来，就别让人手工填。"""
    doc = Document(
        doc_id="d1",
        source_uri="a.docx",
        text="x",
        metadata={"parser": "docx"},
    )

    chunk = make_chunk(
        doc,
        0,
        "看图 ![image](asset:d1/img001.png) 和 ![image](asset:d1/img002.png)\n\n标题",
        heading_path="标题",
        heading_numbers="1",
    )

    assert chunk.assets == ["d1/img001.png", "d1/img002.png"]
    assert len(chunk.content_hash) == 16
    assert chunk.doc_type == "docx"
    assert chunk.heading_path == "标题"


def test_make_chunk_dedupes_repeated_assets() -> None:
    doc = Document(doc_id="d1", source_uri="a.md", text="x")
    marker = "![image](asset:d1/img001.png)"

    chunk = make_chunk(doc, 0, f"{marker} 中间 {marker}")

    assert chunk.assets == ["d1/img001.png"], "同一个资产只记一次"


# ---- 标题编号：有作者编号就用它，没有才按位置数 ----


def test_heading_numbers_fall_back_to_position_when_there_is_no_parseable_number() -> None:
    """解析不出编号的标题（中文编号「第一章、」「一、」）按位置数，编号照样连续可比较。

    能解析出阿拉伯编号时**用作者的编号**（那才是用户会说的「第 2.1 节」），
    见 tests/test_headings.py。
    """
    numbering = HeadingNumbering()

    # 同一份文档里混着「第一章、」「一、」「1、」，编号照样是 1 / 1-1 / 1-1-1
    assert numbering.update({"h1": "第一章、概念"}) == ("第一章、概念", "1")
    assert numbering.update({"h1": "第一章、概念", "h2": "一、定义"}) == (
        "第一章、概念 -> 一、定义",
        "1-1",
    )
    assert numbering.update({"h1": "第一章、概念", "h2": "一、定义", "h3": "1、细节"}) == (
        "第一章、概念 -> 一、定义 -> 1、细节",
        "1-1-1",
    )


def test_a_new_sibling_bumps_only_its_own_level() -> None:
    numbering = HeadingNumbering()
    numbering.update({"h1": "第一章", "h2": "第一节"})

    assert numbering.update({"h1": "第一章", "h2": "第二节"})[1] == "1-2"
    # 换章时，下级要归零
    assert numbering.update({"h1": "第二章", "h2": "第一节"})[1] == "2-1"


def test_no_heading_means_empty_path_and_numbers() -> None:
    assert HeadingNumbering().update({}) == ("", "")


def test_with_heading_path_puts_the_heading_in_front() -> None:
    assert with_heading_path("正文。", "第一章 -> 第一节") == "第一章 -> 第一节\n\n正文。"
    assert with_heading_path("正文。", "") == "正文。"
