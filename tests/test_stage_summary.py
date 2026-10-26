"""产物摘要：页面上那条流程图与四个阶段面板的数字都从这儿来。

纯函数、只吃 dict —— 所以不用起服务、不用碰产物目录，直接喂真实产物的形状。
"""

from __future__ import annotations

from rag.ingest.summary import chunk_stats, parse_stats, section_rollup, vector_bytes


def _chunk(text: str, *, modality: str = "text", path: str = "", numbers: str = "") -> dict:
    return {
        "chunk_id": f"d:{text[:4]}",
        "text": text,
        "modality": modality,
        "heading_path": path,
        "heading_numbers": numbers,
    }


def test_parse_stats_counts_text_images_and_missing_ones() -> None:
    stats = parse_stats(
        {
            "text": "一二三四五",
            "mime": "text/html",
            "images": [{"asset_uri": "d/img01.png"}, {"asset_uri": "d/img02.png"}],
            "metadata": {"parser": "html", "images_missing": ["https://x/a.svg：不是图片"]},
        }
    )

    assert stats == {
        "chars": 5,
        "images": 2,
        "missing_images": ["https://x/a.svg：不是图片"],
        "parser": "html",
        "mime": "text/html",
    }


def test_parse_stats_tolerates_a_document_without_metadata() -> None:
    assert parse_stats({"text": "甲"})["images"] == 0
    assert parse_stats({})["chars"] == 0


def test_chunk_stats_reports_blocks_chars_modalities_and_sizes() -> None:
    """`sizes` 是每块的字数：页面接成一条宽度正比字数的色条，看切得匀不匀。"""
    stats = chunk_stats(
        [
            _chunk("一二三四", path="甲"),
            _chunk("一二", path="甲"),
            _chunk("![image](asset:d/img01.png)", modality="image", path="乙"),
        ]
    )

    assert stats["blocks"] == 3
    assert stats["chars"] == 4 + 2 + len("![image](asset:d/img01.png)")
    assert stats["avg_chars"] == round(stats["chars"] / 3)
    assert stats["max_chars"] == len("![image](asset:d/img01.png)")
    assert stats["text_blocks"] == 2 and stats["image_blocks"] == 1
    assert stats["sizes"] == [4, 2, len("![image](asset:d/img01.png)")]


def test_chunk_stats_on_an_empty_document_does_not_divide_by_zero() -> None:
    stats = chunk_stats([])

    assert (stats["blocks"], stats["chars"], stats["avg_chars"], stats["max_chars"]) == (0, 0, 0, 0)
    assert stats["sizes"] == [] and stats["sections"] == []


def test_sections_roll_up_by_heading_path_keeping_document_order() -> None:
    """大纲要的是「哪几节、各多少块多少字」，不是 393 条块。"""
    rows = section_rollup(
        [
            _chunk("前言", path=""),
            _chunk("甲一", path="章 -> 甲", numbers="1-1"),
            _chunk("甲二", path="章 -> 甲", numbers="1-1"),
            _chunk("乙一", path="章 -> 乙", numbers="1-2"),
        ]
    )

    assert [row["heading_path"] for row in rows] == ["", "章 -> 甲", "章 -> 乙"]
    assert rows[1] == {
        "heading_path": "章 -> 甲",
        "heading_numbers": "1-1",
        "blocks": 2,
        "chars": 4,
        # 这一节第一块的开头：有它才知道「（无标题）133 字」那种行到底是什么
        "head": "甲一",
    }
    assert rows[0]["blocks"] == 1, "第一个标题之前的正文自成一类，排在最前"


def test_each_section_carries_a_preview_of_its_first_block() -> None:
    """「（无标题）1 块 133 字」这种行必须自带一句原文，否则没人判断得出它是什么。"""
    rows = section_rollup(
        [
            _chunk("Search / Edit this Page GitHub Project", path=""),
            _chunk("![image](asset:d/img01.png)\n\n正文第一句。", path="甲"),
        ]
    )

    assert rows[0]["head"].startswith("Search")
    assert rows[1]["head"] == "[图片] 正文第一句。", "图片标记换成 [图片]、压成一行"


def test_vector_bytes_is_blocks_times_dim_times_four() -> None:
    """稠密向量的理论占用：每块 `dim` 个 float32（页面上标明这只是估算）。"""
    assert vector_bytes(393, 2560) == 393 * 2560 * 4
    assert vector_bytes(0, 2560) == 0
