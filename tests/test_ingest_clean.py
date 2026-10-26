"""产线 markdown 的清洗：剥版式、补行边界换行。

纯字符串变换，不碰网络、不碰 pymupdf —— 清洗是它自己的一个环节，出错时
应该能单独定位，而不是必须拉一条 OCR 链路才复现。
"""

from __future__ import annotations

from pathlib import Path

from rag.config.schema import ChunkingConfig
from rag.contracts import Document
from rag.ingest.assets import AssetStore
from rag.ingest.chunkers import build_chunker
from rag.ingest.cleaning import (
    clean_markdown,
    clean_ocr_markdown,
    compact_html,
    strip_invisible,
)
from rag.ingest.loaders.base import pages_to_text
from rag.ingest.ocr import OcrPage
from rag.storage import LocalBlobStore

# 产线真实输出的形态（照抄 2026-09-29 实测的一段）
_TD = "style='text-align: center; word-wrap: break-word;'"
_PRODUCTION_TABLE = (
    "<table border=1 style='margin: auto; word-wrap: break-word;'>"
    f"<tr><td {_TD}>模型</td><td {_TD}>维度</td></tr>"
    f"<tr><td {_TD}>bge-small</td><td {_TD}>512</td></tr>"
    f"<tr><td {_TD}>bge-large</td><td {_TD}>1024</td></tr>"
    "</table>"
)
_CELLS = ("模型", "维度", "bge-small", "512", "bge-large", "1024")


# ---- 剥版式 ----


def test_production_table_loses_its_inline_styles() -> None:
    cleaned = clean_ocr_markdown(_PRODUCTION_TABLE)

    assert "style=" not in cleaned
    assert "border=" not in cleaned
    for cell in _CELLS:
        assert cell in cleaned, f"单元格 {cell!r} 被洗掉了，等于查不到"


def test_cleaning_shrinks_the_table_by_more_than_half() -> None:
    """版式占了大部分字符 —— 这正是切块被撑爆的原因。这条把收益钉住。"""
    cleaned = clean_ocr_markdown(_PRODUCTION_TABLE)

    assert len(cleaned) < len(_PRODUCTION_TABLE) * 0.4, (
        f"原 {len(_PRODUCTION_TABLE)} 字符 → 清洗后 {len(cleaned)}，收益不及预期"
    )


def test_rowspan_and_colspan_survive() -> None:
    """这两个属性说的是结构（跨行/跨列合并），不是版式，必须留下。"""
    cleaned = compact_html('<td rowspan="2" class="x" colspan="3">值</td>')

    assert 'rowspan="2"' in cleaned
    assert 'colspan="3"' in cleaned
    assert "class=" not in cleaned


def test_plain_markdown_and_entities_are_untouched() -> None:
    """只剥标签属性。正文里的 markdown 语法与字符引用一律原样保留。"""
    source = "# 标题\n\n**粗体** 与 `代码`\n\n| a | b |\n\nR&D &amp; 5 &lt; 6"

    assert compact_html(source) == source


def test_text_inside_a_table_is_never_decoded_or_rewritten() -> None:
    cleaned = compact_html(f"<td {_TD}>a &amp; b</td>")

    assert cleaned == "<td>a &amp; b</td>"


# ---- 纯文本 / markdown 的清洗（2026-10-10）----


def test_markdown_heading_anchors_are_dropped() -> None:
    """文档站挂在标题上的空锚点不是内容，删掉。

    实测来源：Firecrawl 抓下来的 mintlify 页面，32 个标题行里 **11 个**长这样
    （`## [<ZWSP>](…#step-1)  Step 1: …`）。不清掉就一路传到标题清单、`heading_path`
    与库里的正文 —— 页面上就是标题前面挂一串 `[](https://…)`。
    """
    text = "## [\u200b](https://docs.example.com/a#step-1)  Step 1: 装依赖\n正文一行。\n"

    cleaned = clean_markdown(text)

    assert cleaned.startswith("## Step 1: 装依赖\n"), "锚点与多余空格都不留"
    assert "https://docs.example.com" not in cleaned and "\u200b" not in cleaned


def test_normal_links_and_images_are_kept() -> None:
    """**正常链接一动不动**；空 alt 的图片 `![](x.png)` 也不是空锚点（别把它删了）。"""
    text = "见 [官方文档](https://docs.example.com/a) 、 ![图](a.png) 与 ![](b.png)\n"

    assert clean_markdown(text) == text


def test_heading_spaces_are_tidied_but_body_indentation_is_untouched() -> None:
    """只收标题行的空格：正文缩进（含代码块）是有意义的，一个字都不动。"""
    text = "###    标题\n\n   缩进的正文\n"

    cleaned = clean_markdown(text)

    assert "### 标题" in cleaned
    assert "\n   缩进的正文" in cleaned


def test_leading_site_chrome_is_dropped_but_body_quotes_are_kept() -> None:
    """文档**开头**的站点提示丢掉；正文**中段**的引用块是内容，一个字不动。"""
    text = (
        "> ## Documentation Index\n"
        "> Fetch the complete documentation index at: [/llms.txt](https://x/llms.txt)\n"
        "\n"
        "[Skip to main content](https://x/a#content)\n"
        "\n"
        "Deep Agents is the easiest way…\n"
        "\n"
        "> 正文中段的引用：这一句是内容，不许动。\n"
    )

    cleaned = clean_markdown(text)

    assert cleaned.startswith("Deep Agents is the easiest way…")
    assert "llms.txt" not in cleaned and "Skip to main content" not in cleaned
    assert "> 正文中段的引用" in cleaned


def test_a_leading_quote_that_is_not_site_chrome_stays() -> None:
    """开头就是引用块、但**不含**站点提示词 → 那是内容（比如一篇摘录），保留。"""
    text = "> 这是摘录的第一句。\n> 第二句。\n\n正文。\n"

    assert clean_markdown(text) == text


def test_invisible_characters_are_stripped_from_ocr_markdown() -> None:
    """OCR 那侧同样会带不可见字符（实测标题尾部一个 U+200B）。

    图标字体的私用区码点也一样 —— 它们会跟着进标题路径与正文，让精确匹配失效。
    """
    cleaned = clean_ocr_markdown("# 标题\u200b\n\n正文\uf0c1 结束")

    assert cleaned == "# 标题\n\n正文 结束"
    assert strip_invisible("\ue000\U000f0001") == ""


# ---- 行边界换行 ----


def test_table_rows_get_line_breaks() -> None:
    """产线给的是一整行 HTML，没有换行可断 —— 补在行边界，切块器才有地方下刀。"""
    cleaned = clean_ocr_markdown(_PRODUCTION_TABLE)

    assert cleaned.startswith("<table>")
    assert "\n<tr>" in cleaned
    assert cleaned.count("\n<tr>") == 3


def test_a_long_table_breaks_between_rows_not_inside_tags() -> None:
    """清洗后仍要保证：块不会从标签中间切开。

    用 20 行表把 chunk size 撑爆，逐块检查尖括号配平 —— 从标签中间切开会留下
    半个 `<td style=...` 这种残骸。这条是「补换行」这个设计的存在理由。
    """
    rows = "".join(f"<tr><td>行{i}</td><td>值{i}</td></tr>" for i in range(20))
    doc = Document(
        doc_id="d",
        source_uri="scan.pdf",
        text=clean_ocr_markdown(f"<table border=1>{rows}</table>"),
    )

    chunks = build_chunker(ChunkingConfig(strategy="recursive", size=400, overlap=0)).split([doc])

    assert len(chunks) > 1, "表没被撑爆，这条用例失去意义"
    for chunk in chunks:
        assert chunk.text.count("<") == chunk.text.count(">"), f"被从标签中间切开了：{chunk.text!r}"


# ---- 落到 loader 的公共入口上 ----


def test_pages_to_text_joins_pages_and_cleans_them(tmp_path: Path) -> None:
    """PDF 与图片两条路径共用这一个入口 —— 清洗必须在这里生效，不能只在一边。"""
    pages = [
        OcrPage(page_index=0, markdown="# 年报"),
        OcrPage(page_index=1, markdown="   "),  # 空白页：产线对空白页返回空串
        OcrPage(page_index=2, markdown=_PRODUCTION_TABLE),
    ]

    result = pages_to_text(pages, store=AssetStore(LocalBlobStore(tmp_path)), doc_id="d1")

    assert result.text.startswith("# 年报")
    assert "style=" not in result.text
    assert "bge-large" in result.text
    assert "\n\n\n" not in result.text, "空白页不该留下成串的空行"
    assert result.images == []


def test_pages_to_text_normalizes_crlf(tmp_path: Path) -> None:
    """CRLF 会原样进库、出现在引用片段里，也干扰分词 —— 入口处统一掉。"""
    result = pages_to_text(
        [OcrPage(page_index=0, markdown="a\r\nb")],
        store=AssetStore(LocalBlobStore(tmp_path)),
        doc_id="d1",
    )

    assert result.text == "a\nb"
