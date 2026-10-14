"""接入层：编码嗅探、跳过逻辑、切块边界。"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from rag.config.schema import (
    ChunkingConfig,
    IndexingConfig,
    IngestConfig,
    IngestPdfConfig,
    Settings,
)
from rag.ingest import prepare
from rag.ingest.chunkers import build_chunker
from rag.ingest.loaders import LoadReport, load_source, read_text_any_encoding

# 默认 ingest 配置：parser=ocr，但 ocr 传 None。这些用例用的是 md/txt，
# 走不到 PDF 那条分支，所以一个网络请求都不该发出去。
_CFG = IngestConfig()


def _load(source: Path) -> LoadReport:
    return load_source(source, _CFG, None)


def _write(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode(encoding))
    return path


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """一份含各种「坑」的语料：GBK 编码、嵌套目录、隐藏目录、不支持后缀、空文件。"""
    _write(tmp_path / "a.md", "# 向量检索\n\nMilvus 支持稀疏向量。也支持混合检索。")
    _write(tmp_path / "b.txt", "中文语料用 GBK 编码很常见，直接按 utf-8 读会乱码。", "gb18030")
    _write(tmp_path / "sub" / "c.md", "嵌套目录里的文档也要被发现。")
    _write(tmp_path / ".hidden" / "h.md", "隐藏目录里的内容不该进库。")
    _write(tmp_path / "d.xyz", "不支持的后缀。")
    _write(tmp_path / "empty.txt", "   \n\n  ")
    return tmp_path


def test_gbk_file_is_decoded_not_mangled(tmp_path: Path) -> None:
    """中文语料常见 GBK。只按 utf-8 读要么抛错要么整篇变替换字符。"""
    target = _write(tmp_path / "gbk.txt", "稀疏向量检索", "gb18030")
    assert read_text_any_encoding(target) == "稀疏向量检索"


def test_crlf_is_normalized_to_lf(tmp_path: Path) -> None:
    """Windows 上的 .md/.txt 是 CRLF，\\r 不该原样进库、进引用片段。"""
    target = tmp_path / "crlf.txt"
    target.write_bytes("第一段。\r\n\r\n第二段。\r\n".encode())
    doc = _load(target).documents[0]
    assert "\r" not in doc.text
    assert doc.text == "第一段。\n\n第二段。\n"


def test_doc_id_follows_path_not_content(corpus: Path) -> None:
    """doc_id 必须只由路径决定：内容哈希会让改一个字就产生一批孤儿 chunk。"""
    first = _load(corpus).documents
    before = {d.source_uri: d.doc_id for d in first}

    _write(corpus / "a.md", "# 向量检索\n\n内容改了。")
    after = {d.source_uri: d.doc_id for d in _load(corpus).documents}
    assert after["a.md"] == before["a.md"]


def test_discover_skips_hidden_dirs(corpus: Path) -> None:
    """隐藏目录整片排除（政策），不支持的后缀则要进 skipped 被报出来。"""
    report = _load(corpus)
    uris = [d.source_uri for d in report.documents]
    assert "a.md" in uris
    assert "sub/c.md" in uris
    assert not any("hidden" in uri for uri in uris)
    assert not any(uri.endswith(".xyz") for uri in uris)
    assert "d.xyz" in {item.path for item in report.skipped}


def test_discover_skips_saved_page_asset_dirs(tmp_path: Path) -> None:
    """浏览器「保存网页」会生成 `<页面名>_files/` 放 JS/CSS/图标。

    不跳过的话，里面的图标 PNG 会被当成图片一张张送去 OCR，跟踪用的小 HTML
    也会进来 —— 实测一份 Confluent 文档页能带出 4 张 PNG + 4 个 HTML。
    """
    page = tmp_path / "page.html"
    page.write_text("<html><body><main><p>正文</p></main></body></html>", encoding="utf-8")
    assets = tmp_path / "page_files"
    assets.mkdir()
    (assets / "icon.png").write_bytes(b"\x89PNG")
    (assets / "tracker.html").write_text("<html><body>1x1</body></html>", encoding="utf-8")

    uris = [d.source_uri for d in _load(tmp_path).documents]

    assert uris == ["page.html"]


def test_skipped_files_are_reported_with_reason(corpus: Path) -> None:
    """空正文/不支持的后缀若不报出来，用户会以为整批都 ingest 成功了。"""
    report = _load(corpus)
    reasons = {item.path: item.reason for item in report.skipped}
    assert "empty.txt" in reasons
    assert "空" in reasons["empty.txt"]
    assert ".xyz" in reasons["d.xyz"]


def test_source_uri_is_relative_to_corpus_root(corpus: Path) -> None:
    """绝对路径会随机器漂移，引用里也不好看。"""
    for doc in _load(corpus).documents:
        assert not Path(doc.source_uri).is_absolute()


def test_pymupdf_fallback_extracts_text_and_counts_pages(tmp_path: Path) -> None:
    """`parser: pymupdf` 是显式降级：只要文字层，标题/表格/图片全拿不到。
    空白页不进正文，但仍计入页数。默认路径已统一走 OCR（见 test_ingest_ocr.py）。
    """
    target = tmp_path / "doc.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Hello PDF", fontsize=12)
        page.insert_text((72, 100), "向量数据库", fontsize=12, fontname="china-s")
        pdf.new_page()  # 空白页
        pdf.save(target)

    cfg = IngestConfig(pdf=IngestPdfConfig(parser="pymupdf"))
    doc = load_source(target, cfg, None).documents[0]

    assert "Hello PDF" in doc.text
    assert "向量数据库" in doc.text
    assert doc.metadata["page_count"] == 2
    assert doc.metadata["parser"] == "pymupdf"
    assert doc.source_uri == "doc.pdf"


def test_recursive_chunker_honours_size_and_makes_stable_ids() -> None:
    from rag.contracts import Document

    cfg = ChunkingConfig(strategy="recursive", size=20, overlap=0)
    doc = Document(
        doc_id="abc",
        source_uri="x.md",
        text="向量数据库支持稀疏检索。它还能做混合检索！这是第三句话，用来凑长度。",
    )
    chunks = build_chunker(cfg).split([doc])

    assert len(chunks) > 1
    assert all(len(c.text) <= cfg.size for c in chunks)
    # chunk_id 确定性生成，且带上来源 —— Chunk 本身没有 source_uri 字段
    assert [c.chunk_id for c in chunks] == [f"abc:{i:04d}" for i in range(len(chunks))]
    assert all(c.metadata["source_uri"] == "x.md" for c in chunks)
    # 这段没有任何换行，而句末标点已不再是切分边界（见下一条用例），所以是按字符硬切的
    assert chunks[0].text == doc.text[: cfg.size]


def test_recursive_chunker_only_breaks_at_newlines() -> None:
    """分隔符只留换行（2026-09-30 收窄）。

    退到句末标点、逗号、空格时，切口会越来越接近句子中间 —— 一块只剩半句话，
    dense 与 BM25 两路同时失真。所以：**换行是唯一边界**，没有换行的长段落宁可硬切。
    """
    from rag.contracts import Document

    cfg = ChunkingConfig(strategy="recursive", size=12, overlap=0)
    doc = Document(
        doc_id="abc",
        source_uri="x.md",
        text="第一句。第二句。\n第三段里有句号和逗号，但没有换行，还有很长。",
    )
    chunks = build_chunker(cfg).split([doc])

    assert chunks[0].text == "第一句。第二句。", "第一个换行才是边界，句号不是"
    assert all(len(c.text) <= cfg.size for c in chunks)
    # 硬切只是切尺寸，不该丢字
    assert "".join(c.text for c in chunks) == doc.text.replace("\n", "")


def test_markdown_chunker_carries_heading_path() -> None:
    from rag.contracts import Document

    text = """# 检索

总览。

## 稀疏

BM25 是统计式的稀疏检索。

### 分词

中文要专门分词。"""
    doc = Document(doc_id="m1", source_uri="guide.md", text=text)
    chunks = build_chunker(ChunkingConfig(strategy="markdown", size=400, overlap=0)).split([doc])

    assert len(chunks) == 3
    assert chunks[0].heading_path == "检索"
    assert chunks[1].heading_path == "检索 -> 稀疏"
    assert chunks[2].heading_path == "检索 -> 稀疏 -> 分词"
    assert chunks[2].heading_numbers == "1-1-1"
    # 标题不进正文：剥掉之后贴到开头（读这一块时先知道它在哪一节）
    assert chunks[2].text == "检索 -> 稀疏 -> 分词\n\n中文要专门分词。"


def test_unknown_strategy_fails_loudly() -> None:
    with pytest.raises(KeyError, match="未注册的 chunker"):
        build_chunker(ChunkingConfig(strategy="no_such_strategy"))


def test_prepare_returns_chunks_and_skip_report(corpus: Path) -> None:
    settings = Settings(
        indexing=IndexingConfig(chunk=ChunkingConfig(strategy="recursive", size=200, overlap=20))
    )
    prep = prepare(corpus, settings, None)
    assert prep.documents == 3
    assert len(prep.chunks) >= 3
    assert [item.path for item in prep.skipped]  # 至少 empty.txt / d.xyz
    # chunk_id 全局唯一：它是 Milvus 主键，重复会互相覆盖
    assert len({c.chunk_id for c in prep.chunks}) == len(prep.chunks)
