"""接入层的 OCR 分支：整份路由、图片接入、注册表。

不碰网络 —— 这里的 OCR 是手写的假客户端（记下被要求解析哪个文件、返回预设页）。
真的 HTTP 交互在 test_ocr_client.py 里用 MockTransport 验。
PDF 一律用 pymupdf 现造，版本差异比签入二进制件小。

2026-09-29 起 **所有 PDF 都走 OCR**，不再探测扫描页 —— 数字版走 pymupdf 只能拿
纯文本，标题/表格/图片全丢，下游的 markdown 切分与图片描述都会空转。
`parser: pymupdf` 降级成显式开关，用例见「显式降级」那一段。
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from rag.config.schema import (
    ChunkingConfig,
    IndexingConfig,
    IngestConfig,
    IngestOcrConfig,
    IngestPdfConfig,
    Settings,
)
from rag.ingest import build_ocr, prepare
from rag.ingest.loaders import LOADERS, build_loader, load_source
from rag.ingest.ocr import FILE_IMAGE, FILE_PDF, PARSER_NAME, OcrClient, OcrPage


class _FakeOcr:
    # 真实 client 都会报这个名字（工厂约定的一部分，见 ingest/ocr_client.py）
    name = "paddleocr"

    """假 OCR 客户端：记下调用，按预设页返回。"""

    def __init__(self, *pages: str) -> None:
        self.calls: list[tuple[Path, int]] = []
        self._pages = list(pages)

    def parse(self, path: Path, *, file_type: int) -> list[OcrPage]:
        self.calls.append((path, file_type))
        return [OcrPage(page_index=i, markdown=text) for i, text in enumerate(self._pages)]


# 默认字体写不进中文：插进去能画出来，抽回来是乱码甚至丢字。造真语料得用内建中文字体
_FONT = "china-s"


def _image_bytes() -> bytes:
    """一张真图片：把一页文字渲染成 PNG。造「有图无文字」的扫描页要用它。"""
    with pymupdf.open() as src:
        src.new_page().insert_text((72, 72), "扫描页", fontsize=20, fontname=_FONT)
        return src[0].get_pixmap().tobytes("png")


def _digital_pdf(path: Path) -> Path:
    """纯数字版：两页文字 + 一页空白尾页。"""
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "纯数字版正文", fontsize=12, fontname=_FONT)
        pdf.new_page().insert_text((72, 72), "第二章", fontsize=12, fontname=_FONT)
        pdf.new_page()  # 空白尾页：无文字、也无图
        pdf.save(path)
    return path


def _mixed_pdf(path: Path) -> Path:
    """混合型：第 1 页是扫描图（无文字层），第 2 页是数字版正文。"""
    png = _image_bytes()
    with pymupdf.open() as pdf:
        pdf.new_page().insert_image(pdf[0].rect, stream=png)
        pdf.new_page().insert_text((72, 72), "数字版正文", fontsize=12, fontname=_FONT)
        pdf.save(path)
    return path


def _cfg(parser: str = "paddleocr") -> IngestConfig:
    return IngestConfig(pdf=IngestPdfConfig(parser=parser))


def _load(source: Path, cfg: IngestConfig, ocr: object | None) -> object:
    # 假客户端不是 OcrClient 实例，但这里靠鸭子类型：只用到 .parse()
    return load_source(source, cfg, ocr)  # type: ignore[arg-type]


# ---- 默认：所有 PDF 统一走 OCR ----


def test_a_digital_pdf_also_goes_to_ocr(tmp_path: Path) -> None:
    """数字版也走 OCR，不再探测扫描页。

    pymupdf 只能拿纯文本：标题层级、表格结构、图片全丢。而下游 markdown-semantic
    切分器的第一级就是 markdown 标题，图片描述那套要的是正文里的图片标记。
    实测同一份数字版 PDF：pymupdf 路径 0/20 个块带标题路径、0 个图片标记。
    """
    target = _digital_pdf(tmp_path / "digital.pdf")
    ocr = _FakeOcr("# 产线给的标题", "产线给的正文。")

    report = _load(target, _cfg(), ocr)

    assert len(ocr.calls) == 1
    assert ocr.calls[0][1] == FILE_PDF
    doc = report.documents[0]
    assert doc.metadata["parser"] == PARSER_NAME
    # 正文来自产线，不是 pymupdf 抽出来的文字层
    assert "产线给的正文" in doc.text
    assert "数字版正文" not in doc.text


def test_the_whole_pdf_goes_in_one_call(tmp_path: Path) -> None:
    """整份而不是逐页：跨页表格合并与标题重建都要求产线一次看到全部页。"""
    target = _mixed_pdf(tmp_path / "mixed.pdf")
    ocr = _FakeOcr("# 标题", "正文。")

    report = _load(target, _cfg(), ocr)

    assert len(ocr.calls) == 1
    called_path, file_type = ocr.calls[0]
    assert called_path == target
    assert file_type == FILE_PDF
    # 页数由产线返回的页数决定
    assert report.documents[0].metadata["page_count"] == 2


def test_ocr_pages_are_joined_in_order(tmp_path: Path) -> None:
    target = _mixed_pdf(tmp_path / "mixed.pdf")

    report = _load(target, _cfg(), _FakeOcr("第一页内容", "第二页内容", "第三页内容"))

    text = report.documents[0].text
    assert text.index("第一页内容") < text.index("第二页内容") < text.index("第三页内容")


def test_pdf_without_ocr_configured_names_both_ways_out(tmp_path: Path) -> None:
    """没配 OCR 客户端就报错退出，而不是偷偷退回 pymupdf 抽个残缺正文。

    报错要同时给出两条出路：去 .env 配哪一个变量（按 parser 给具体名字）/ 显式接受有损的
    pymupdf 降级。parser 是 ocr，所以指的应该是 PADDLEOCR_BASE_URL。
    """
    target = _digital_pdf(tmp_path / "digital.pdf")

    report = _load(target, _cfg(), None)

    assert report.documents == []
    reason = report.skipped[0].reason
    assert "PADDLEOCR_BASE_URL" in reason, "要指到具体那个 .env 变量，光说「检查配置」等于没说"
    assert "pymupdf" in reason


# ---- 显式降级 ----


def test_pymupdf_fallback_is_lossy_but_works(tmp_path: Path) -> None:
    """降级路径拿得到文字，但图片一张都没有 —— 这正是它不该做默认的原因。"""
    target = _mixed_pdf(tmp_path / "mixed.pdf")
    ocr = _FakeOcr("不该被用到的内容")

    report = _load(target, _cfg("pymupdf"), ocr)

    assert ocr.calls == []
    doc = report.documents[0]
    assert doc.metadata["parser"] == "pymupdf"
    assert "数字版正文" in doc.text
    assert doc.images == []


def test_unknown_parser_value_is_rejected(tmp_path: Path) -> None:
    """枚举字段是普通 str（本仓库不用 Literal），非法值要在 build/lookup 处当场拦下。"""
    target = _digital_pdf(tmp_path / "digital.pdf")

    report = _load(target, _cfg("magic"), _FakeOcr())

    assert "magic" in report.skipped[0].reason
    assert "paddleocr" in report.skipped[0].reason


def test_the_old_ocr_name_still_works(tmp_path: Path) -> None:
    """`parser: ocr` 是 PaddleOCR 的老名字：配置归一化后走同一条路，一个字节都不差。"""
    target = _digital_pdf(tmp_path / "digital.pdf")
    ocr = _FakeOcr("# 产线给的标题", "产线给的正文。")

    report = _load(target, _cfg("ocr"), ocr)

    assert len(ocr.calls) == 1
    assert "产线给的正文" in report.documents[0].text


def test_the_old_ocr_name_normalizes_to_paddleocr() -> None:
    """归一化发生在**配置层**：指纹与界面只有一种说法。

    两种写法必须落到**同一个指纹**上（`parse_key` 里写着 parser 的名字）—— 否则同一份
    文档会因为"写法不同"被判成换了配置，白重跑一遍 OCR，那是这条链路上最慢最贵的一步。
    """
    from rag.config.schema import Settings
    from rag.ingest.stages import parse_key

    assert IngestPdfConfig(parser="ocr").parser == "paddleocr"
    assert IngestPdfConfig(parser="paddleocr").parser == "paddleocr"
    assert IngestPdfConfig(parser="magic").parser == "magic", "认不出来的原样留着，由 loader 报错"

    old = Settings(ingest=IngestConfig(pdf=IngestPdfConfig(parser="ocr")))
    new = Settings(ingest=IngestConfig(pdf=IngestPdfConfig(parser="paddleocr")))
    assert parse_key(old, ".pdf") == parse_key(new, ".pdf")


# ---- 图片 ----


def test_image_files_are_indexed_through_ocr(tmp_path: Path) -> None:
    target = tmp_path / "shot.png"
    target.write_bytes(_image_bytes())
    ocr = _FakeOcr("# 截图里的标题\n\n截图里的正文。")

    report = _load(target, _cfg(), ocr)

    assert len(ocr.calls) == 1
    assert ocr.calls[0][1] == FILE_IMAGE
    doc = report.documents[0]
    assert doc.metadata["parser"] == PARSER_NAME
    assert "截图里的正文" in doc.text


def test_image_without_ocr_is_skipped_with_a_pointer(tmp_path: Path) -> None:
    """图片没有文字层可抽，没配 OCR 就只能跳过 —— 但原因里要说清去哪配。"""
    target = tmp_path / "shot.png"
    target.write_bytes(_image_bytes())

    report = _load(target, _cfg(), None)

    assert report.documents == []
    assert "PADDLEOCR_BASE_URL" in report.skipped[0].reason


def test_image_suffixes_are_registered() -> None:
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        assert suffix in LOADERS


# ---- 注册表 ----


def test_build_loader_rejects_an_unknown_suffix() -> None:
    with pytest.raises(KeyError, match="不支持的后缀"):
        build_loader(".xyz", _cfg(), None)


def test_build_loader_returns_the_loader_for_that_suffix() -> None:
    assert build_loader(".md", _cfg(), None).name == "text"
    assert build_loader(".PDF", _cfg(), None).name == "pdf"
    assert build_loader(".png", _cfg(), None).name == "image"


def test_duplicate_suffix_registration_fails_loudly() -> None:
    """重新登记已占用的后缀必须当场报错，否则会「加了新格式，老格式悄悄改了行为」。"""
    from rag.ingest.loaders import register_loader

    with pytest.raises(ValueError, match="已被"):

        @register_loader(".pdf")
        class _Clash:
            pass


# ---- 落到 chunk 上 ----


def test_ocr_config_is_off_without_an_address() -> None:
    assert build_ocr(IngestOcrConfig()) is None
    assert build_ocr(IngestOcrConfig(base_url="   ")) is None
    assert isinstance(build_ocr(IngestOcrConfig(base_url="http://h:8080")), OcrClient)


def test_headings_reach_the_chunks_only_with_the_markdown_strategy(tmp_path: Path) -> None:
    """relevelTitles 重建出来的 #/## 只有 markdown chunker 才认。

    这条差异写成测试，免得以后有人以为标题重建白做了 —— 也记下默认值为什么
    留在 recursive（换掉会改现有语料的分块边界，那是另一个决定）。
    """
    target = tmp_path / "shot.png"
    target.write_bytes(_image_bytes())
    markdown = "# 年报\n\n总览。\n\n## 财务\n\n营收增长。"

    def _settings(strategy: str) -> Settings:
        return Settings(
            indexing=IndexingConfig(chunk=ChunkingConfig(strategy=strategy, size=400, overlap=0))
        )

    with_headings = prepare(target, _settings("markdown"), _FakeOcr(markdown))
    assert with_headings.chunks[0].heading_path == "年报"
    assert with_headings.chunks[0].heading_numbers == "1"
    assert with_headings.chunks[1].heading_path == "年报 -> 财务"
    assert with_headings.chunks[1].heading_numbers == "1-1"

    without = prepare(target, _settings("recursive"), _FakeOcr(markdown))
    assert all(not c.heading_path for c in without.chunks)
