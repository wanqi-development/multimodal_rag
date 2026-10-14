"""OCR 产线服务集成测试：要一台起了产线服务的 GPU 主机，默认不跑。

    RAG_TEST_OCR=1 python -m pytest tests/test_ocr_service.py

地址跟 CLI 走同一份配置（config/base.yaml 的 ${PADDLEOCR_BASE_URL} + .env），
不另开一个环境变量，免得「CLI 连的是 A、测试连的是 B」。

为什么必须用真的：这里要验的是「产线能不能把一张扫描图读成 markdown」，
这整件事都在服务端。把 HTTP 换成替身（见 test_ocr_client.py）只能验我们发出去的
请求形状，验不了模型认不认得这些字。

真语料由 pymupdf 现画再渲染成位图 —— 位图进 PDF，就没有文字层可抽，
是一份真的扫描件，而不是「假装是扫描件的文字 PDF」。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import httpx
import pymupdf
import pytest

from rag.config import load_settings
from rag.ingest.ocr import FILE_PDF, build_ocr

pytestmark = pytest.mark.skipif(
    os.environ.get("RAG_TEST_OCR") != "1",
    reason="集成测试：需要真的 PaddleOCR-VL 产线服务，设 RAG_TEST_OCR=1 才跑",
)

# 插中文必须用内建中文字体，默认字体写进去是乱码
_FONT = "china-s"
BASE_URL = (load_settings().ingest.ocr.base_url or "").rstrip("/")


def _font_for(text: str) -> str:
    """中文用内建中文字体，纯拉丁用拉丁字体。

    别拿 ``china-s`` 去画 ``bge-small`` —— 它会按全角字形渲染，产线读回来是
    ``b g e - s m a l``（实测，且提高 DPI 也救不回来）。那是测试图自己的问题，
    不是产线的能力问题：换成 ``helv`` 后单元格值全对。
    """
    return _FONT if any("\u4e00" <= ch <= "\u9fff" for ch in text) else "helv"


def _service_up() -> bool:
    """只要能连通就算在：/docs 返回 404 也说明有东西在那个端口上听着。"""
    try:
        httpx.get(f"{BASE_URL}/docs", timeout=5)
    except httpx.HTTPError:
        return False
    return True


@pytest.fixture(scope="module", autouse=True)
def _require_service() -> None:
    if not BASE_URL:
        pytest.skip("没配 PADDLEOCR_BASE_URL（.env）：产线服务地址未知")
    if not _service_up():
        pytest.skip(f"连不上 OCR 产线服务：{BASE_URL}")


def _scan(path: Path, draw: Callable[[pymupdf.Page], None], *, dpi: int = 150) -> Path:
    """把一页「画」出来 → 渲染成位图 → 塞进新 PDF：得到一份没有文字层的扫描件。"""
    with pymupdf.open() as src:
        page = src.new_page()
        draw(page)
        png = page.get_pixmap(dpi=dpi).tobytes("png")

    with pymupdf.open() as scan:
        scan.new_page().insert_image(scan[0].rect, stream=png)
        scan.save(path)
    return path


def _scan_pdf(tmp_path: Path) -> Path:
    def draw(page: pymupdf.Page) -> None:
        page.insert_text((72, 140), "向量检索", fontsize=40, fontname=_FONT)
        page.insert_text((72, 220), "Milvus 支持稀疏向量与稠密向量。", fontsize=20, fontname=_FONT)

    return _scan(tmp_path / "scan.pdf", draw)


def _table_scan(tmp_path: Path) -> Path:
    rows = (("模型", "维度"), ("bge-small", "512"), ("bge-large", "1024"))

    def draw(page: pymupdf.Page) -> None:
        left, top, width, height = 72.0, 140.0, 260.0, 40.0
        for r in range(len(rows) + 1):
            y = top + r * height
            page.draw_line((left, y), (left + width, y), width=1)
        for c in range(3):
            x = left + c * (width / 2)
            page.draw_line((x, top), (x, top + len(rows) * height), width=1)
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                page.insert_text(
                    (left + c * (width / 2) + 10, top + r * height + 26),
                    cell,
                    fontsize=16,
                    fontname=_font_for(cell),
                )

    return _scan(tmp_path / "table.pdf", draw)


def _has_cjk(text: str) -> bool:
    return any("一" <= char <= "鿿" for char in text)


def test_scanned_pdf_comes_back_as_chinese_markdown(tmp_path: Path) -> None:
    client = build_ocr(load_settings().ingest.ocr)
    assert client is not None  # 上面已经确认配了地址

    pages = client.parse(_scan_pdf(tmp_path), file_type=FILE_PDF)

    assert len(pages) == 1
    text = pages[0].markdown
    assert text.strip(), "产线返回了空 markdown，等于什么都没读到"
    assert _has_cjk(text), f"markdown 里一个中文都没有，八成是识图没生效：{text!r}"


def test_a_table_page_comes_back_as_a_usable_table(tmp_path: Path) -> None:
    """表格还原成可解析的形态，是「这版产线值不值得接」的关键一条 ——
    RAG 里表格拍成一行散字基本等于查不到。

    不接受「必须是 `|` 语法」：实测产线给的是**内联 HTML**（`<table border=1 …><tr><td …>`），
    不是 markdown 竖线，按 `|` 断言会在真服务上失败。两种形态都算过，但**单元格的值必须在** ——
    那才是检索真正要命中的东西，光有表格语法而内容是糊的等于没接。
    """
    client = build_ocr(load_settings().ingest.ocr)
    assert client is not None

    pages = client.parse(_table_scan(tmp_path), file_type=FILE_PDF)

    text = "\n".join(page.markdown for page in pages)
    assert "<table" in text or "|" in text, f"表格没还原成任何可解析的形态：{text!r}"
    for cell in ("模型", "维度", "bge-small", "512", "bge-large", "1024"):
        assert cell in text, f"表格里找不到单元格 {cell!r}，等于查不到：{text!r}"
