"""接入层第一步：把磁盘上的语料读成 Document。

用法（上层只该用这几个函数，不要直接摸 LOADERS）：

    report = load_source(Path("./data/docs"), settings.ingest, ocr)
    report.documents    # 解析成功的文档
    report.skipped      # 跳过的文件及原因（不支持的后缀 / 解析失败 / 正文为空）

cfg 收的是 ``settings.ingest``（PDF 的解析策略 + OCR 服务地址），ocr 是建好的
OcrClient、可以为 None（没配地址就整条链路不发请求）。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from rag.config.schema import IngestConfig
from rag.ingest.loaders.base import (
    LOADERS,
    LoadReport,
    SkippedFile,
    build_loader,
    data_hash,
    decode_text,
    doc_id_for,
    image_suffixes,
    mime_for,
    normalize_newlines,
    read_text_any_encoding,
    register_loader,
    rel_uri,
    suffix_for_mime,
)
from rag.ingest.loaders.docx import DocxLoader  # noqa: F401  触发注册
from rag.ingest.loaders.html import HtmlLoader  # noqa: F401  触发注册
from rag.ingest.loaders.image import ImageLoader  # noqa: F401  触发注册
from rag.ingest.loaders.pdf import PdfLoader  # noqa: F401  触发注册
from rag.ingest.loaders.text import TextLoader  # noqa: F401  触发注册
from rag.ingest.loaders.xlsx import XlsxLoader  # noqa: F401  触发注册

if TYPE_CHECKING:
    from rag.ingest.ocr_client import OcrEngine

__all__ = [
    "LOADERS",
    "DocxLoader",
    "HtmlLoader",
    "ImageLoader",
    "LoadReport",
    "PdfLoader",
    "SkippedFile",
    "TextLoader",
    "XlsxLoader",
    "build_loader",
    "data_hash",
    "decode_text",
    "discover",
    "doc_id_for",
    "image_suffixes",
    "load_paths",
    "load_source",
    "mime_for",
    "normalize_newlines",
    "read_text_any_encoding",
    "register_loader",
    "rel_uri",
    "suffix_for_mime",
]


def _iter_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(root).parts[:-1]
        # 跳过隐藏目录：.git/.idea 里躺着成堆的 .md/.txt，
        # 谁把语料根指到项目根就会灌一库无关内容
        if any(part.startswith(".") for part in parts):
            continue
        # 跳过浏览器「保存网页」生成的 `<页面名>_files/`：里面是 JS/CSS/图标与
        # 几个跟踪用的小 HTML，不是语料。不跳过的话，那些 .png 还会被当成图片
        # 一张张送去 OCR —— 实测一份文档页能带出 4 张图标 PNG
        if any(part.endswith("_files") for part in parts):
            continue
        yield path


def discover(source: Path) -> list[Path]:
    """列出待处理的文件。目录则递归，单文件则它自己。

    这里**不**按后缀过滤：后缀是否受支持由 load_paths 判定并计入 skipped。
    若在发现阶段就滤掉，「目录里有个 .xyz 被悄悄忽略」就没人知道了 ——
    少进一个文件却毫无提示，是那种要等到检索缺内容才被发现的问题。

    按路径排序：保证多次 ingest 的处理顺序一致，日志与 chunk 顺序可复现。
    """
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(_iter_files(source), key=lambda p: p.as_posix())
    raise FileNotFoundError(f"语料路径不存在: {source}")


def load_paths(
    paths: list[Path],
    cfg: IngestConfig,
    root: Path | None = None,
    ocr: OcrEngine | None = None,
    assets: object | None = None,
) -> LoadReport:
    """逐个加载。单个文件失败只记录、不中断整批。"""
    report = LoadReport()
    for path in paths:
        uri = rel_uri(path, root)
        if path.suffix.lower() not in LOADERS:
            report.skipped.append(SkippedFile(path=uri, reason=f"不支持的后缀 {path.suffix!r}"))
            continue
        try:
            doc = build_loader(path.suffix, cfg, ocr, assets).load(path, root=root)
        except Exception as exc:
            # OCR 服务连不上、PDF 加密、扫描页没配 OCR …… 都走这里，
            # 原因原样带出去，不吞成一句「失败」
            report.skipped.append(SkippedFile(path=uri, reason=str(exc)))
            continue
        if not doc.text.strip():
            report.skipped.append(
                SkippedFile(
                    path=uri,
                    reason=(
                        "解析后正文为空。若是扫描件：确认 .env 里的 PADDLEOCR_BASE_URL 已指向"
                        "产线服务（默认端口 8080），且 ingest.pdf.parser 设的不是 pymupdf。"
                    ),
                )
            )
            continue
        report.documents.append(doc)
    return report


def load_source(
    source: Path,
    cfg: IngestConfig,
    ocr: OcrEngine | None = None,
) -> LoadReport:
    """入口：给一个目录或单个文件，返回加载报告。

    root 取「语料根目录」而不是单文件所在目录 —— 这样 source_uri 和 doc_id
    只取决于文件在语料里的相对位置，换个工作目录 ingest 结果不变。
    """
    source = Path(source).resolve()
    root = source if source.is_dir() else source.parent
    return load_paths(discover(source), cfg, root=root, ocr=ocr)
