"""PDF loader：**统一走 OCR**，具体哪一家由 ``ingest.pdf.parser`` 选（工厂决定）。

为什么不再探测「是不是扫描件」：数字版 PDF 走 pymupdf 只能拿到**纯文本** ——
没有标题层级、没有表格、没有图片。而下游 `markdown-semantic` 切分器的第一级就是
markdown 标题，图片描述那套要的是正文里的 `<img>` 标记 —— 两者在纯文本上都是空转。
实测同一份数字版 PDF：pymupdf 路径 **0/20** 个块带标题路径、0 个图片标记；
OCR 路径 **8/8**、9 个图片标记。

三家 OCR 的取舍（**换一家只改配置，这一层一行不动**，因为它们出口同一种格式）：

| `parser` | 谁 | 特点 |
|---|---|---|
| `mineru` | MinerU 远程 API（6006） | `middle_json` 逐页逐块 + bbox，**定位最规整**；不给层级、慢 |
| `dots` | dots.ocr（vLLM 6006） | **给标题层级**（`Section-header` 自带 `#`/`##`/`###`），快 |
| `ocr` | PaddleOCR 产线（8080） | 整份一次调用；`relevelTitles` 猜的层级在教材 PDF 上不可用 |

`parser: pymupdf` 保留有两个用处：OCR 都挂掉时还能出个降级结果；排查「这份是不是
OCR 读歪了」时拿它做对照。**它是有损的**（标题/表格/图片全丢），所以不是默认。
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from rag.config.schema import PDF_PARSERS, IngestConfig, normalize_pdf_parser
from rag.contracts import Document
from rag.ingest.assets import asset_store_for_cfg
from rag.ingest.loaders.base import (
    doc_id_for,
    mime_for,
    normalize_newlines,
    pages_to_text,
    register_loader,
    rel_uri,
)
from rag.ingest.ocr import FILE_PDF, OcrClient

# ingest.pdf.parser 的可选值。枚举字段是普通 str（本仓库不用 Literal），
# 非法值在这里当场拦下，报错时把人可选的都列出来。
#
#   三家 OCR 的名字与 `rag.ingest.ocr_client.OCR_ENGINES` 一一对应（由工厂决定用谁）；
#   pymupdf 不是 OCR，是本地文字层提取，属于 loader 自己的降级路径。
# **名单的唯一真相在 config.schema**（那里还带 `ocr` → `paddleocr` 的别名归一化），
# 这里只是转出个短名字给本模块与老朋友用。
PARSERS = PDF_PARSERS

# 没配客户端时，报错要直接指到该填哪个 .env 变量 —— 光说「检查配置」等于没说
_ENV_HINTS = {
    "paddleocr": "PADDLEOCR_BASE_URL",
    "dots": "DOTS_OCR_BASE_URL",
    "mineru": "MINERU_BASE_URL",
}


def _reject_encrypted(path: Path) -> None:
    """加密 PDF 先拦下来。

    不拦的话它会一路送到产线，回一个看不出所以然的错 —— 而原因其实一句话就能说清。
    """
    try:
        with pymupdf.open(path) as pdf:
            if pdf.needs_pass:
                raise ValueError("PDF 已加密，需要密码；OCR 产线读不了它")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"PDF 打开失败: {exc}") from exc


@register_loader(".pdf")
class PdfLoader:
    """PDF 整份交给产线；`parser: pymupdf` 是显式降级。"""

    name = "pdf"
    suffixes = (".pdf",)

    def __init__(
        self, cfg: IngestConfig, ocr: OcrClient | None = None, assets: object | None = None
    ) -> None:
        self._cfg = cfg
        self._ocr = ocr
        self._assets = assets

    def load(self, path: Path, *, root: Path | None = None) -> Document:
        # 归一化只在这里兜一次：配置里写 `ocr`（老名字）也当成 `paddleocr`
        mode = normalize_pdf_parser(self._cfg.pdf.parser)
        if mode not in PARSERS:
            raise ValueError(f"未知的 ingest.pdf.parser: {mode!r}；可选 {list(PARSERS)}")

        if mode == "pymupdf":
            return self._load_with_pymupdf(path, root)
        return self._load_with_engine(path, root, mode)

    def _load_with_engine(self, path: Path, root: Path | None, mode: str) -> Document:
        """把整份文件交给**工厂选出来的那个 OCR 客户端**。

        这里只有一段代码、不认识任何一家：三家 client 的出口都是
        ``list[OcrPage]``（每页 markdown + 图片），换 OCR 不用动这一层 ——
        这正是「中间产物格式保持一致」换来的。
        """
        if self._ocr is None:
            hint = _ENV_HINTS.get(mode, "")
            raise ValueError(
                f"ingest.pdf.parser={mode!r} 需要 OCR 客户端，但没配上：在 .env 里填 "
                f"{hint}（或改 ingest.{mode} 那一段配置）。"
                f"确实要只抽文字层（会丢标题/表格/图片），把 ingest.pdf.parser 设成 pymupdf。"
            )
        _reject_encrypted(path)
        pages = self._ocr.parse(path, file_type=FILE_PDF)
        doc_id = doc_id_for(path, root)
        store = self._assets if self._assets is not None else asset_store_for_cfg(self._cfg.assets)
        extracted = pages_to_text(pages, store=store, doc_id=doc_id)
        return Document(
            doc_id=doc_id,
            source_uri=rel_uri(path, root),
            mime=mime_for(path),
            text=extracted.text,
            images=extracted.images,
            positions=extracted.positions,
            metadata={
                # 客户端自己报名字（paddleocr / dots / mineru）——
                # 这个名字是已入库数据的一部分（Milvus 的 doc_type），所以由 client 定
                "parser": self._ocr.name,
                # OCR 路径下页数由客户端返回的页数决定
                "page_count": len(pages),
                "image_count": len(extracted.images),
            },
        )

    def _load_with_pymupdf(self, path: Path, root: Path | None) -> Document:
        """降级路径：只要文字层。

        **有损**：标题层级、表格结构、图片全都拿不到，正文里也不会有图片标记。
        只在产线不可用或做对照时用。
        """
        try:
            with pymupdf.open(path) as pdf:
                if pdf.needs_pass:
                    raise ValueError("PDF 已加密，需要密码")
                total = pdf.page_count
                text = normalize_newlines(
                    "\n\n".join(s for page in pdf if (s := page.get_text("text").strip()))
                )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"PDF 打开失败: {exc}") from exc

        return Document(
            doc_id=doc_id_for(path, root),
            source_uri=rel_uri(path, root),
            mime=mime_for(path),
            text=text,
            metadata={"parser": "pymupdf", "page_count": total},
        )
