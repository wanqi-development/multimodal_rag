"""图片 loader：图片没有文字层，只能整张交给 OCR 产线。"""

from __future__ import annotations

from pathlib import Path

from rag.config.schema import IngestConfig
from rag.contracts import Document
from rag.ingest.assets import asset_store_for_cfg
from rag.ingest.loaders.base import (
    doc_id_for,
    mime_for,
    pages_to_text,
    register_loader,
    rel_uri,
)
from rag.ingest.ocr import FILE_IMAGE
from rag.ingest.ocr_client import OcrEngine

SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@register_loader(*SUFFIXES)
class ImageLoader:
    """一张图 = 一份文档。

    OCR 是唯一出路：png/jpg 里没有文字层可抽，不像 PDF 还能用 pymupdf 碰运气。
    所以没配 OCR 服务时直接报错并在原因里指路，而不是产出一份空正文。
    多页 TIFF 由产线按页返回（fileType=1 覆盖它），页序由服务端保证，这里按序拼起来。
    """

    name = "image"
    suffixes = SUFFIXES

    def __init__(
        self, cfg: IngestConfig, ocr: OcrEngine | None = None, assets: object | None = None
    ) -> None:
        """签名见 base.build_loader 的统一约定。

        识别相关的参数全在 OCR 那边，但资产目录（图片本体落在哪）在 cfg 里，要留着。
        """
        self._cfg = cfg
        self._ocr = ocr
        self._assets = assets

    def load(self, path: Path, *, root: Path | None = None) -> Document:
        if self._ocr is None:
            raise ValueError(
                "图片没有文字层可抽，必须走 OCR：在 .env 里配一种 OCR 服务"
                "（PADDLEOCR_BASE_URL / DOTS_OCR_BASE_URL / MINERU_BASE_URL 任一），"
                "并让 ingest.pdf.parser 指向它。"
            )
        pages = self._ocr.parse(path, file_type=FILE_IMAGE)
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
                "parser": self._ocr.name,
                "page_count": len(pages),
                "image_count": len(extracted.images),
            },
        )
