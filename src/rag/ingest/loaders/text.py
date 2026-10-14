"""纯文本类文件的 loader。

**markdown 里的本地图片会换成 `asset:` 标记**（2026-10-10 加）：网页交给 Firecrawl
之后落盘件是 `.md`，图片由 `ingest/fetch.py` 下到同目录的 `<真名>-<哈希>_files/` 下并
改写成相对路径。不换成标记的话，图就**只在盘上** —— assets → VLM 描述 → 多模态 embedding
那条线全断，等于切到 Firecrawl 之后图用不上。出口与 html / docx / OCR 三条路完全一致。

取不到的图只记账（`metadata["images_missing"]`）、不废掉整份文档 —— 与 html 那条同一个
立场：外链、图标、坏路径都算常态，一张图不该让整份文档 skipped。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

from rag.config.schema import IngestConfig
from rag.contracts import Document, ImageAsset
from rag.ingest.assets import asset_store_for_cfg, make_marker
from rag.ingest.cleaning import clean_markdown
from rag.ingest.loaders.base import (
    doc_id_for,
    image_suffixes,
    mime_for,
    normalize_newlines,
    read_text_any_encoding,
    register_loader,
    rel_uri,
)
from rag.ingest.outline import first_title

if TYPE_CHECKING:
    from rag.ingest.ocr import OcrClient

SUFFIXES = (".txt", ".log", ".md", ".markdown", ".rst", ".csv")

# 只有 markdown 认图片语法（`.txt` / `.csv` 里出现 `![x](y)` 多半是数据本身，别动它）
_MARKDOWN_SUFFIXES = (".md", ".markdown")

# 整段 `![alt](相对路径 "标题")`，允许尖括号与可选标题。**只认本地相对路径**：外链本机
# 读不到、内联图（data URI）不值得为它落一份资产 —— 理由与 `html._Images` 完全一样。
# **连 `![alt](` 和 `)` 一起换掉**：留着它们会拼出 `![图](![image](asset:…))` —— 既难看，
# 又会让「空壳块」判不出来（`pipeline.is_shell_chunk` 剥掉标记后还剩 `![图]()`）。
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)>?\s*(?:\"[^\"]*\")?\s*\)")


def _local_images_to_markers(
    text: str, *, path: Path, root: Path | None, cfg: IngestConfig, assets: object | None
) -> tuple[str, list[ImageAsset], list[str]]:
    """markdown 里的本地图片 → `asset:` 标记 + 图片落 assets。

    返回 ``(新的正文, 图片清单, 没拿到图的原因)``。同一张图出现两次只落一份资产、
    复用同一个标记（多落一份就等于同一张图被 VLM 描述两遍，那是花钱的）。
    """
    store = assets if assets is not None else asset_store_for_cfg(cfg.assets)
    doc_id = doc_id_for(path, root)
    base = path.parent.resolve()
    known: dict[str, str] = {}
    images: list[ImageAsset] = []
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        ref = match.group(1)
        parts = urlsplit(ref)
        if parts.scheme or ref.startswith("//"):
            # 外链 / data: / 其它协议：这条路上不该有（fetch 已经把图下到本地了）。
            # 真遇到就原样留着 —— 删掉它等于悄悄丢一张图。
            return match.group(0)
        suffix = Path(parts.path).suffix.lower()
        if suffix not in image_suffixes():
            return match.group(0)
        cached = known.get(ref)
        if cached is not None:
            return cached
        target = (base / unquote(parts.path)).resolve()
        if target != base and base not in target.parents:
            # markdown 是不可信的输入，`../../` 这种别让它读到语料目录外面去
            missing.append(f"{ref}：路径跑到这份文档所在目录外面了")
            return match.group(0)
        if not target.is_file():
            missing.append(f"{ref}：本地没有这个文件")
            return match.group(0)
        index = len(images) + 1
        uri = store.save(target.read_bytes(), doc_id=doc_id, name=f"img{index:02d}{suffix}")
        marker = make_marker(uri)
        known[ref] = marker
        images.append(ImageAsset(marker=marker, asset_uri=uri, source_name=ref))
        return marker

    return _MD_IMAGE_RE.sub(replace, text), images, missing


@register_loader(*SUFFIXES)
class TextLoader:
    """整篇读进 text，正文**不做任何加工** —— 唯一例外是 markdown 里的本地图片
    换成 `asset:` 标记（见模块说明）。

    「不做加工」是有意的：markdown 的标题层级交给 chunker 处理（那里才知道切块策略），
    csv 的结构化解析属于 parser 层。这里只负责「把字节变成正确的字符串」这一件事，
    其中编码嗅探是中文语料的关键。
    """

    name = "text"
    suffixes = SUFFIXES

    def __init__(
        self, cfg: IngestConfig, ocr: OcrClient | None = None, assets: object | None = None
    ) -> None:
        """签名见 base.build_loader 的统一约定：文本文件用不上 OCR，图片要落 `assets`。"""
        self._cfg = cfg
        self._assets = assets

    def load(self, path: Path, *, root: Path | None = None) -> Document:
        text = normalize_newlines(read_text_any_encoding(path))
        metadata: dict[str, object] = {"parser": self.name}
        images: list[ImageAsset] = []
        if path.suffix.lower() in _MARKDOWN_SUFFIXES:
            # 入库前清洗：去不可见字符、去标题锚点（`[](url)`）、收标题行多余空格。
            # 先清洗再换图片标记 —— 否则空 alt 的图片（`![](x.png)`）会被当成空链接删掉。
            text = clean_markdown(text)
            text, images, missing = _local_images_to_markers(
                text, path=path, root=root, cfg=self._cfg, assets=self._assets
            )
            if images:
                metadata["image_count"] = len(images)
            if missing:
                metadata["images_missing"] = missing
            if title := first_title(text):
                metadata["title"] = title
        return Document(
            doc_id=doc_id_for(path, root),
            source_uri=rel_uri(path, root),
            mime=mime_for(path),
            text=text,
            images=images,
            # 与 docx / html / pdf / image 保持一致：**页面上要显示「谁解析的」**。
            # 这条路上没有别的元数据，但空着的话解析面板那一格是空的。
            metadata=metadata,
        )
