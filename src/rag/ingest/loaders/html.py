"""HTML loader：BeautifulSoup 解出标题、正文、列表、表格与图片。

和 docx 同一个道理 —— HTML 的标签就是结构，明明白白写着 ``<h1>`` / ``<table>``，
不该送去识别。出口同样是 markdown，与 OCR / docx 两条路对齐。

用标准库的 ``html.parser`` 作后端，不依赖 lxml：解析器的选择会影响容错程度，
但这里要的是「别崩、尽量读出来」，标准库够用，也少一个编译型依赖。

**图片**（2026-10-07 加）：`<img>` 换成 ``![image](asset:…)`` 标记，图片本体进
``ingest.assets``，与 docx / OCR 两条路**同一套出口** —— 下游的图片描述（VLM）与
多模态 embedding 不必分来源。

图片从哪来：**HTML 文件所在目录**（相对路径）。抓链接那份 HTML 里的 ``src`` 在
``ingest/fetch.py`` 里就被改写成这种相对形式了（下到同名 ``_files/`` 目录），
所以这里只从磁盘读，不联网 —— loader 一旦联网，``rag ingest ./data/docs`` 就会意外出网。

取不到的图**只记账、不废掉整页**：网页里混着图标、徽标、外链图，一张 404 不该让整份文档
skipped（与 docx / OCR 那条「响应里本该有这张图」的严格报错不同）。原因进
``Document.metadata["images_missing"]``，日志与产物里都看得到。
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

from bs4 import BeautifulSoup, Tag

from rag.config.schema import IngestConfig
from rag.contracts import Document, ImageAsset
from rag.ingest.assets import asset_store_for_cfg, make_marker
from rag.ingest.cleaning import strip_invisible
from rag.ingest.loaders.base import (
    decode_text,
    doc_id_for,
    image_suffixes,
    mime_for,
    normalize_newlines,
    register_loader,
    rel_uri,
    to_markdown_table,
)
from rag.ingest.outline import first_title

SUFFIXES = (".html", ".htm")

# 这些块里要么是脚本样式，要么是站点导航 —— 灌进库只会变成噪声命中
_DROP = ("script", "style", "noscript", "iframe", "svg", "nav", "footer", "form")

# 按文档顺序取这些块。
#   - div/section/article 是**兜底容器**：只在它内部没有任何块级元素时才取它的文字，
#     否则现代前端里「只有 div 没有 p」的段落会整段丢掉（实测撞到过）
#   - pre 单独成块：代码里的换行与缩进是有意义的，不能被压平
_BLOCKS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "table")
_CONTAINERS = ("div", "section", "article")
_LEAF_CONTAINER_CHECK = _BLOCKS + _CONTAINERS
# 图片跟着块一起按文档顺序遍历（`find_all` 出来的就是文档顺序）
_IMAGES = ("img",)

# 一张图的地址可能藏在这几个属性里。**懒加载那三个排在 `src` 前面**：这类站点往 `src`
# 里塞的是 1×1 的占位图（或 data URI），真图在 `data-src` 里 —— 按 `src` 先看就会
# 抓到占位图。实测 Spring / Kafka 那几个文档站用的是普通 `src`（相对路径），也照样命中。
IMG_SRC_ATTRS = ("data-src", "data-original", "data-lazy-src", "src")
SRCSET = "srcset"


# 标题尾部挂着的**装饰符号**：Go 文档站（pkg.go.dev 那一类）每个标题后面一个 `¶` 锚点，
# 折叠面板还有 `▹`/`▾` 箭头。它们是导航控件，不是标题的一部分 —— 不清掉就会进
# heading_path（实测 313 个标题里 307 个带 `¶`），章节名就没法精确匹配了。
_TITLE_DECORATION = re.compile(r"[\u00b6\u25b8\u25b9\u25be\u25c2\u2039\u203a]")


def _text(node: object) -> str:
    """取元素的可见文字。

    **不用 ``get_text(" ")``**：它会在每个行内标签边界插一个空格，中文会被拆坏 ——
    ``注意：<strong>不要</strong>把`` 会变成 ``注意： 不要 把``，「不要把」就检索不到了。
    用空分隔符保留原文里本来就有的空白，再把连续空白压成一个空格（HTML 的渲染语义就是这样）。
    """
    raw = strip_invisible(node.get_text(""))  # type: ignore[attr-defined]
    return re.sub(r"\s+", " ", raw).strip()


def _title(node: object) -> str:
    """标题文字：在 ``_text`` 的基础上再剥掉锚点/折叠箭头这类装饰符号。"""
    return _TITLE_DECORATION.sub("", _text(node)).strip()


def _content_areas(soup: BeautifulSoup) -> list[Tag]:
    """正文区：优先 ``<main>``，其次所有 ``<article>``，都没有才退回 ``<body>``。

    真实网页的页头、搜索框、面包屑、侧边导航常占去一半文字量 —— 实测 Confluent 那两篇，
    body 里 4,882 字符中正文只占 2,778。整页扫会把这些一并灌进库，变成检索噪声。
    """
    for selector in ("main", "article"):
        areas = soup.find_all(selector)
        if areas:
            return areas
    return [soup.body] if soup.body else [soup]


def img_source(node: Tag) -> str:
    """一张图的地址：先看 `src`，再看懒加载那几个属性，最后取 `srcset` 里的第一个。

    公开（不带下划线）是因为 `ingest/fetch.py` 改写成相对路径时要用**同一套**判定 ——
    两边各认一套，就会出现「抓了但改写不到」这种对不上的情况。
    """
    for attr in IMG_SRC_ATTRS:
        value = node.get(attr)
        if isinstance(value, str) and value.strip():
            return value.strip()
    srcset = node.get(SRCSET)
    if isinstance(srcset, str) and srcset.strip():
        return srcset.split(",")[0].strip().split(" ")[0]
    return ""


def content_image_urls(text: str, base: str) -> list[str]:
    """正文区里引用的图片地址（绝对化、按文档顺序去重）。

    给 `ingest/fetch.py` 用：抓页面时照着这个清单把图一起下下来。判定「哪些图算正文」
    的规则**与 loader 完全一样**（同一套 `_content_areas` + `_DROP`），否则抓了一堆
    侧边栏图标、或者正文里的图漏抓，两边对不上。
    """
    soup = BeautifulSoup(text, "html.parser")
    out: list[str] = []
    seen: set[str] = set()
    for area in _content_areas(soup):
        # 先清噪声再找图：导航/页脚里的图标不是正文的一部分
        for tag in area.find_all(_DROP):
            tag.decompose()
        for node in area.find_all(_IMAGES):
            source = img_source(node)
            if not source:
                continue
            resolved = urljoin(base, source)
            if resolved not in seen:
                seen.add(resolved)
                out.append(resolved)
    return out


class _Images:
    """把 `<img>` 换成正文标记：图片本体落 assets、正文留短标记（`assets.make_marker`）。

    同一张图出现两次（页头页脚各一个徽标之类）只落一份资产、复用同一个标记 ——
    下游按标记建图片块，重复落盘只会让同一张图被描述两遍（那是花钱的）。
    """

    def __init__(
        self, cfg: IngestConfig, *, doc_id: str, base: Path, assets: object | None = None
    ) -> None:
        # 调用方（run_parse）会按完整配置建好传进来（可能是 MinIO）；没传就按 ingest 那段配置
        # 建本地那份 —— 只有 backend=local 才允许这条兜底路。
        self._store = assets if assets is not None else asset_store_for_cfg(cfg.assets)
        self._doc_id = doc_id
        self._base = base
        self._markers: dict[str, str] = {}
        self.images: list[ImageAsset] = []
        self.missing: list[str] = []

    def marker_for(self, node: Tag) -> str:
        """这个 `<img>` 对应的标记；拿不到图就返回空串（原因记进 `missing`）。"""
        source = img_source(node)
        if not source:
            self._note("有一个 <img> 既没有 src 也没有 srcset")
            return ""
        known = self._markers.get(source)
        if known is not None:
            return known
        marker = self._load(source)
        if marker:
            self._markers[source] = marker
        return marker

    def _load(self, source: str) -> str:
        parts = urlsplit(source)
        if parts.scheme == "data":
            # 内联图片（通常是图标/占位图）：没有独立文件可读，也不值得为它落一份资产
            self._note(f"{source[:40]}…：内联图片（data URI）")
            return ""
        if parts.scheme in ("http", "https"):
            # 抓取时本该把图下到本地（ingest/fetch.py）；还挂着外链说明这份 HTML 不是我们
            # 抓的（自己另存的页面），或者抓的时候这张图没下下来
            self._note(f"{source}：还是外链，本地没有副本")
            return ""
        if parts.scheme:
            self._note(f"{source}：不支持这种地址（{parts.scheme}）")
            return ""
        suffix = Path(parts.path).suffix.lower()
        if suffix not in image_suffixes():
            self._note(f"{source}：不是能进库的图片格式（{suffix or '没有后缀'}）")
            return ""
        path = (self._base / unquote(parts.path)).resolve()
        if not path.is_file():
            self._note(f"{source}：本地没有这个文件")
            return ""
        if self._base.resolve() not in path.parents:
            # HTML 是不可信的输入，`../../` 这种别让它读到语料目录外面去
            self._note(f"{source}：路径跑到 HTML 所在目录外面了")
            return ""
        index = len(self.images) + 1
        uri = self._store.save(
            path.read_bytes(), doc_id=self._doc_id, name=f"img{index:02d}{suffix}"
        )
        marker = make_marker(uri)
        self.images.append(ImageAsset(marker=marker, asset_uri=uri, source_name=source))
        return marker

    def _note(self, reason: str) -> None:
        if reason not in self.missing:
            self.missing.append(reason)


@register_loader(*SUFFIXES)
class HtmlLoader:
    name = "html"
    suffixes = SUFFIXES

    def __init__(
        self, cfg: IngestConfig, ocr: object | None = None, assets: object | None = None
    ) -> None:
        """签名见 base.build_loader 的统一约定：这条路上用不到 OCR，图片要落到 `assets`。"""
        self._cfg = cfg
        self._assets = assets

    def load(self, path: Path, *, root: Path | None = None) -> Document:
        soup = BeautifulSoup(decode_text(path.read_bytes()), "html.parser")
        doc_id = doc_id_for(path, root)
        images = _Images(self._cfg, doc_id=doc_id, base=path.parent, assets=self._assets)

        lines: list[str] = []
        for area in _content_areas(soup):
            # 先选内容区、再在区内清噪声：反过来的话，套着正文的 <form> 会把正文一起干掉
            for tag in area.find_all(_DROP):
                tag.decompose()
            lines.extend(self._blocks(area, images))

        body = normalize_newlines("\n\n".join(lines))
        metadata: dict[str, object] = {"parser": self.name, "image_count": len(images.images)}
        # 文档标题：第一个一级标题（**跳过代码围栏** —— 代码样例里的 `# 注释` 不是标题）。
        # 与 text loader 同一个字段名 —— 列表/引用将来要显示文档标题时不用分来源。
        if title := first_title(body):
            metadata["title"] = title
        if images.missing:
            metadata["images_missing"] = list(images.missing)
        return Document(
            doc_id=doc_id,
            source_uri=rel_uri(path, root),
            mime=mime_for(path),
            # 入口处归一化换行：代码块里的 \r\n 会原样进库、出现在引用片段里
            text=body,
            images=images.images,
            metadata=metadata,
        )

    def _blocks(self, area: Tag, images: _Images) -> list[str]:
        out: list[str] = []
        for node in area.find_all(_BLOCKS + _CONTAINERS + _IMAGES):
            if node.name == "img":
                if marker := images.marker_for(node):
                    out.append(marker)
                continue
            # 表格的单元格也是 <p>/<li> 的话会被重复取一遍，跳过表内的
            if node.name != "table" and node.find_parent("table") is not None:
                continue
            # 嵌套 <li> 会被取两遍：外层 li 的 get_text 已经把子项文字吞进去了，
            # 子项再单独出一遍就是同一段文字进库三次（检索里就是三次噪声命中）
            if node.name == "li" and node.find_parent("li") is not None:
                continue
            # li 里套 p 同理：li 的文字已经含了那个 p
            if node.name == "p" and node.find_parent("li") is not None:
                continue

            if node.name == "table":
                rows = [
                    [_text(cell) for cell in row.find_all(["td", "th"])]
                    for row in node.find_all("tr")
                ]
                table = to_markdown_table([r for r in rows if r])
                if table:
                    out.append(table)
                continue

            # 兜底容器只在「内部没有任何块级元素」时才算一段 ——
            # 否则 <div><p>…</p></div> 会和里面的 <p> 重复取一遍
            if node.name in _CONTAINERS and node.find(_LEAF_CONTAINER_CHECK) is not None:
                continue

            if node.name == "pre":
                code = strip_invisible(node.get_text()).strip("\n")
                if code:
                    # 代码块用围栏：里面的换行和缩进是有意义的
                    out.append(f"```\n{code}\n```")
                continue

            if node.name.startswith("h") and len(node.name) == 2 and node.name[1].isdigit():
                text = _title(node)
                if not text:
                    continue
                out.append(f"{'#' * int(node.name[1])} {text}")
                continue
            text = _text(node)
            if not text:
                continue
            out.append(f"- {text}" if node.name == "li" else text)
        return out
