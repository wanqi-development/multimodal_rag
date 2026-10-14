"""图片资产：落盘、正文标记、上下文。

两条 OCR 路都把图片内联在 markdown 里，而且**夹在前后段落之间** —— 位置本身就是信息：

* PaddleOCR 产线：``<div><img src="imgs/…"></div>``，图片数据在响应的另一个字段里；
* dots.ocr：``![](data:image/png;base64,…)``，裁剪好的图就内联在正文里。

两条路在这里汇合，出口是同一套短标记，下游不用分来源。这里做三件事：

1. **把 base64 从正文里赶出去**：图片写进资产目录，正文里换成短标记。
   一张图的 base64 有几十万字符，留在正文里会把 chunk 预算整个吃掉。
2. **标记留在文本流里的原位**：切块之后，谁含哪个标记是一目了然的，图片的位置信息没丢。
3. **按标记取「上边一段 / 下边一段」**：图片描述要带上下文，光有图不够
   （一张架构图脱离上下文根本说不清在讲什么）。

标记长这样：``![image](asset:<doc_id>/p0003_01.jpg)`` —— 里面的相对 uri 直接就是
资产目录下的路径。换成对象存储时，改的是 ``AssetStore``，正文里的标记不用动。
"""

from __future__ import annotations

import base64
import bisect
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from rag.config.schema import IngestAssetsConfig
from rag.contracts import Chunk, ImageAsset, Position
from rag.storage import BlobStore

if TYPE_CHECKING:  # 只为标注：`Settings` 在运行期不必真 import
    from rag.config.schema import Settings

# 正文里的标记。alt 文本不参与匹配 —— 将来改文案不该让解析失效。
MARKER_RE = re.compile(r"!\[[^\]]*\]\(asset:([^)]+)\)")

# 产线输出的是一个居中的 <div> 包着 <img>；也有裸 <img> 的情况，两种都吃。
_IMG_BLOCK_RE = re.compile(r"<div[^>]*>\s*<img\b[^>]*>\s*</div>|<img\b[^>]*?/?>", re.IGNORECASE)
_SRC_RE = re.compile(r"""src\s*=\s*["']([^"']+)["']""", re.IGNORECASE)

# dots.ocr 把裁剪好的图内联成 markdown 的 data URI：`![](data:image/png;base64,<很长的串>)`
_DATA_URI_RE = re.compile(r"!\[[^\]]*\]\(data:image/([a-zA-Z0-9.+-]+);base64,([^)]+)\)")

_SUFFIX_BY_MIME = {
    "png": ".png",
    "jpeg": ".jpg",
    "jpg": ".jpg",
    "webp": ".webp",
    "bmp": ".bmp",
    "tiff": ".tiff",
}


def make_marker(asset_uri: str) -> str:
    return f"![image](asset:{asset_uri})"


def markers_in(text: str) -> list[str]:
    """正文里出现的所有标记，按出现顺序。"""
    return [match.group(0) for match in MARKER_RE.finditer(text)]


def asset_uris_in(text: str) -> list[str]:
    """正文里引用到的资产 uri，按出现顺序去重。

    用途是把「这一块里有哪几张图」变成结构化字段 —— 否则只能靠正则扫正文，
    追溯不了。见 docs/milvus-schema-design.md。
    """
    seen: dict[str, None] = {}
    for match in MARKER_RE.finditer(text):
        seen.setdefault(match.group(1), None)
    return list(seen)


class AssetStore:
    """图片本体的读写。**本地盘与 MinIO 是同一个类**，差别只在 `storage.BlobStore`
    （2026-10-09 起；单机默认走本地盘，行为与以前逐字节一致）。
    """

    def __init__(self, blobs: BlobStore) -> None:
        self._blobs = blobs

    @property
    def backend(self) -> str:
        return self._blobs.backend

    @property
    def name(self) -> str:
        return self._blobs.name

    def save(self, data: bytes, *, doc_id: str, name: str) -> str:
        """写入并返回**相对 uri**（进正文标记的就是它）。

        按 doc_id 分目录、按页与序号命名：同一份文档重灌是原地覆盖。
        代价是文档改短后多出来的旧图片会留在盘上成为孤儿 —— 与 Milvus 那边
        「先按 doc_id 删旧块」不同，这里没有清理动作，先记着（见 §21）。
        """
        relative = f"{doc_id}/{name}"
        self._blobs.put(relative, data)
        return relative

    def read(self, asset_uri: str) -> bytes:
        return self._blobs.get(asset_uri)

    def exists(self, asset_uri: str) -> bool:
        return self._blobs.exists(asset_uri)

    def path_of(self, asset_uri: str) -> Path | None:
        """**只认 Path 的消费方**（VLM 读图、多模态 embedding 送图）用这个。

        local：就是那个真路径。minio：读穿透到本地缓存后给缓存路径 —— 桶是真相，
        缓存删掉只会让它下次再下一次，不影响正确性。
        **这份资产不在 → None**（两个后端一个说法，见 `storage.local_path`）。
        """
        return self._blobs.local_path(asset_uri)


def build_asset_store(settings: Settings) -> AssetStore:
    """按 `ingest.assets.backend` 建：`auto`（默认，听统一开关 `storage.backend`）→ 本地盘或桶。"""
    from rag.storage import build_blob_store

    cfg = settings.ingest.assets
    return AssetStore(
        build_blob_store(backend=cfg.backend, settings=settings, dir=cfg.dir, prefix=cfg.prefix)
    )


def asset_store_for_cfg(cfg: IngestAssetsConfig) -> AssetStore:
    """**只给"手上只有 `ingest` 那一段配置"的调用方兜底**（html / docx 的 direct 构造）。

    这里**看不到统一开关** `storage.backend`（手上只有这一小段配置），所以 `backend=auto`
    按默认的 `local` 处理；要跟随统一开关就得由调用方传一个按 settings 建好的 AssetStore
    （正常入口 `run_parse` / `load_paths` 都是这么传的，见 `build_asset_store`）。

    `backend=minio` 时直接报错而不是悄悄写本地盘 —— 那种"看起来成了、文件却不在该在的地方"
    最难查。
    """
    from rag.storage import BlobError, LocalBlobStore, resolve_storage_backend

    try:
        # settings=None：这一层拿不到统一开关，auto 按默认的 local 解析；显式值照旧
        backend = resolve_storage_backend(None, getattr(cfg, "backend", "auto"))
    except BlobError as exc:
        raise ValueError(str(exc)) from exc
    if backend != "local":
        raise ValueError(
            "这个入口只按 ingest 配置建本地资产目录；backend="
            f"{cfg.backend!r} 时要由调用方传一个按 settings 建好的 AssetStore"
            "（见 ingest.assets.build_asset_store）"
        )
    return AssetStore(LocalBlobStore(cfg.dir))


class ExtractedText(NamedTuple):
    """一页/一份文档清洗后的正文，外加它引用到的图片与来源位置。"""

    text: str
    images: list[ImageAsset]
    # 文本偏移 → 来源位置（页码/行号）。loader 在拼接正文时打点，切块后靠它反查
    positions: list[Position] = []


def extract_images(
    markdown: str,
    images: Mapping[str, bytes],
    *,
    store: AssetStore,
    doc_id: str,
    page: int,
) -> ExtractedText:
    """把正文里的图片换成标记、图片落盘。**html 的 ``<img>`` 与 markdown 的
    data URI 都吃**（前者来自 PaddleOCR 产线，后者来自 dots.ocr）。"""
    found: list[ImageAsset] = []
    counter = 0

    def save_one(data: bytes, suffix: str, source_name: str) -> str:
        nonlocal counter
        counter += 1
        name = f"p{page:04d}_{counter:02d}{suffix}"
        uri = store.save(data, doc_id=doc_id, name=name)
        marker = make_marker(uri)
        found.append(ImageAsset(marker=marker, asset_uri=uri, page=page, source_name=source_name))
        return marker

    def replace_html(match: re.Match[str]) -> str:
        src_match = _SRC_RE.search(match.group(0))
        if src_match is None:
            return match.group(0)
        src = src_match.group(1)
        data = images.get(src)
        if data is None:
            # 不静默丢图：调用方会把这个文件计入 skipped 并说明原因，
            # 而不是让一份「少了几张图」的文档悄悄进库。
            raise ValueError(
                f"第 {page} 页引用了图片 {src!r}，但响应里没有它的数据。"
                f"检查 ingest.ocr.return_images 是否为 true。"
            )
        return save_one(data, Path(src).suffix or ".jpg", src)

    def replace_data_uri(match: re.Match[str]) -> str:
        mime = match.group(1).lower()
        try:
            # data URI 里的 base64 可能带换行（长串会被折行），先去掉再解
            data = base64.b64decode("".join(match.group(2).split()))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"第 {page} 页有一张内联图片的 base64 解不开：{exc}") from exc
        return save_one(data, _SUFFIX_BY_MIME.get(mime, ".png"), f"data:image/{mime}")

    text = _IMG_BLOCK_RE.sub(replace_html, markdown)
    return ExtractedText(_DATA_URI_RE.sub(replace_data_uri, text), found)


def line_positions(text: str, *, page: int, start: int) -> list[Position]:
    """给一段文本**逐行**打点：行首偏移 → ``(page, 页内第几行)``。

    为什么逐行而不是只记页首：要回答的是「在第几页**第几行**」，只记页首就只能给出页码。
    行数按 ``\\n`` 数，与 markdown 的行一致。
    """
    out = [Position(offset=start, page=page, line=1)]
    for index, char in enumerate(text):
        if char == "\n":
            out.append(Position(offset=start + index + 1, page=page, line=len(out) + 1))
    return out


def position_at(positions: Sequence[Position], offsets: Sequence[int], offset: int) -> Position:
    """二分找出「最后一个 offset <= 目标的点」—— 偏移 → (页码, 行号)。

    ``offsets`` 是 ``positions`` 里各项的 offset（调用方一次算好、反复用）：
    这里会被逐块/逐图调用，别在热路径上重复建列表。
    """
    index = bisect.bisect_right(offsets, offset) - 1
    return positions[max(index, 0)]


def context_for_marker(chunks: Sequence[Chunk], marker: str) -> tuple[str, str]:
    """一张图的「上边一段 / 下边一段」。

    取的是**段落**不是整块：块有 400 字符，一个段落才是「这段话在讲什么」的自然粒度。

    两处兜底，都是实测撞出来的：

    1. **标记正好落在块首/块尾** —— 同一侧取不到东西，向相邻块继续找。边界把上下文
       切断是常事，不兜住的话图片描述会凭空少掉一半信息。
    2. **连续整页图** —— 两张图之间本来就没有文字，那一侧拿到的会是**另一个图片标记**。
       标记不是段落，塞给模型等于没给。所以把标记剔掉再判断，一直找到真正的正文为止。
    """
    for index, chunk in enumerate(chunks):
        position = chunk.text.find(marker)
        if position < 0:
            continue
        return (
            _nearest_paragraph(chunks, index, chunk.text[:position], step=-1),
            _nearest_paragraph(chunks, index, chunk.text[position + len(marker) :], step=1),
        )
    raise KeyError(f"没有任何 chunk 含这个标记：{marker}")


def image_paths_for(
    chunks: Sequence[Chunk], store: AssetStore
) -> tuple[list[list[Path]], list[str]]:
    """每条 chunk 对应的**图片本体**路径（没有图就是空列表），外加盘上找不到的 uri。

    给多模态 embedding 用：图片块（``modality=image``）要把「图 + 描述文字」一起送进
    同一个内容项，服务端融合成**一个向量** —— 这才有以图搜图。返回的列表与 ``chunks``
    等长、按下标对齐，正好对上 ``embed_documents`` 的 ``images`` 参数。

    **文字块一律不带图**，哪怕它的正文里有 ``![image](asset:…)`` 标记：那些图在同一次
    切分里已经有自己的图片块了（``images.py`` 为每张图单独出一条记录）。再附一次会把
    同一张图编两遍，还会把一块好端端的正文向量拽向图片 —— 那是两种记录的活，不该混着干。

    读不到本体就**跳过该图**并把 uri 记出来，不抛异常：一块图挂了不该让整批 ingest
    失败（与 ``images.py`` 里「一张图失败不牵连其余」同一个立场）。调用方要把这个
    清单打给用户看 —— 悄悄退化成「只编文字」是查不出来的。
    """
    paths: list[list[Path]] = []
    missing: list[str] = []
    for chunk in chunks:
        uris = chunk.assets if chunk.modality == "image" else []
        found: list[Path] = []
        for uri in uris:
            path = store.path_of(uri)
            if path is not None and path.is_file():
                found.append(path)
            elif uri not in missing:
                missing.append(uri)
        paths.append(found)
    return paths, missing


def _paragraphs(text: str) -> list[str]:
    """剥掉图片标记之后的非空段落。"""
    return [p.strip() for p in MARKER_RE.sub("", text).split("\n\n") if p.strip()]


def _nearest_paragraph(chunks: Sequence[Chunk], index: int, text: str, *, step: int) -> str:
    """沿 step 方向，找离标记最近的一段**真正的正文**。"""
    cursor = index
    candidate = text
    while True:
        paragraphs = _paragraphs(candidate)
        if paragraphs:
            return paragraphs[-1] if step < 0 else paragraphs[0]
        cursor += step
        if not 0 <= cursor < len(chunks):
            return ""
        candidate = chunks[cursor].text
