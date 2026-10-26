"""图片描述：把「一张图」变成一条**可检索的记录**。

文字块里只有 ``![image](asset:…)`` 这个标记 —— 检索到它，模型什么也看不到
（实测过一次：答案里写「具体对比内容以图片形式呈现（img003），文字部分未展开说明」，
没编，但也答不了）。所以每张图要有自己的一条记录：

    <标题路径>
    （空行）
    <图片描述>
    <图片地址>

与正文块**同一套字段**（``modality=image`` 区分开），标题贴开头也与正文块一致 ——
这样两种块在库里的布局是同一种，检索与展示都不必按模态分支。

三件事值得单独说：

1. **上下文**：光有图不够。一张架构图脱离「上面那段在讲什么」根本说不清，
   所以把「上边一段 + 下边一段」一起喂给 VLM（取段落不取整块，见
   ``assets.context_for_marker``）。
2. **位置**：``page`` / ``line`` 取**标记自己**的偏移（标记就落在原文里，能精确到
   图所在的第几行），``heading_path`` / ``heading_numbers`` 则继承含这张图的那一块
   —— 标题层级只有切块那一步知道。
3. **缓存**：描述是整条链路最贵的一步（每张一次 VLM 调用、按量计费）。
   键是「图片内容哈希 + 模型 + 提示词版本」—— 改一段正文不该让 45 张图全部重问。
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from rag.config.schema import Settings
from rag.contracts import Chunk, Document
from rag.ingest.assets import build_asset_store, context_for_marker, markers_in, position_at
from rag.ingest.chunkers.base import make_chunk
from rag.ingest.loaders.base import SkippedFile
from rag.storage import BlobNotFound, BlobStore, build_blob_store, resolve_storage_backend

if TYPE_CHECKING:
    from rag.providers.vision import VisionClient


def describe_images(
    documents: Sequence[Document],
    chunks: Sequence[Chunk],
    settings: Settings,
    client: VisionClient | None,
) -> tuple[list[Chunk], list[SkippedFile]]:
    """给每张图生成一条描述块。返回 ``(新块, 失败的图)``。

    在 ``_locate`` **之后**调用：标题路径要继承含这张图的那一块，而页码/行号要按标记
    自己的偏移去反查 —— 两者都得等切块与 locate 做完。
    新块**追加在文档已有块之后**，序号接着往下排（不改已有块的序号 —— 那会让
    前后两次灌库的 chunk_id 错位）。

    单张图失败**只记这条图**、不中断整份文档，也不让文档进 skipped：
    十几张图里挂一张，为它丢掉另外十几张的描述是更坏的结果。失败原因照样带出去，
    CLI 会逐条打印。
    """
    cfg = settings.ingest.images
    if not cfg.describe:
        return [], []
    if client is None:
        raise ValueError(
            "ingest.images.describe 开着，但没有传 VLM 客户端："
            "调用方要用 providers.build_vision(settings) 建一个传进来。"
        )

    store = build_asset_store(settings)
    cache = build_description_cache(settings)
    described: list[Chunk] = []
    failed: list[SkippedFile] = []
    # 钱的口子：生成这么几条描述就停。两条口径写清楚，免得下次有人以为它是 bug：
    #   * 命中缓存**也占名额** —— 它就是个「先跑几张看看效果」的闸门，不值得为省几次
    #     缓存查询把语义搞复杂
    #   * 失败的不占名额 —— 失败没有描述可言，占了名额等于「跑 3 张却只看得到 1 张结果」
    budget = cfg.max_images if cfg.max_images > 0 else None

    for doc in documents:
        if budget is not None and budget <= 0:
            break
        if not doc.images:
            continue
        owners = _owners(chunks, doc.doc_id)
        offsets = [p.offset for p in doc.positions]
        index = _next_index(chunks, doc.doc_id)
        for image in doc.images:
            if budget is not None and budget <= 0:
                break
            owner = owners.get(image.marker)
            path = store.path_of(image.asset_uri)
            if path is None or not path.is_file():
                # 资产不在了（比如换过 assets.dir）就说清楚，别去问 VLM 一张读不到的图
                failed.append(SkippedFile(path=image.asset_uri, reason=f"资产文件不存在：{path}"))
                continue
            data = path.read_bytes()

            before, after = _context(chunks, image.marker, cfg.context_chars) if owner else ("", "")
            description = _describe(cache, client, data, image.asset_uri, before, after, failed)
            if description is None:
                continue

            heading_path = owner.heading_path if owner else ""
            chunk = make_chunk(
                doc,
                index,
                _body(heading_path, description, image.marker),
                metadata={"image": image.asset_uri, "vision": client.fingerprint},
                heading_path=heading_path,
                heading_numbers=owner.heading_numbers if owner else "",
                modality="image",
            )
            chunk.page, chunk.line = _image_position(doc, image.marker, offsets, owner, image.page)
            described.append(chunk)
            index += 1
            if budget is not None:
                budget -= 1

    cache.save()
    return described, failed


def _describe(
    cache: _DescriptionCache | _BlobDescriptionCache,
    client: VisionClient,
    data: bytes,
    asset_uri: str,
    before: str,
    after: str,
    failed: list[SkippedFile],
) -> str | None:
    """缓存命中就用缓存，否则问 VLM。失败返回 None（原因记进 failed）。"""
    key = f"{hashlib.sha1(data).hexdigest()[:16]}:{client.fingerprint}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    try:
        description = client.describe(data, before=before, after=after, name=asset_uri)
    except Exception as exc:
        # 不吞异常类型：连不上、鉴权失败、返回空描述在 VisionError 的文案里已经分开写了。
        # 这里刻意接住所有异常：一张图的问题（包括客户端实现本身的意外）不该让
        # 整份文档 45 张图全废；原因原样带进 skipped，CLI 会打印出来。
        failed.append(SkippedFile(path=asset_uri, reason=f"VLM 调用失败：{exc}"))
        return None
    cache.put(key, description)
    return description


def _body(heading_path: str, description: str, marker: str) -> str:
    """描述块的正文：标题路径贴开头（与 ``chunkers.base.with_heading_path`` 同一种布局），
    然后是描述本身，最后一行是图片地址。

    图片地址用**标记**而不是裸 uri：与正文里那个标记逐字相同，``assets`` 字段
    （由 ``make_chunk`` 从正文正则抽出）自动就有值，回查图片也只需要一个字符串。
    """
    body = f"{description.strip()}\n{marker}"
    return f"{heading_path}\n\n{body}" if heading_path else body


def _context(chunks: Sequence[Chunk], marker: str, limit: int) -> tuple[str, str]:
    """上边一段 / 下边一段，各截到 ``limit`` 字符。

    截断留**靠近图片的那一头**：上边一段留尾巴、下边一段留开头 ——
    离图最近的句子才说明它在讲什么，截掉远端。
    """
    before, after = context_for_marker(chunks, marker)
    if limit <= 0:
        return "", ""
    return before[-limit:], after[:limit]


def _owners(chunks: Sequence[Chunk], doc_id: str) -> dict[str, Chunk]:
    """标记 → 含这张图的那一块。同一条标记出现在两块里（overlap）时取前面那块。"""
    out: dict[str, Chunk] = {}
    for chunk in chunks:
        if chunk.doc_id != doc_id:
            continue
        for marker in markers_in(chunk.text):
            out.setdefault(marker, chunk)
    return out


def _next_index(chunks: Sequence[Chunk], doc_id: str) -> int:
    return max((c.index for c in chunks if c.doc_id == doc_id), default=-1) + 1


def _image_position(
    doc: Document,
    marker: str,
    offsets: Sequence[int],
    owner: Chunk | None,
    image_page: int | None,
) -> tuple[int, int]:
    """图片自己的 (页码, 行号)。

    标记就落在原文里，所以直接找它的偏移再二分 —— 得到的是**图在哪一行**，
    而不是块首在哪一行（同一节里一块可以跨几十行，块首对「第几行」没有意义）。
    能定位到行就定位到行，这正是设计里要的。

    三级兜底：标记不在 ``doc.text`` 里（手搓出来的 Document 会这样）→ 用**所在块**
    的位置；连块都没有 → 退到 ``ImageAsset.page``（那是 loader 知道的页码）。
    """
    offset = doc.text.find(marker)
    if offset >= 0 and offsets:
        position = position_at(doc.positions, offsets, offset)
        return position.page, position.line
    if owner is not None:
        return owner.page, owner.line
    return image_page or 0, 0


class _DescriptionCache:
    """图片描述的磁盘缓存：``{"<图片哈希>:<模型@提示词版本>": {"text": …, "at": …}}``。

    缓存**坏了不报错**（当作空缓存重来）：它只是省钱的手段，不是数据本身 ——
    为了一个坏掉的 json 让整次 ingest 失败是更坏的结果。

    **线程安全**：IndexGraph 会把不同文件放到不同线程里跑（`--graph` + 并发 >1），
    而描述缓存是**所有文件共用**的一个 json 文件。没有锁的话两个线程各自
    「读一遍 → 加一条 → 写回去」，后写的那个会把前一个刚加的条目覆盖掉 ——
    症状是「描述明明生成过、下次还去问 VLM」，白花钱且看不出来。
    锁按文件路径分（`_LOCKS`），同一份缓存串行、不同缓存互不影响。
    """

    _LOCKS: dict[str, threading.Lock] = {}
    _LOCKS_GUARD = threading.Lock()

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = _lock_for(path)
        self._data: dict[str, dict[str, object]] | None = None
        self._dirty = False

    def get(self, key: str) -> str | None:
        with self._lock:
            entry = self._load().get(key)
        if not isinstance(entry, dict):
            return None
        text = entry.get("text")
        return text if isinstance(text, str) and text else None

    def put(self, key: str, text: str) -> None:
        with self._lock:
            self._load()[key] = {"text": text, "at": int(time.time())}
            self._dirty = True

    def save(self) -> None:
        with self._lock:
            if not self._dirty or self._data is None:
                return
            # 写之前**重新读一遍**再合并：别的线程/进程可能在这期间加过条目
            merged = {**_read_cache(self._path), **self._data}
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            self._data = merged
            self._dirty = False

    def _load(self) -> dict[str, dict[str, object]]:
        if self._data is None:
            self._data = _read_cache(self._path)
        return self._data


class _BlobDescriptionCache:
    """多 worker 共享缓存：一个键一个对象，不需要读改写整份 JSON。"""

    def __init__(self, store: BlobStore, prefix: str) -> None:
        self._store = store
        self._prefix = prefix.strip("/")

    def _key(self, key: str) -> str:
        return f"{self._prefix}/{key.replace(':', '/')}.json"

    def get(self, key: str) -> str | None:
        try:
            payload = json.loads(self._store.get_text(self._key(key)))
        except (BlobNotFound, ValueError, OSError, UnicodeDecodeError):
            return None
        text = payload.get("text") if isinstance(payload, dict) else None
        return text if isinstance(text, str) and text else None

    def put(self, key: str, text: str) -> None:
        self._store.put_text(
            self._key(key),
            json.dumps({"text": text, "at": int(time.time())}, ensure_ascii=False),
        )

    def save(self) -> None:
        return None


def build_description_cache(settings: Settings) -> _DescriptionCache | _BlobDescriptionCache:
    cfg = settings.ingest.images
    if resolve_storage_backend(settings, cfg.cache_backend) == "minio":
        store = build_blob_store(
            backend="minio",
            settings=settings,
        )
        return _BlobDescriptionCache(store, cfg.cache_prefix)
    return _DescriptionCache(Path(cfg.cache_path))


def _lock_for(path: Path) -> threading.Lock:
    """一个缓存文件一把锁（进程内）。"""
    key = str(path.resolve())
    with _DescriptionCache._LOCKS_GUARD:
        return _DescriptionCache._LOCKS.setdefault(key, threading.Lock())


def _read_cache(path: Path) -> dict[str, dict[str, object]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (ValueError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if isinstance(v, dict)}
