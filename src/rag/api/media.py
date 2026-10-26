"""对话 / 检索用的媒体（图片）落点。**与文档语料分开**。

**为什么不能落在 `data/uploads/`**：那是文档语料的根，`LocalUploadStore.list()` 按 loader
后缀扫顶层文件，而 `.png/.jpg` 本来就在 loader 注册表里 —— 丢进去就会被当成一份「文档」
列进文档页、跟着走五步流程（解析 → 核对 → 灌库）。所以媒体单开一个目录
（`api.media_dir`，默认 `data/chat/media/`）。

落盘名带内容哈希：同一张图重传就是同一个文件，不重复占盘；同名不同内容也不会互相覆盖。
这些文件**不参与任何清理**（与上传件同一个立场：清理是另一件事，见 backlog）。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

# 认得的图片后缀。**不做格式转换**：以图搜图那条 embedding 自己会解码，
# 转一道只会引入「转完跟原图不一样」的可能。
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")

# Content-Type → 后缀（浏览器给 File 的时候两边都可能有）
_BY_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/gif": ".gif",
    "image/tiff": ".tiff",
}

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class MediaError(ValueError):
    """媒体被拒（类型不认 / 超过上限）。**说清原因**。"""


def suffix_for(name: str, content_type: str = "") -> str:
    """定后缀：先看文件名，再看 Content-Type。两个都认不出来就拒 —— 猜错是下游解码报错。"""
    suffix = Path(name or "").suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return suffix
    kind = (content_type or "").split(";")[0].strip().lower()
    if kind in _BY_MIME:
        return _BY_MIME[kind]
    raise MediaError(
        f"认不出的图片类型：{name or '(没有文件名)'} / {content_type or '(没有类型)'}；"
        f"能收 {'、'.join(IMAGE_SUFFIXES)}"
    )


def save_image(settings: Any, data: bytes, *, name: str = "", content_type: str = "") -> Path:
    """把一张图落盘，返回它的路径（`/search` 的 `image_path` 要的就是这个）。

    上限 `api.media_max_image_mb`（默认 5MB）。**当场拒**，别等发到下游才报错 ——
    以图搜图那条要过一次线上 embedding，按量计费。

    落点按 `api.media_backend`：`auto`（默认，听统一开关 `storage.backend`）→ 落
    `api.media_dir`（本地盘）或进桶（键 = `media/<落盘名>`，2026-10-09）。
    **返回的一律是一个本地路径** —— 下游
    （看图的 VLM、以图搜图那条多模态 embedding）要的就是路径；minio 下那是读穿透缓存里的
    副本，桶才是真相。
    """
    if not data:
        raise MediaError("图是空的")
    limit = int(float(settings.api.media_max_image_mb) * 1_000_000)
    if len(data) > limit:
        raise MediaError(f"图超过上限 {limit / 1e6:.0f}MB（要放大就改 api.media_max_image_mb）")
    suffix = suffix_for(name, content_type)
    digest = hashlib.sha1(data).hexdigest()[:16]
    # 真名只留来给人看：落盘名带哈希，同名不同内容才不会被覆盖
    stem = Path(name or "image").stem.strip() or "image"
    stem = _CONTROL.sub("", stem)[:60]
    key = f"{stem}-{digest}{suffix}"
    from rag.storage import build_blob_store

    blobs = build_blob_store(
        backend=getattr(settings.api, "media_backend", "local"),
        settings=settings,
        dir=settings.api.media_dir,
        prefix=getattr(settings.api, "media_prefix", "media"),
    )
    blobs.put(key, data)
    path = blobs.local_path(key)
    if path is None:  # 刚 put 完就该在（local 是文件、minio 是桶）—— 不在说明后端出了问题
        raise MediaError(f"图片写进去了但取不回来：{key}")
    return path
