"""上传件的落点：**现在落本地盘，但按「以后换成文件服务器」的接口来写**。

多 worker 的那一天，上传件不能只留在某一个 worker 的本地盘上 —— 解析可能被另一个
worker 领走。所以调用方拿到的是一份 `Upload` 描述（doc_id / 原名 / 相对路径），
要读文件时走 `path_of()`：本地实现直接给路径，远端实现（MinIO / S3 / 共享盘）在那里
下载到临时目录再给路径。**换实现不动调用方**，job 里存的也是引用而不是本机绝对路径。

现在这一版是 `LocalUploadStore`，**保留不删**（使用者定的）。

**落盘名带内容哈希**（使用者 2026-10-07 定）：`<真名去扩展名>-<sha1 前 16 位><扩展名>`。
哈希用的是 `ingest.incremental.file_hash` 那一个 —— 也就是说，**文件名里的哈希就是
「解析产物是否过期」用的那把钥匙**，不是另发明的一个。三点由此而来：

1. **同一份文件重传 = 同一个文件名 = 同一个 doc_id**，静默幂等（`replaced=True`），
   stages 产物照旧有效（source_hash 一样）；
2. **同名、不同内容 = 两个文件**，互不覆盖 —— 所以两版能同时在盘上，供页面问一句
   「覆盖更新还是保留两份」（`by_display()` 就是给这个问题查的）；
3. 代价是**真名看不出来了**，所以真名存在 `_index.json` 里（`display_name`），
   页面显示它、不靠「把名字里的 -hash 切掉」去猜（真名本身就可能带 `-1a2b3c4d` 这种尾巴）。

扁平存（不按批次分目录）：doc_id 由「相对语料根的路径」算出（`loaders.base.doc_id_for`），
扁平 + 名字定死 → 重传原地 upsert，Milvus 里不会留旧块。

`_index.json` 是**附属信息**：真名 / 哈希 / 来源 URL / 抓取时间。删掉它不会丢文档，
只会让老文件退回「展示名 = 文件名」，所以读写都按「坏了就当没有」处理。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rag.ingest.loaders import LOADERS
from rag.ingest.loaders.base import data_hash
from rag.storage import BlobStore

# 附属信息的文件名。**故意以 `_` 开头、`.json` 结尾**：`list()` 只认 loader 注册表里的
# 后缀，所以它自己不会出现在文档清单里
INDEX_NAME = "_index.json"

# 文件名里只留这些：`../`、路径分隔符、控制字符一律挡在外面（中文、括号、空格留着 ——
# 用户的原文件名长什么样就什么样，我们只防「跑到上传目录外面」）。
_UNSAFE = re.compile(r"[^0-9A-Za-z._\u4e00-\u9fff()（）\[\] \-]+")


class UploadError(ValueError):
    """上传被拒（后缀不支持 / 太大 / 文件名不可用）。**明确说原因，别静默吞。**"""


@dataclass(frozen=True)
class Upload:
    """一份上传件。`doc_id` 与 ingest 用的是同一套算法（相对路径的 sha1 前 16 位）。

    `name` 是**落盘名**（带哈希），`display_name` 是**真名**（页面显示这个）。
    `sha1` 是内容的 sha1[:16]，与 `file_hash` 同一个值。
    """

    doc_id: str
    name: str
    display_name: str
    size: int
    rel_path: str
    sha1: str = ""
    # 来源（链接导入才有）。`source_url` 是用户给的那个，`final_url` 是跟完重定向之后真正的
    # 那个 —— 下载链接常常跳一次 CDN，出问题时要看得出跳到了哪
    source_url: str = ""
    final_url: str = ""
    content_type: str = ""
    # 落盘时间（老文件没有这份记录，为 0）
    created_at: int = 0
    # 附属文件（抓取页面时一起下下来的图片）的相对路径 —— 删这份文档时要一起清掉
    attachments: tuple[str, ...] = ()
    # **同一份内容**已经在了（同名同哈希）—— 如实告诉用户，别让人以为新增了一份
    replaced: bool = False


class UploadStore(Protocol):
    """上传件的落点。换文件服务器时换的就是这个实现。"""

    root: Path

    def save(
        self,
        name: str,
        data: bytes,
        *,
        source_url: str = "",
        final_url: str = "",
        content_type: str = "",
        attachments: Mapping[str, bytes] | None = None,
    ) -> Upload: ...

    def list(self) -> list[Upload]: ...

    def find(self, doc_id: str) -> Upload | None: ...

    def by_display(self, name: str) -> list[Upload]: ...

    def path_of(self, doc_id: str) -> Path | None: ...

    def delete(self, doc_id: str) -> bool: ...


def safe_name(name: str) -> str:
    """文件名消毒：只取最后一段、去掉危险字符。"""
    base = str(name).replace("\\", "/").split("/")[-1].strip()
    base = _UNSAFE.sub("_", base).lstrip(".")
    if not base:
        raise UploadError("文件名不可用（空的，或全是特殊字符）")
    return base


def supported(name: str) -> bool:
    """后缀是否受支持 —— 直接问 loader 注册表，不另立一份名单（两边会不一致）。"""
    return Path(name).suffix.lower() in LOADERS


def doc_id_for_name(name: str) -> str:
    """`doc_id` = **相对上传根的路径**的 sha1[:16]；扁平目录下相对路径就是落盘名。

    单独写成函数（而不是 `doc_id_for(path, root)`）是为了**不碰文件系统**：
    对象存储下没有"根目录"这回事，而 doc_id 必须与老数据一模一样。
    """
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]


def _stored_name(display: str, sha1: str) -> str:
    """`save()` 那条公式的反向使用：真名 + sha1 → 落盘名（导入老台账时兜底用）。"""
    path = Path(display)
    return f"{path.stem}-{sha1}{path.suffix}" if path.stem else display


def _safe_key(relative: str) -> str:
    """附属文件（HTML 里抓来的图）的键：**不许跑到上传根外面**（路径来自第三方页面）。"""
    key = Path(relative).as_posix().lstrip("/")
    if not key or ".." in Path(key).parts:
        raise UploadError(f"附属文件路径跑到上传目录外面了：{relative!r}")
    return key


def to_upload(row: Mapping[str, Any]) -> Upload:
    """台账那一行 → `Upload`（`name` 空时退回 doc_id，至少不是空白）。"""
    name = str(row.get("name") or "")
    return Upload(
        doc_id=str(row.get("doc_id") or ""),
        name=name,
        display_name=str(row.get("display_name") or name),
        size=int(row.get("size") or 0),
        rel_path=name,
        sha1=str(row.get("sha1") or ""),
        source_url=str(row.get("source_url") or ""),
        final_url=str(row.get("final_url") or ""),
        content_type=str(row.get("content_type") or ""),
        created_at=int(row.get("created_at") or 0),
        attachments=tuple(str(item) for item in (row.get("attachments") or ())),
    )


def record_of(upload: Upload) -> dict[str, Any]:
    """`Upload` → 台账那一行。`replaced` **不入库**：它是当次的返回值，不是持久状态。"""
    return {
        "doc_id": upload.doc_id,
        "name": upload.name,
        "display_name": upload.display_name,
        "sha1": upload.sha1,
        "size": upload.size,
        "source_url": upload.source_url,
        "final_url": upload.final_url,
        "content_type": upload.content_type,
        "created_at": upload.created_at,
        "attachments": list(upload.attachments),
    }


class BlobUploadStore:
    """上传件：**文件本体走 `BlobStore`，附属信息走 `upload_meta` 表**（2026-10-09）。

    `root` 是"本地根"：`local` 后端就是配置里那个上传目录；`minio` 后端是读穿透缓存目录
    （桶是真相）。**它必须与对象键那套对齐** —— 调用方拿它算 `rel_uri` / `doc_id`，
    相对路径就是落盘名，所以两个后端下 `doc_id` **完全相同**，老数据一行都不用改。

    台账为什么不再是那个 JSON 文件：多 worker 同时上传会互相覆盖（丢真名与来源），
    而现在每份上传各写自己那一行。附带的好处是 `list()` 变成一次 select ——
    minio 下也不会变成 N 次往返，`path_of()` 也从 O(N) 变成单行查。
    """

    def __init__(self, blobs: BlobStore, meta: Any, *, root: Path | str, max_bytes: int) -> None:
        self._blobs = blobs
        self._meta = meta
        self.root = Path(root)
        self.max_bytes = max_bytes

    @property
    def backend(self) -> str:
        return self._blobs.backend

    # ---- 写 ----

    def save(
        self,
        name: str,
        data: bytes,
        *,
        source_url: str = "",
        final_url: str = "",
        content_type: str = "",
        attachments: Mapping[str, bytes] | None = None,
    ) -> Upload:
        display = safe_name(name)
        if not supported(display):
            raise UploadError(
                f"不支持的文件类型 {Path(display).suffix!r}；支持：{'、'.join(sorted(LOADERS))}"
            )
        if len(data) > self.max_bytes:
            raise UploadError(
                f"{display} 有 {len(data) / 1e6:.1f}MB，超过上限 {self.max_bytes / 1e6:.0f}MB"
            )
        digest = data_hash(data)  # 与 ingest.incremental.file_hash 同值
        suffix = Path(display).suffix
        stored = f"{Path(display).stem}-{digest}{suffix}"
        # 「同一份内容已经在了」→ 问存储（local 是查文件、minio 是一次 HEAD）
        replaced = self._blobs.exists(stored)
        # local 的 `.part` + replace 与 minio 的 PUT 各自原子，所以这里直接 put 就够
        self._blobs.put(stored, data)
        # 附属文件（抓来的图片）—— 与主文件同一条路，两种后端都成立
        written = self._write_extras(attachments)
        upload = Upload(
            doc_id=doc_id_for_name(stored),
            name=stored,
            display_name=display,
            size=len(data),
            rel_path=stored,
            sha1=digest,
            source_url=source_url,
            final_url=final_url,
            content_type=content_type,
            created_at=int(time.time()),
            attachments=written,
            replaced=replaced,
        )
        self._meta.upsert([record_of(upload)])
        return upload

    # ---- 读 ----

    def list(self) -> list[Upload]:
        """台账就是清单（**不再扫目录**）：一次 select 拿完。

        排序：**新的在最前**（`created_at` 降序），同一秒的再按落盘名 —— 使用者
        2026-10-10 报的「导入的顺序和列表展示的顺序不一致」：以前按落盘名排，而落盘名是
        `<真名>-<内容哈希><后缀>`，等于**按真名字典序**，跟导入先后毫无关系。
        从老 `_index.json` 补进来、没有时间戳的那些（`created_at=0`）自然落在最后；
        同一次多选上传会挤在同一秒里，那时按名字排（稳定，但不是严格的点选顺序 ——
        要严格到毫秒就得给台账加一列序号，见 `docs/backlog.md` 第 19 条）。
        """
        rows = self._meta.all()
        return [
            to_upload(row)
            for row in sorted(
                rows.values(),
                key=lambda item: (-int(item.get("created_at") or 0), item["name"]),
            )
        ]

    def find(self, doc_id: str) -> Upload | None:
        """按 doc_id 取一份上传件（单行查，不再是全扫）。"""
        row = self._meta.get(doc_id)
        return to_upload(row) if row else None

    def by_display(self, name: str) -> list[Upload]:
        """同一个**真名**已经落过的版本（页面拿它问「覆盖还是保留两份」）。"""
        return [to_upload(row) for row in self._meta.by_display(safe_name(name))]

    def path_of(self, doc_id: str) -> Path | None:
        """按 doc_id 拿一个**本地路径**：local 是真路径，minio 是读穿透缓存里的副本
        （桶是真相，缓存可丢）。只剩单行查，不再是 O(N) 扫目录。"""
        row = self._meta.get(doc_id)
        if row is None:
            return None
        return self._blobs.local_path(str(row["name"]))

    def delete(self, doc_id: str) -> bool:
        row = self._meta.get(doc_id)
        if row is None:
            return False
        self._blobs.delete(str(row["name"]))
        # 附属文件跟着走（抓来的图片）—— 否则每抓一页就留一坨图在桶里
        for relative in row.get("attachments") or []:
            self._blobs.delete(str(relative))
        self._meta.delete([doc_id])
        return True

    # ---- 附属文件（抓来的图片）----

    def _write_extras(self, attachments: Mapping[str, bytes] | None) -> tuple[str, ...]:
        written: list[str] = []
        for relative, blob in (attachments or {}).items():
            key = _safe_key(relative)
            self._blobs.put(key, blob)
            written.append(key)
        return tuple(written)


def import_index_file(meta: Any, uploads_dir: Path | str) -> int:
    """把老的 `data/uploads/_index.json` 补进台账，返回补了几行。

    **只补空位**（`only_if_absent`）：绝不覆盖已经写好的行 —— 导入是"补历史"，
    不是"用旧文件覆盖新状态"。为什么自动做：单机升上来的人不该因为台账换了地方
    就丢掉真名与来源。

    `name`（落盘名）老 JSON 里没有：先按本地目录扫出来（那时文件还在本地），
    扫不到才按 `save()` 那条公式从「真名 + sha1」复原。
    """
    path = Path(uploads_dir) / INDEX_NAME
    if not path.is_file():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if not isinstance(payload, dict):
        return 0

    on_disk: dict[str, tuple[str, int]] = {}
    root = Path(uploads_dir)
    if root.is_dir():
        for item in root.iterdir():
            if item.is_file() and supported(item.name):
                on_disk[doc_id_for_name(item.name)] = (item.name, item.stat().st_size)

    rows: list[dict[str, Any]] = []
    for doc_id, item in payload.items():
        if not isinstance(item, dict):
            continue
        display = str(item.get("display_name") or "")
        sha1 = str(item.get("sha1") or "")
        name, size = on_disk.get(str(doc_id), (_stored_name(display, sha1), 0))
        rows.append(
            {
                "doc_id": str(doc_id),
                "name": name,
                "display_name": display or name,
                "sha1": sha1,
                "size": size,
                "source_url": str(item.get("source_url") or ""),
                "final_url": str(item.get("final_url") or ""),
                "content_type": str(item.get("content_type") or ""),
                "created_at": int(item.get("created_at") or 0),
                "attachments": [str(entry) for entry in (item.get("attachments") or [])],
            }
        )
    return int(meta.upsert(rows, only_if_absent=True))


# 老 `_index.json` **每进程只导一次**（导入是幂等的，但不必每次建 store 都查一遍）
_INDEX_IMPORTED: set[str] = set()


def build_upload_store(settings: Any) -> UploadStore:
    """按 `ingest.uploads.backend` 建：`auto`（默认，听统一开关 `storage.backend`）→
    本地盘或 MinIO 桶。

    **两种后端只有 blob 那一层不同** —— 台账一律走 `upload_meta` 表（单机 SQLite /
    多 worker PG，见 `persistence/uploads.py`）。老的 `_index.json` 在本进程第一次建
    store 时自动补一次（只补空位，文件本身不删、也不再写）。
    """
    from rag.persistence.uploads import build_upload_meta_store
    from rag.storage import LocalBlobStore, build_blob_store, resolve_storage_backend

    cfg = settings.ingest.uploads
    # auto（默认）= 听统一开关 `storage.backend`；先解析成具体的 local / minio 再分支
    backend = resolve_storage_backend(settings, getattr(cfg, "backend", "auto") or "auto")
    if backend == "local":
        blobs: Any = LocalBlobStore(cfg.dir)
        root = Path(cfg.dir)
    else:
        blobs = build_blob_store(backend=backend, settings=settings, dir=cfg.dir, prefix=cfg.prefix)
        # 读穿透缓存的根：`rel_uri(缓存里的文件, 这个根)` == 落盘名 → doc_id 与 local 一致
        root = Path(settings.storage.minio.cache_dir) / (cfg.prefix or "uploads")
    meta = build_upload_meta_store(settings)
    if meta.name not in _INDEX_IMPORTED:
        _INDEX_IMPORTED.add(meta.name)
        import_index_file(meta, cfg.dir)
    return BlobUploadStore(blobs, meta, root=root, max_bytes=int(cfg.max_mb) * 1_000_000)
