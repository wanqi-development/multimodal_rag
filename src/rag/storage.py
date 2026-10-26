"""对象存储门面：**一份代码，单机走本地盘、多 worker 走 MinIO**（2026-10-09）。

为什么要有这一层：多 worker 下 worker A 上传的文件，worker B 得读得到 —— 本地盘做不到，
所以上传件 / 产物 / 图片 / 会话图片这四个落点要能换成 MinIO 桶。使用者定的三件事：

1. **每个域各自一个 `backend: local | minio` 开关**（形态 A）：单机不装 MinIO 也能跑，
   多 worker 只把那几个开关改掉。**默认全 `local`，所以今天的行为一个字都不变。**
2. **用官方 `minio` SDK**（不走 boto3，也不走 FUSE 挂载）。
3. **键的写法两个后端完全一样**：`<域前缀>/<doc_id>/<名字>`。local 下根目录就是域目录，
   minio 下靠前缀隔开 —— 所以 `doc_id`（相对路径的 sha1）与正文里的 `asset:<doc_id>/x.png`
   标记在两个后端下**逐字节一致**，不用迁移数据、也不用改前端。

一个刻意的设计：`local_path()` 让"只认 Path 的消费方"（loader / PyMuPDF / OCR / FileResponse）
继续可用 —— local 直接给真路径；minio **读穿透到本地缓存**（一次 HEAD 对大小，命中就不再下载）。
缓存只是缓存：桶是真相，删掉缓存只会让它下次再下一次。

凭证不进这里：`access_key` / `secret_key` 从 `Secrets`（`.env`）取，与其它机密同一条规矩。
"""

from __future__ import annotations

import contextlib
import io
import os
import threading
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class BlobError(RuntimeError):
    """存储层的错误（都带人话原因：桶不在、配少了、连不上）。"""


class BlobNotFound(BlobError):
    """这个键不存在。调用方靠它区分"没有"与"读不到"。"""


@runtime_checkable
class BlobStore(Protocol):
    """键 → 字节的读写。**键是斜杠分层的字符串**（两个后端同一套写法）。"""

    backend: str
    name: str

    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def size(self, key: str) -> int: ...
    def delete(self, key: str) -> bool: ...
    def delete_prefix(self, prefix: str) -> int: ...
    def keys(self, prefix: str = "") -> list[str]: ...
    def local_path(self, key: str) -> Path | None: ...


def _text(data: bytes) -> str:
    return data.decode("utf-8")


def _etag_sidecar(cached: Path) -> Path:
    """本地副本旁边那个「这份是桶里哪个版本」的记号文件（内容就是 ETag）。

    放在缓存文件同目录、文件名后缀追加 `.etag`：整目录删掉时它跟着走，不会留孤儿。
    """
    return cached.with_suffix(f"{cached.suffix}.etag")


def _read_etag(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _write_etag(path: Path, etag: str) -> None:
    # 写不进去只意味着下次多下一次 —— 缓存本来就是可有可无的
    with contextlib.suppress(OSError):
        path.write_text(etag, encoding="utf-8")


class LocalBlobStore:
    """本地盘。**这就是今天那份实现**，只是被包成了键 → 文件的写法。"""

    backend = "local"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.name = f"local:{self.root}"

    def _path(self, key: str) -> Path:
        # 键是我们自己拼的（doc_id 是 sha1、名字过过 safe_name），不做 `..` 兜底：
        # 真要防的是外部输入直接进 key —— 那一步在各自的 store 里就挡住了。
        return self.root / key

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 先写 `.part` 再 replace：中途断掉不会留半个文件（与上传件同一个做法）
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise BlobNotFound(f"没有这个文件：{path}")
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def size(self, key: str) -> int:
        path = self._path(key)
        if not path.is_file():
            raise BlobNotFound(f"没有这个文件：{path}")
        return path.stat().st_size

    def delete(self, key: str) -> bool:
        path = self._path(key)
        if not path.is_file():
            return False
        path.unlink()
        # **空目录随手收掉**：对象存储里本就没有"目录"这回事，本地也照这个语义来 ——
        # 否则抓一页 HTML 就留一个 `<真名>_files/` 空壳，删完文档还挂在那儿（实测踩过）。
        parent = path.parent
        while parent != self.root and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
        return True

    def delete_prefix(self, prefix: str) -> int:
        """删一个"目录"。返回删了几个文件（空目录顺手收掉）。"""
        target = self._path(prefix)
        if not target.is_dir():
            return 0
        count = sum(1 for item in target.rglob("*") if item.is_file())
        import shutil

        shutil.rmtree(target, ignore_errors=True)
        return count

    def keys(self, prefix: str = "") -> list[str]:
        base = self._path(prefix)
        root = base if base.is_dir() else self.root
        if not root.is_dir():
            return []
        out = [item.relative_to(self.root).as_posix() for item in root.rglob("*") if item.is_file()]
        return sorted(out)

    def local_path(self, key: str) -> Path | None:
        """本地盘：键就是路径，直接给（不复制、不缓存）。**没有这个键就给 None。**

        「没有」与「有」必须两个后端说法一致（2026-10-10）：调用方要区分的是
        「这个产物存不存在」，而不是「用的是哪个后端」—— 以前只有本地盘能安静地说
        「没有」，minio 那边直接抛 `BlobNotFound`，于是同一段代码换个后端就 500。
        """
        path = self._path(key)
        return path if path.is_file() else None

    def get_text(self, key: str) -> str:
        return _text(self.get(key))

    def put_text(self, key: str, text: str) -> None:
        self.put(key, text.encode("utf-8"))


class MinioBlobStore:
    """MinIO 桶（S3 兼容）。**懒 import**：单机不装 `minio` 也起得来。"""

    backend = "minio"
    # 每个 key 一次 HEAD 就能判「缓存还新不新」，够用且便宜
    _HEAD_TIMEOUT_S = 5

    def __init__(
        self,
        *,
        endpoint: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
        region: str = "",
        cache_dir: str | Path = "./data/cache/blobs",
    ) -> None:
        if not endpoint or not bucket:
            raise BlobError("配了 backend=minio，但 `storage.minio.endpoint` 或 `bucket` 是空的")
        if not access_key or not secret_key:
            raise BlobError(
                "MinIO 的 `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` 没配（机密只从 .env 读）"
            )
        self.endpoint = endpoint
        self.bucket = bucket
        self.secure = secure
        self.region = region
        self.cache_dir = Path(cache_dir)
        self.name = f"minio:{endpoint}/{bucket}"
        self._access_key = access_key
        self._secret_key = secret_key
        self._bucket_checked = False
        self._lock = threading.Lock()

    @cached_property
    def _client(self) -> Any:
        from minio import Minio

        return Minio(
            self.endpoint,
            access_key=self._access_key,
            secret_key=self._secret_key,
            secure=self.secure,
            region=self.region or None,
        )

    def _ensure_bucket(self) -> None:
        """桶不在就建一个（幂等、每进程一次）。建桶是写操作，但它是**自举**：
        没这一步，第一次上传会以一堆看不懂的 S3 错误结束。"""
        if self._bucket_checked:
            return
        with self._lock:
            if self._bucket_checked:
                return
            from minio.error import S3Error

            try:
                if not self._client.bucket_exists(self.bucket):
                    self._client.make_bucket(self.bucket, location=self.region or None)
            except S3Error as exc:  # pragma: no cover —— 要真桶才跑到
                raise BlobError(
                    f"连不上 MinIO / 建不了桶 {self.bucket}：{exc.code} {exc.message}"
                ) from exc
            self._bucket_checked = True

    def put(self, key: str, data: bytes) -> None:
        from minio.error import S3Error

        self._ensure_bucket()
        try:
            # S3 的 PUT 本身是原子的：不需要 `.part` 那套
            self._client.put_object(self.bucket, key, io.BytesIO(data), length=len(data))
        except S3Error as exc:  # pragma: no cover —— 要真桶才跑到
            raise BlobError(f"写 {key} 到 MinIO 失败：{exc.code} {exc.message}") from exc

    def get(self, key: str) -> bytes:
        from minio.error import S3Error

        try:
            response = self._client.get_object(self.bucket, key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except S3Error as exc:  # pragma: no cover —— 要真桶才跑到
            if exc.code in ("NoSuchKey", "NoSuchObject"):
                raise BlobNotFound(f"桶里没有这个键：{key}") from exc
            raise BlobError(f"读 {key} 失败：{exc.code} {exc.message}") from exc

    def exists(self, key: str) -> bool:
        try:
            self._client.stat_object(self.bucket, key)
        except Exception:  # noqa: BLE001 —— 没有 / 桶还没建，都当"没有"
            return False
        return True

    def size(self, key: str) -> int:
        from minio.error import S3Error

        try:
            return int(self._client.stat_object(self.bucket, key).size)
        except S3Error as exc:  # pragma: no cover
            raise BlobNotFound(f"桶里没有这个键：{key}") from exc

    def _remote_etag(self, key: str) -> str | None:
        """桶里这个键的 ETag；**没有这个键 → None**。

        缓存新鲜度就靠它比（`local_path`）。不能比大小：`stages/<doc_id>/outline.tsv`
        是**人就地编辑**的产物，`已核对: 否 → 是`、层级 `2 → 3` 这类改动**字节数一模一样**
        —— 按大小比，缓存永远算命中，于是核对结果永远读不出来（实测 2026-10-10：
        桶里 `confirmed=True`、应用读到 `False`）。
        """
        from minio.error import S3Error

        try:
            stat = self._client.stat_object(self.bucket, key)
        except S3Error as exc:
            if exc.code in ("NoSuchKey", "NoSuchObject"):
                return None
            raise BlobError(f"看 {key} 的状态失败：{exc.code} {exc.message}") from exc
        return str(getattr(stat, "etag", "") or "")

    def delete(self, key: str) -> bool:
        if not self.exists(key):
            return False
        self._client.remove_object(self.bucket, key)
        return True

    def delete_prefix(self, prefix: str) -> int:
        names = [item.object_name for item in self._iter(prefix) if item.object_name]
        if not names:
            return 0
        from minio.deleteobjects import DeleteObject

        errors = list(
            self._client.remove_objects(self.bucket, [DeleteObject(name) for name in names])
        )
        if errors:  # pragma: no cover —— 要真桶才跑到
            raise BlobError(f"删 {prefix} 时出错：{errors[0].code} {errors[0].message}")
        return len(names)

    def keys(self, prefix: str = "") -> list[str]:
        return sorted(item.object_name for item in self._iter(prefix) if item.object_name)

    def _iter(self, prefix: str) -> Iterator[Any]:
        # recursive=True：`stages/<doc_id>/` 底下还有一层文件，非递归只会列出"目录"
        return self._client.list_objects(self.bucket, prefix=prefix, recursive=True)

    def local_path(self, key: str) -> Path | None:
        """读穿透到本地缓存：桶是真相，缓存只是让只认 Path 的消费方能用。

        **命中判据是 ETag**（2026-10-10 改，一次 HEAD 就带回来了）：本地副本旁边放一个
        `<文件>.etag`，记着这份副本对的是桶里哪个版本，ETag 一变就重新拉。
        以前比的是**大小一致**，理由是"键里带内容哈希（上传件名、`doc_id`）"—— 那条对
        上传件成立，但 `outline.tsv` 是**人**就地改的，键不带哈希、改动还常常同字节数，
        于是核对结果永远读不出来（见 `_remote_etag` 里的实测记录）。

        **桶里没有这个键 → None**（不再抛 `BlobNotFound`，2026-10-10 修）：调用方问的是
        「这个产物在不在」，而"不在"是正常状态（刚上传、还没解析、解析失败、老文档没生成过
        清单）。以前这里抛异常，而本地盘那边只是返回一个不存在的 Path —— 同一段调用代码
        换个后端就从"照常往下走"变成 500（实测：详情页 `GET /documents/{id}` 必崩在
        `outline_fingerprint` 上）。
        """
        target = self.cache_dir / key
        remote_etag = self._remote_etag(key)
        if remote_etag is None:
            return None
        if target.is_file() and _read_etag(_etag_sidecar(target)) == remote_etag:
            return target
        data = self.get(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, target)
        _write_etag(_etag_sidecar(target), remote_etag)
        return target

    def get_text(self, key: str) -> str:
        return _text(self.get(key))

    def put_text(self, key: str, text: str) -> None:
        self.put(key, text.encode("utf-8"))


class PrefixedBlobStore:
    """给内层加一层键前缀：**一个桶装四个域**（`uploads/` `stages/` `assets/` `media/`）。"""

    def __init__(self, inner: BlobStore, prefix: str) -> None:
        self._inner = inner
        self._prefix = prefix.strip("/")
        self.backend = inner.backend
        self.name = f"{inner.name}/{self._prefix}" if self._prefix else inner.name

    def _key(self, key: str) -> str:
        return f"{self._prefix}/{key}" if self._prefix else key

    def put(self, key: str, data: bytes) -> None:
        self._inner.put(self._key(key), data)

    def get(self, key: str) -> bytes:
        return self._inner.get(self._key(key))

    def exists(self, key: str) -> bool:
        return self._inner.exists(self._key(key))

    def size(self, key: str) -> int:
        return self._inner.size(self._key(key))

    def delete(self, key: str) -> bool:
        return self._inner.delete(self._key(key))

    def delete_prefix(self, prefix: str) -> int:
        return self._inner.delete_prefix(self._key(prefix) or self._prefix)

    def keys(self, prefix: str = "") -> list[str]:
        full = self._key(prefix) if prefix else f"{self._prefix}/" if self._prefix else ""
        cut = len(self._prefix) + 1 if self._prefix else 0
        return [name[cut:] for name in self._inner.keys(full)]

    def local_path(self, key: str) -> Path | None:
        return self._inner.local_path(self._key(key))

    def get_text(self, key: str) -> str:
        return _text(self.get(key))

    def put_text(self, key: str, text: str) -> None:
        self.put(key, text.encode("utf-8"))


STORAGE_BACKENDS = ("local", "minio")


def resolve_storage_backend(settings: Any, want: str | None = None) -> str:
    """定这一份文件落哪。返回 `"local"` 或 `"minio"`（**不会再是 auto**）。

    **优先级：域自己显式写的 > 统一开关 `storage.backend` > local**（2026-10-10 统一）。

    * 统一开关在 base.yaml 里声明成 `${RAG_STORAGE_BACKEND:-local}`，值放 `.env`；
      它管四个域：`ingest.uploads` / `ingest.stages` / `ingest.assets` / `api.media`。
    * 那四个域默认写 `auto`（= 听统一开关）；想把某一个域单独钉死（比如产物进桶、上传件留
      本地盘），把它写回 `local` / `minio` 即可。
    * `local` 仍是统一开关的默认值：不设任何东西时，单机行为与从前逐字节一致。
    """
    value = str(want or "auto").strip().lower()
    if value == "auto":
        unified = getattr(getattr(settings, "storage", None), "backend", "local")
        value = str(unified or "local").strip().lower()
    if value not in STORAGE_BACKENDS:
        raise BlobError(f"未知的存储后端 {value!r}；可选：local、minio（或 auto = 听统一开关）")
    return value


def build_blob_store(
    *,
    backend: str,
    settings: Any,
    dir: str = "",
    prefix: str = "",
) -> BlobStore:
    """按**选库开关**建：`local` → 本地盘，`minio` → 桶 + 该域前缀。

    `backend` 传 `auto`（**四个域的默认值**）就听统一开关 `storage.backend`
    （base.yaml 里 `${RAG_STORAGE_BACKEND:-local}`，值放 .env）；某一域显式写
    `local` / `minio` 就听它自己 —— 见 `resolve_storage_backend()`。
    **默认 local 是刻意的**：单机部署不带 MinIO、不改配置，行为与今天逐字节一致。
    参数显式传（不是把 cfg 塞进来）：四个域的字段名不一样（三个是 `dir`，媒体叫
    `media_dir`），显式传反而少一层猜。
    """
    backend = resolve_storage_backend(settings, backend)
    if backend == "local":
        return LocalBlobStore(dir)
    from rag.config import get_secrets

    mini = settings.storage.minio
    secrets = get_secrets()
    inner = MinioBlobStore(
        endpoint=mini.endpoint,
        bucket=mini.bucket,
        access_key=secrets.minio_access_key or "",
        secret_key=secrets.minio_secret_key or "",
        secure=mini.secure,
        region=mini.region,
        cache_dir=mini.cache_dir,
    )
    return PrefixedBlobStore(inner, prefix) if prefix else inner
