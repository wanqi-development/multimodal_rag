"""对象存储门面（2026-10-09）。**只测本地盘与装配逻辑** —— 真 MinIO 要凭证与网络，
属于手工验收（见 docs/backlog.md）。

要钉住的是"单机与多 worker 并存"的前提：默认必须还是本地盘，且键的写法两个后端一致。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag.config.schema import Settings
from rag.contracts import Document
from rag.ingest.stages import ArtifactStore
from rag.storage import (
    BlobError,
    BlobNotFound,
    LocalBlobStore,
    MinioBlobStore,
    PrefixedBlobStore,
    build_blob_store,
)


def test_local_blob_store_round_trips_bytes_and_text(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "root")

    store.put("a/b.bin", b"\x00\x01")
    store.put_text("a/c.json", '{"k": 1}')

    assert store.get("a/b.bin") == b"\x00\x01" and store.get_text("a/c.json") == '{"k": 1}'
    assert store.exists("a/b.bin") and store.size("a/b.bin") == 2
    assert store.keys() == ["a/b.bin", "a/c.json"], "键是斜杠分层的相对路径"
    assert store.local_path("a/b.bin") == tmp_path / "root" / "a" / "b.bin"
    assert store.delete("a/b.bin") is True and store.delete("a/b.bin") is False


def test_local_blob_store_treats_a_missing_key_as_not_found(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path)

    assert store.exists("nope") is False
    with pytest.raises(BlobNotFound):
        store.get("nope")


def test_delete_prefix_removes_a_whole_document_and_counts_the_files(tmp_path: Path) -> None:
    """`ArtifactStore.drop(doc_id)` 就是靠它 —— 一次收掉 `<doc_id>/` 下的全部产物。"""
    store = LocalBlobStore(tmp_path)
    store.put("d1/document.md", b"x")
    store.put("d1/chunks.json", b"{}")
    store.put("d2/document.md", b"y")

    assert store.delete_prefix("d1/") == 2
    assert store.keys() == ["d2/document.md"], "别家的产物一个都不许少"
    assert store.delete_prefix("d1/") == 0


def test_the_prefix_wrapper_hides_the_prefix_from_callers(tmp_path: Path) -> None:
    """一个桶装四个域：调用方只见 `<doc_id>/<名字>`，前缀是配置的事。"""
    inner = LocalBlobStore(tmp_path)
    store = PrefixedBlobStore(inner, "stages")

    store.put("d1/document.md", b"x")

    assert store.get("d1/document.md") == b"x"
    assert store.keys() == ["d1/document.md"], "回给调用方的键不带前缀"
    assert inner.exists("stages/d1/document.md"), "落到桶里时才加上前缀"
    assert store.delete_prefix("d1/") == 1


def test_the_default_is_still_local_so_a_single_machine_needs_nothing(tmp_path: Path) -> None:
    """**默认全 local**：单机部署不带 MinIO、不改配置，行为与以前逐字节一致。

    四个域（2026-10-10 起）默认写 `auto` = 听统一开关 `storage.backend`，而统一开关的
    默认值是 `local` —— 所以解析出来仍是本地盘，这一步是"默认不变"的那条硬保证。
    """
    settings = Settings()

    assert settings.storage.backend == "local", "统一开关默认必须是 local"
    assert settings.ingest.uploads.backend == "auto"
    assert settings.ingest.stages.backend == "auto"
    assert settings.ingest.assets.backend == "auto"
    assert settings.api.media_backend == "auto"

    store = build_blob_store(
        backend=settings.ingest.assets.backend,
        settings=settings,
        dir=str(tmp_path / "assets"),
        prefix=settings.ingest.assets.prefix,
    )
    assert isinstance(store, LocalBlobStore) and store.root == tmp_path / "assets"


def test_minio_without_endpoint_or_credentials_complains_in_plain_words(monkeypatch) -> None:
    """没配地址/桶/密钥时要**当场说清缺哪一样**，而不是等第一次上传才报 S3 错误。"""
    settings = Settings()

    with pytest.raises(BlobError, match="endpoint"):
        build_blob_store(backend="minio", settings=settings, dir="", prefix="assets")

    settings.storage.minio.endpoint = "127.0.0.1:9000"
    settings.storage.minio.bucket = "rag"

    class _NoCreds:
        """这台机器上的 `.env` 可能真配了 key —— 用例不能依赖它，所以这里塞一对空的。"""

        minio_access_key = None
        minio_secret_key = None

    monkeypatch.setattr("rag.config.get_secrets", lambda: _NoCreds())
    with pytest.raises(BlobError, match="MINIO_ACCESS_KEY"):
        build_blob_store(backend="minio", settings=settings, dir="", prefix="assets")


def test_an_unknown_backend_is_rejected_instead_of_silently_going_local() -> None:
    with pytest.raises(BlobError, match="未知的存储后端"):
        build_blob_store(backend="ceph", settings=Settings(), dir="")


# ---- 读穿透缓存：新鲜度按 ETag 比，不按大小（2026-10-10）----


class _FakeMinio:
    """只实现 `local_path()` 用到的两个调用（stat / get），不碰网络。

    etag 用内容的 md5 —— 真 MinIO 单次 PUT 也是这个，所以"**内容变了但字节数没变**"
    这种情况在用例里能如实复现。
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.gets = 0

    def _etag(self, key: str) -> str:
        return hashlib.md5(self.objects[key]).hexdigest()

    def stat_object(self, bucket: str, key: str) -> SimpleNamespace:
        if key not in self.objects:
            from minio.error import S3Error

            raise S3Error(None, "NoSuchKey", "Object does not exist", key, "req", "host")
        return SimpleNamespace(etag=self._etag(key), size=len(self.objects[key]))

    def get_object(self, bucket: str, key: str) -> SimpleNamespace:
        self.gets += 1
        payload = self.objects[key]
        return SimpleNamespace(read=lambda: payload, close=lambda: None, release_conn=lambda: None)

    # 下面两个只有"写"那条路会用到（`ArtifactStore.save_document` 等）
    def bucket_exists(self, bucket: str) -> bool:
        return True

    def put_object(self, bucket: str, key: str, data, length: int) -> None:
        self.objects[key] = data.read()


def _minio_store(tmp_path: Path) -> tuple[MinioBlobStore, _FakeMinio]:
    store = MinioBlobStore(
        endpoint="127.0.0.1:9000",
        bucket="rag",
        access_key="a",
        secret_key="b",
        cache_dir=tmp_path / "cache",
    )
    fake = _FakeMinio()
    store.__dict__["_client"] = fake  # `_client` 是 cached_property，塞进 __dict__ 就绕开了
    return store, fake


def test_a_same_size_edit_in_the_bucket_invalidates_the_cache(tmp_path: Path) -> None:
    """**「标题核对结果不生效」的回归**（2026-10-10 实测那一份）。

    缓存旧判据是"大小一致"，而人改清单的典型改动（`已核对: 否 → 是`、层级 `2 → 3`）
    **字节数完全相同** —— 缓存永远算命中，应用一直读到旧清单（桶里 `confirmed=True`、
    读出来 `False`）。现在按 ETag 比，同尺寸改动照样能发现。
    """
    store, fake = _minio_store(tmp_path)
    fake.objects["stages/d1/outline.tsv"] = "# 已核对: 否\n".encode()

    first = store.local_path("stages/d1/outline.tsv")
    assert first is not None and first.read_text(encoding="utf-8").endswith("否\n")
    assert fake.gets == 1

    edited = "# 已核对: 是\n".encode()
    assert len(edited) == len(fake.objects["stages/d1/outline.tsv"]), "这条用例要的就是同字节数"
    fake.objects["stages/d1/outline.tsv"] = edited

    second = store.local_path("stages/d1/outline.tsv")
    assert second is not None and second.read_text(encoding="utf-8").endswith("是\n")
    assert fake.gets == 2, "同一个键、同样大小、内容变了 —— 必须重新拉"


def test_an_unchanged_object_comes_from_the_cache(tmp_path: Path) -> None:
    """没变就别重复下：桶是真相，但每次读都下一次就没有缓存的意义了。"""
    store, fake = _minio_store(tmp_path)
    fake.objects["stages/d1/chunks.json"] = b'{"k": 1}'

    for _ in range(3):
        assert store.local_path("stages/d1/chunks.json") is not None

    assert fake.gets == 1


def test_a_key_that_is_not_in_the_bucket_is_none(tmp_path: Path) -> None:
    """「这份产物还没有」是正常状态：给 None，不抛（详情页 500 那条的回归）。"""
    store, _ = _minio_store(tmp_path)

    assert store.local_path("stages/nope/outline.tsv") is None


def test_the_reviewed_outline_is_what_the_app_reads(tmp_path: Path) -> None:
    """把用户看到的现象钉在端到端那一层：桶里勾了「已核对」之后，`load_outline` 跟着变。"""
    store, fake = _minio_store(tmp_path)
    artifacts = ArtifactStore(PrefixedBlobStore(store, "stages"))
    doc = Document(doc_id="d1", source_uri="scan.png", text="# 第一章 概述\n\n正文。")
    artifacts.save_document(doc, source_hash="h", parse="p")

    assert artifacts.load_outline("d1", source_hash="h", parse="p").confirmed is False

    key = "stages/d1/outline.tsv"
    fake.objects[key] = fake.objects[key].replace("已核对: 否".encode(), "已核对: 是".encode())

    outline = artifacts.load_outline("d1", source_hash="h", parse="p")
    assert outline is not None and outline.confirmed is True
    assert (
        artifacts.outline_fingerprint("d1")
        == hashlib.sha1(f"{outline.levels}|{outline.max_level}".encode()).hexdigest()[:16]
    ), "指纹也要跟着新清单算"
