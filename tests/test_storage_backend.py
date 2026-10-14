"""文件后端的**统一开关** `storage.backend`（2026-10-10 加，与数据库那套同一形状）。

要钉住的是这条优先级：**域显式写的 > `storage.backend` > local**，以及四个域（上传件 /
中间产物 / 原始资产 / 会话图片）都听同一个开关。真 MinIO 要凭证与网络，属于手工验收；
这里只测装配逻辑 —— `MinioBlobStore` 的构造只校验配置，本身不联网。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config.schema import Settings
from rag.storage import (
    BlobError,
    LocalBlobStore,
    PrefixedBlobStore,
    build_blob_store,
    resolve_storage_backend,
)


class _Creds:
    """本机 `.env` 可能真配了 key —— 用例不能依赖它，所以塞一对假的。"""

    minio_access_key = "key"
    minio_secret_key = "secret"


def _settings(tmp_path: Path, *, unified: str) -> Settings:
    settings = Settings()
    settings.storage.backend = unified
    settings.storage.minio.endpoint = "127.0.0.1:9000"
    settings.storage.minio.bucket = "rag"
    settings.storage.minio.cache_dir = str(tmp_path / "cache")
    settings.ingest.uploads.dir = str(tmp_path / "uploads")
    settings.ingest.stages.dir = str(tmp_path / "stages")
    settings.ingest.assets.dir = str(tmp_path / "assets")
    settings.api.media_dir = str(tmp_path / "media")
    return settings


def _build(settings: Settings, *, backend: str, directory: str, prefix: str):
    return build_blob_store(backend=backend, settings=settings, dir=directory, prefix=prefix)


def test_the_unified_switch_pulls_all_four_domains_into_the_bucket(
    tmp_path: Path, monkeypatch
) -> None:
    """`storage.backend=minio` + 四个域都留 auto → 四个域全进桶（这就是"统一"的意思）。"""
    monkeypatch.setattr("rag.config.get_secrets", lambda: _Creds())
    settings = _settings(tmp_path, unified="minio")

    domains = (
        ("uploads", settings.ingest.uploads.backend, settings.ingest.uploads.dir, "uploads"),
        ("stages", settings.ingest.stages.backend, settings.ingest.stages.dir, "stages"),
        ("assets", settings.ingest.assets.backend, settings.ingest.assets.dir, "assets"),
        ("media", settings.api.media_backend, settings.api.media_dir, settings.api.media_prefix),
    )
    for name, backend, directory, prefix in domains:
        assert backend == "auto", f"{name} 的默认值应当是 auto"
        store = _build(settings, backend=backend, directory=directory, prefix=prefix)
        assert store.backend == "minio", f"{name} 没跟着统一开关进桶"
        assert isinstance(store, PrefixedBlobStore), f"{name} 应带自己的域前缀"
        assert store.name.startswith(f"minio:127.0.0.1:9000/rag/{prefix}"), store.name


def test_a_domain_that_pins_local_wins_over_the_unified_switch(tmp_path: Path, monkeypatch) -> None:
    """域的显式值优先：统一开关进桶，也允许把某几个域单独留在本地盘。"""
    monkeypatch.setattr("rag.config.get_secrets", lambda: _Creds())
    settings = _settings(tmp_path, unified="minio")
    settings.ingest.uploads.backend = "local"  # 只把上传件钉在本地盘

    pinned = _build(
        settings,
        backend=settings.ingest.uploads.backend,
        directory=settings.ingest.uploads.dir,
        prefix=settings.ingest.uploads.prefix,
    )
    followed = _build(
        settings,
        backend=settings.ingest.stages.backend,
        directory=settings.ingest.stages.dir,
        prefix=settings.ingest.stages.prefix,
    )

    assert isinstance(pinned, LocalBlobStore) and pinned.root == Path(settings.ingest.uploads.dir)
    assert followed.backend == "minio"


def test_auto_with_a_local_unified_switch_stays_local(tmp_path: Path) -> None:
    """默认这套（统一开关 local + 域留 auto）解析出来就是本地盘，行为与从前一致。"""
    settings = _settings(tmp_path, unified="local")

    assert resolve_storage_backend(settings, "auto") == "local"
    assert resolve_storage_backend(settings, None) == "local"
    store = _build(
        settings,
        backend=settings.ingest.assets.backend,
        directory=settings.ingest.assets.dir,
        prefix=settings.ingest.assets.prefix,
    )
    assert isinstance(store, LocalBlobStore)


def test_an_unknown_unified_value_is_rejected(tmp_path: Path) -> None:
    """统一开关写错（ceph / s3 之类）要当场报错，不能静默当 local。"""
    settings = _settings(tmp_path, unified="ceph")

    with pytest.raises(BlobError, match="未知的存储后端"):
        resolve_storage_backend(settings, "auto")
    with pytest.raises(BlobError, match="未知的存储后端"):
        _build(settings, backend="auto", directory=settings.ingest.assets.dir, prefix="assets")


def test_an_explicit_domain_value_is_matched_case_insensitively(tmp_path: Path) -> None:
    """`MINIO` / `Local` 这类大小写写法照收（历史配置里出现过）。"""
    settings = _settings(tmp_path, unified="local")

    assert resolve_storage_backend(settings, "Local") == "local"
    assert resolve_storage_backend(settings, "LOCAL") == "local"
