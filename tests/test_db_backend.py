"""统一选库开关：`persistence.backend`（base.yaml 声明，值放 `.env` 的 `RAG_DB_BACKEND`）。

三份关系型数据（文档注册表 + 上传件台账、模型选型、日志索引）现在听同一个开关；
优先级是「**某一类显式写的 > 统一开关 > auto**」，而 `auto` 的含义没变
（配了 url 用 pg，否则 sqlite）—— 所以不设这个环境变量时行为与从前逐字一致。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config.schema import Settings
from rag.obs.pg import build_pg_handler
from rag.persistence.backend import resolve_backend
from rag.persistence.documents import PgDocumentStore, SqliteDocumentStore, build_document_store
from rag.persistence.model_settings import build_model_setting_store
from rag.persistence.uploads import build_upload_meta_store

PG_URL = "postgresql://rag@127.0.0.1:5432/rag_test"


def _settings(
    tmp_path: Path,
    *,
    url: str = PG_URL,
    unified: str = "auto",
    documents: str = "auto",
    models: str = "auto",
) -> Settings:
    settings = Settings()
    settings.persistence.postgres.url = url
    settings.persistence.backend = unified
    settings.documents.backend = documents
    settings.models.backend = models
    settings.documents.sqlite_path = str(tmp_path / "documents.sqlite3")
    settings.models.sqlite_path = str(tmp_path / "model_settings.sqlite3")
    return settings


def test_auto_keeps_the_old_behaviour(tmp_path: Path) -> None:
    """默认 `auto`：配了 url 用 PG —— 不设环境变量时与从前一致。"""
    settings = _settings(tmp_path)

    assert isinstance(build_document_store(settings), PgDocumentStore)
    assert isinstance(build_upload_meta_store(settings), type(build_upload_meta_store(settings)))
    assert type(build_model_setting_store(settings)).__name__ == "PgModelSettingStore"
    assert isinstance(build_document_store(_settings(tmp_path, url="")), SqliteDocumentStore), (
        "没配 url 就落 sqlite"
    )


def test_the_unified_switch_forces_sqlite_even_when_a_url_is_configured(tmp_path: Path) -> None:
    """**`pg` 被换成 sqlite 的那一天**：一个开关，三份数据一起落单机文件。"""
    settings = _settings(tmp_path, unified="sqlite")

    assert isinstance(build_document_store(settings), SqliteDocumentStore)
    assert type(build_upload_meta_store(settings)).__name__ == "SqliteUploadMetaStore"
    assert type(build_model_setting_store(settings)).__name__ == "SqliteModelSettingStore"


def test_the_unified_switch_accepts_the_pg_alias(tmp_path: Path) -> None:
    settings = _settings(tmp_path, unified="pg")

    assert resolve_backend(settings) == "postgres"
    assert isinstance(build_document_store(settings), PgDocumentStore)


def test_an_explicit_per_store_choice_beats_the_unified_switch(tmp_path: Path) -> None:
    """某一类可以单独指定：`documents.backend=postgres` 压在统一开关的 sqlite 之上。"""
    settings = _settings(tmp_path, unified="sqlite", documents="postgres")

    assert isinstance(build_document_store(settings), PgDocumentStore), "文档那组听自己的"
    assert type(build_model_setting_store(settings)).__name__ == "SqliteModelSettingStore"


def test_choosing_postgres_without_a_url_fails_loudly(tmp_path: Path) -> None:
    """选了 pg 却没配 url → 当场报错，**不静默退回 sqlite**（免得你以为在用 PG）。"""
    settings = _settings(tmp_path, url="", unified="postgres")

    with pytest.raises(ValueError, match="RAG_PG_URL"):
        build_document_store(settings)
    with pytest.raises(ValueError, match="RAG_PG_URL"):
        build_upload_meta_store(settings)
    with pytest.raises(ValueError, match="RAG_PG_URL"):
        build_model_setting_store(settings)


def test_the_log_index_is_off_when_sqlite_is_chosen(tmp_path: Path) -> None:
    """日志索引只有 PG 一份实现：选了 sqlite 就不写库（文件仍是唯一真相）。"""
    assert build_pg_handler(_settings(tmp_path, unified="sqlite")) is None
    assert build_pg_handler(_settings(tmp_path, url="")) is None, "没配 url 本来就不写"


def test_an_unknown_value_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="未知的数据库开关"):
        resolve_backend(_settings(tmp_path, unified="mysql"))
