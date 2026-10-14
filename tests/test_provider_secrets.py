"""供应商密钥的**存储边界**（2026-10-09，B 方案）。

加密本身在 `persistence/secrets.py`；这里钉存储层：未传字段沿用、空串清空、
**明文不落盘（连 WAL 一起扫）**、只回指纹、审计不记 key、没主密钥就拒收。
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

from rag.persistence import secrets as S
from rag.persistence.model_settings import SqliteModelSettingStore

KEY = "sk-REAL-KEY-123"


def _store(tmp_path: Path) -> tuple[SqliteModelSettingStore, bytes]:
    store = SqliteModelSettingStore(tmp_path / "m.sqlite3")
    store.setup()
    return store, S.parse_master_key(base64.b64encode(os.urandom(32)).decode())


def test_the_key_never_lands_in_plaintext_and_only_the_fingerprint_comes_back(
    tmp_path: Path,
) -> None:
    store, master = _store(tmp_path)
    store.set_provider("dashscope", api_key=KEY, master=master, actor="t")

    row = store.providers()["dashscope"]
    assert row["has_key"] is True and row["api_key_fingerprint"] == S.fingerprint(KEY)
    assert "api_key_cipher" not in row, "密文也不回给调用方"
    # **主库 + WAL 一起扫**：只扫主库会得到假阴性（WAL 模式，实测踩过）
    blob = b"".join(p.read_bytes() for p in tmp_path.glob("m.sqlite3*"))
    assert KEY.encode() not in blob, "明文不许落盘"
    assert b"v1:" in blob, "落的是密文"


def test_untouched_fields_are_kept_and_an_empty_string_clears_the_key(tmp_path: Path) -> None:
    store, master = _store(tmp_path)
    store.set_provider("dashscope", base_url="https://x/v1", models=[{"name": "q"}], master=master)

    store.set_provider("dashscope", api_key=KEY, master=master)  # 只改 key

    row = store.providers()["dashscope"]
    assert row["base_url"] == "https://x/v1" and row["models"] == [{"name": "q"}], (
        "没传的不许被清空"
    )
    assert store.set_provider("dashscope", api_key="", master=master)["has_key"] is False
    assert store.providers()["dashscope"]["has_key"] is False, "空串 = 显式清除"


def test_the_audit_row_only_keeps_fingerprints(tmp_path: Path) -> None:
    store, master = _store(tmp_path)
    store.set_provider("dashscope", api_key=KEY, master=master, actor="t")

    [change] = store.changes(limit=1)

    assert change["slot"] == "provider:dashscope" and change["actor"] == "t"
    assert S.fingerprint(KEY) in change["new_value"]
    assert KEY not in change["new_value"], "审计里不许出现 key"


def test_without_a_master_key_the_store_refuses_to_take_a_key(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)

    with pytest.raises(S.SecretError, match="RAG_SECRET_KEY"):
        store.set_provider("openai", api_key=KEY, master=None)
